"""
Implements a Variational Sparse Autoencoder with Gaussian Mixture Prior.
Designed to better model correlated and anti-correlated feature pairs.
"""
import torch as t
from typing import Optional, List, Dict, Tuple
from collections import namedtuple

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn, ConstrainedAdam
from ..dictionary import Dictionary
from ..config import DEBUG


class VSAEMixDict(Dictionary, t.nn.Module):
    """
    Variational Sparse Autoencoder with Gaussian Mixture Prior
    
    This extends the isotropic Gaussian VSAE by using a mixture of Gaussians
    as the prior distribution to better model correlated and anti-correlated
    feature pairs.
    
    The prior means are structured as follows:
    - Correlated pairs: Both features have positive means
    - Anti-correlated pairs: One feature has positive mean, the other negative
    - Uncorrelated features: Zero mean (standard prior)
    """
    
    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        
        # Initialize encoder and decoder parameters
        self.W_enc = t.nn.Parameter(t.empty(activation_dim, dict_size))
        self.b_enc = t.nn.Parameter(t.zeros(dict_size))
        self.W_dec = t.nn.Parameter(t.empty(dict_size, activation_dim))
        self.b_dec = t.nn.Parameter(t.zeros(activation_dim))
        
        # Initialize with Kaiming uniform
        t.nn.init.kaiming_uniform_(self.W_enc)
        t.nn.init.kaiming_uniform_(self.W_dec)
        
        # Normalize decoder weights
        self.normalize_decoder()
        
    def encode(self, x, output_log_var=False):
        """
        Encode a vector x in the activation space.
        
        Args:
            x: Input tensor
            output_log_var: Whether to output log_var (always False without variance params)
            
        Returns:
            Encoded features (and log_var if output_log_var=True)
        """
        x_cent = x - self.b_dec
        z = t.nn.functional.relu(x_cent @ self.W_enc + self.b_enc)
        
        if output_log_var:
            # Fixed variance
            log_var = t.zeros_like(z)
            return z, log_var
        return z
    
    def decode(self, f):
        """
        Decode a dictionary vector f
        
        Args:
            f: Dictionary vector
            
        Returns:
            Decoded vector
        """
        return f @ self.W_dec + self.b_dec
    
    def forward(self, x, output_features=False):
        """
        Forward pass through the autoencoder.
        """
        f = self.encode(x)
        x_hat = self.decode(f)
        
        if output_features:
            return x_hat, f
        else:
            return x_hat
    
    @t.no_grad()
    def normalize_decoder(self):
        """Normalize decoder weights to have unit norm"""
        norm = t.norm(self.W_dec, dim=1, keepdim=True)
        self.W_dec.data = self.W_dec.data / norm.clamp(min=1e-6)
    
    def scale_biases(self, scale: float):
        """Scale biases by a factor"""
        self.b_dec.data *= scale
        self.b_enc.data *= scale
    
    @classmethod
    def from_pretrained(cls, path, device=None):
        """Load a pretrained autoencoder from a file."""
        state_dict = t.load(path)
        activation_dim, dict_size = state_dict["W_enc"].shape
        autoencoder = cls(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class VSAEMixVarDict(VSAEMixDict):
    """VSAEMixDict with learned variance parameters"""
    
    def __init__(self, activation_dim, dict_size):
        super().__init__(activation_dim, dict_size)
        
        # Add variance parameters
        self.W_enc_var = t.nn.Parameter(t.empty(activation_dim, dict_size))
        self.b_enc_var = t.nn.Parameter(t.zeros(dict_size))
        
        # Initialize
        t.nn.init.kaiming_uniform_(self.W_enc_var)
    
    def encode(self, x, output_log_var=False):
        """
        Encode with learned variance.
        """
        x_cent = x - self.b_dec
        z = t.nn.functional.relu(x_cent @ self.W_enc + self.b_enc)
        
        if output_log_var:
            log_var = t.nn.functional.relu(x_cent @ self.W_enc_var + self.b_enc_var)
            return z, log_var
        return z
    
    def scale_biases(self, scale: float):
        """Scale all biases"""
        super().scale_biases(scale)
        self.b_enc_var.data *= scale


class VSAEMixTrainer(SAETrainer):
    """
    Variational Sparse Autoencoder with Gaussian Mixture Prior.
    
    This extends the isotropic Gaussian VSAE by using a mixture of Gaussians
    as the prior distribution to better model correlated and anti-correlated
    feature pairs.
    """
    
    def __init__(self,
                 steps: int,                    # total number of steps to train for
                 activation_dim: int,           # dimension of input activations
                 dict_size: int,                # size of the dictionary
                 layer: int,                    # layer of the transformer model
                 lm_name: str,                  # name of the language model
                 kl_coeff: float = 3e-4,        # coefficient for KL divergence term (like l1_penalty)
                 warmup_steps: int = 1000,      # lr warmup period at start of training and after each resample
                 sparsity_warmup_steps: Optional[int] = 2000,  # sparsity warmup period at start of training
                 decay_start: Optional[int] = None,  # decay learning rate after this many steps
                 resample_steps: Optional[int] = 3000,  # how often to resample neurons
                 var_flag: int = 0,             # whether to learn variance (0: fixed, 1: learned)
                 n_correlated_pairs: int = 0,   # number of correlated feature pairs
                 n_anticorrelated_pairs: int = 0,  # number of anticorrelated feature pairs
                 correlation_prior_scale: float = 1.0,  # scale for correlation prior
                 seed: Optional[int] = None,
                 device = None,
                 wandb_name: Optional[str] = 'VSAEMixTrainer',
                 submodule_name: Optional[str] = None,
                ):
        super().__init__(seed)
        
        # Set model parameters
        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        
        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)
            
        # Initialize dictionary based on var_flag
        if var_flag == 0:
            self.ae = VSAEMixDict(activation_dim, dict_size)
        else:
            self.ae = VSAEMixVarDict(activation_dim, dict_size)
            
        self.lr = 1e-4  # Default learning rate
        self.kl_coeff = kl_coeff
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name
        self.var_flag = var_flag
        self.n_correlated_pairs = n_correlated_pairs
        self.n_anticorrelated_pairs = n_anticorrelated_pairs
        self.correlation_prior_scale = correlation_prior_scale
        
        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.ae.to(self.device)
        
        # Initialize the prior means based on correlation structure
        self.prior_means = self._initialize_prior_means()
        
        # For dead neuron detection and resampling
        self.resample_steps = resample_steps
        if self.resample_steps is not None:
            # How many steps since each neuron was last activated?
            self.steps_since_active = t.zeros(dict_size, dtype=t.long).to(self.device)
        else:
            self.steps_since_active = None
        
        # Create optimizer with constraints on decoder weights
        self.optimizer = ConstrainedAdam(self.ae.parameters(), [self.ae.W_dec], lr=self.lr)
        
        # Create learning rate and sparsity schedules
        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps, sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)
        
        # Add tracking metrics for logging
        self.logging_parameters = ["kl_coeff", "var_flag", "n_correlated_pairs", "n_anticorrelated_pairs"]
    
    def _initialize_prior_means(self) -> t.Tensor:
        """
        Initialize the prior means for the latent variables based on 
        the specified correlation structure.
        """
        means = t.zeros(self.ae.dict_size, device=self.device)
        scale = self.correlation_prior_scale
        
        # Process correlated pairs
        for i in range(self.n_correlated_pairs):
            # Both features in a correlated pair have positive means
            means[2*i] = scale
            means[2*i + 1] = scale
        
        # Process anticorrelated pairs
        offset = 2 * self.n_correlated_pairs
        for i in range(self.n_anticorrelated_pairs):
            # First feature has positive mean, second has negative mean
            means[offset + 2*i] = scale
            means[offset + 2*i + 1] = -scale
        
        return means
    
    def reparameterize(self, mu: t.Tensor, log_var: t.Tensor) -> t.Tensor:
        """
        Apply the reparameterization trick:
        z = mu + eps * sigma, where eps ~ N(0, 1)
        """
        std = t.exp(0.5 * log_var)
        eps = t.randn_like(std)
        return mu + eps * std
    
    def compute_kl_divergence(self, mu: t.Tensor, log_var: t.Tensor) -> t.Tensor:
        """
        Compute KL divergence between q(z|x) = N(mu, sigma^2) and 
        the mixture prior distribution p(z) with structured means.
        """
        # Expand prior_means to match batch dimension
        prior_means = self.prior_means.expand_as(mu)
        
        # Calculate KL divergence with non-zero mean prior
        # KL = 0.5 * (log(1/sigma^2) + sigma^2 + (mu-prior_mu)^2 - 1)
        kl = 0.5 * (
            -log_var + 
            log_var.exp() + 
            (mu - prior_means).pow(2) - 
            1
        )
        
        return kl.sum(-1).mean()
    
    def resample_neurons(self, deads, activations):
        """
        Resample dead neurons with high loss activations
        """
        with t.no_grad():
            if deads.sum() == 0: 
                return
                
            print(f"resampling {deads.sum().item()} neurons")

            # Compute loss for each activation
            losses = (activations - self.ae(activations)).norm(dim=-1)

            # Sample input to create encoder/decoder weights from
            n_resample = min([deads.sum(), losses.shape[0]])
            indices = t.multinomial(losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]

            # Get norm of living neurons
            alive_norm = self.ae.W_enc[:, ~deads].norm(dim=0).mean()

            # Resample first n_resample dead neurons
            deads[deads.nonzero()[n_resample:]] = False
            self.ae.W_enc.data[:, deads] = (sampled_vecs - self.ae.b_dec).T * alive_norm * 0.2
            self.ae.W_dec.data[deads, :] = ((sampled_vecs - self.ae.b_dec) / (sampled_vecs - self.ae.b_dec).norm(dim=-1, keepdim=True)).T
            self.ae.b_enc.data[deads] = 0.
            
            # Also reset variance params if needed
            if self.var_flag == 1 and hasattr(self.ae, 'W_enc_var'):
                self.ae.W_enc_var.data[:, deads] = 0.
                self.ae.b_enc_var.data[deads] = 0.

            # Reset Adam parameters for dead neurons
            self._reset_optimizer_stats(deads)
    
    def _reset_optimizer_stats(self, dead_mask):
        """Reset optimizer state for resampled neurons"""
        state_dict = self.optimizer.state_dict()['state']
        
        # Loop through the parameters in the optimizer
        params_to_reset = []
        for param_idx, param in enumerate(self.optimizer.param_groups[0]['params']):
            if param_idx not in state_dict:
                continue
            
            # Check which parameter this is and reset accordingly
            if param.shape == self.ae.W_enc.shape:
                params_to_reset.append((param_idx, 'W_enc', dead_mask))
            elif param.shape == self.ae.b_enc.shape:
                params_to_reset.append((param_idx, 'b_enc', dead_mask))
            elif param.shape == self.ae.W_dec.shape:
                params_to_reset.append((param_idx, 'W_dec', dead_mask))
            elif hasattr(self.ae, 'W_enc_var') and param.shape == self.ae.W_enc_var.shape:
                params_to_reset.append((param_idx, 'W_enc_var', dead_mask))
            elif hasattr(self.ae, 'b_enc_var') and param.shape == self.ae.b_enc_var.shape:
                params_to_reset.append((param_idx, 'b_enc_var', dead_mask))
        
        # Reset optimizer stats
        for param_idx, param_type, mask in params_to_reset:
            if param_type == 'W_enc' or param_type == 'W_enc_var':
                state_dict[param_idx]['exp_avg'][:, mask] = 0.
                state_dict[param_idx]['exp_avg_sq'][:, mask] = 0.
            elif param_type == 'b_enc' or param_type == 'b_enc_var':
                state_dict[param_idx]['exp_avg'][mask] = 0.
                state_dict[param_idx]['exp_avg_sq'][mask] = 0.
            elif param_type == 'W_dec':
                state_dict[param_idx]['exp_avg'][mask, :] = 0.
                state_dict[param_idx]['exp_avg_sq'][mask, :] = 0.
    
    def loss(self, x, step: int, logging=False, **kwargs):
        """
        Compute the VSAE loss with mixture prior
        """
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Get mean and log_var from encoder
        if self.var_flag == 0:
            mu = self.ae.encode(x)
            log_var = t.zeros_like(mu)
        else:
            mu, log_var = self.ae.encode(x, output_log_var=True)
        
        # Sample from the latent distribution using reparameterization trick
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x_hat = self.ae.decode(z)
        
        # Compute losses
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        recon_loss = ((x - x_hat) ** 2).sum(dim=-1).mean()
        kl_loss = self.compute_kl_divergence(mu, log_var)
        
        # Track active features for resampling
        if self.steps_since_active is not None:
            # Update steps_since_active
            deads = (z == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
        
        loss = recon_loss + self.kl_coeff * sparsity_scale * kl_loss
        
        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, z,
                {
                    'l2_loss': l2_loss.item(),
                    'mse_loss': recon_loss.item(),
                    'kl_loss': kl_loss.item(),
                    'loss': loss.item()
                }
            )
    
    def update(self, step, activations):
        """
        Perform a single training step
        """
        activations = activations.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass and compute loss
        loss = self.loss(activations, step=step)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        # Resample dead neurons if needed
        if self.resample_steps is not None and step > 0 and step % self.resample_steps == 0:
            self.resample_neurons(self.steps_since_active > self.resample_steps / 2, activations)
    
    # Dictionary interface methods - delegate to self.ae
    def encode(self, x, **kwargs):
        """Delegate to ae.encode"""
        return self.ae.encode(x, **kwargs)
    
    def decode(self, f):
        """Delegate to ae.decode"""
        return self.ae.decode(f)
    
    def forward(self, x, output_features=False):
        """Delegate to ae.forward"""
        return self.ae.forward(x, output_features=output_features)
    
    def scale_biases(self, scale: float):
        """Delegate to ae.scale_biases"""
        self.ae.scale_biases(scale)
    
    @property
    def config(self):
        """Return configuration for logging"""
        return {
            'dict_class': self.ae.__class__.__name__,
            'trainer_class': 'VSAEMixTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'kl_coeff': self.kl_coeff,
            'warmup_steps': self.warmup_steps,
            'sparsity_warmup_steps': self.sparsity_warmup_steps,
            'steps': self.steps,
            'decay_start': self.decay_start,
            'resample_steps': self.resample_steps,
            'var_flag': self.var_flag,
            'n_correlated_pairs': self.n_correlated_pairs,
            'n_anticorrelated_pairs': self.n_anticorrelated_pairs,
            'correlation_prior_scale': self.correlation_prior_scale,
            'seed': self.seed,
            'device': self.device,
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
        }