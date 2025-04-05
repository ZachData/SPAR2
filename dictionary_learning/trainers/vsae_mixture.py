"""
Implements a Variational Sparse Autoencoder with Gaussian Mixture Prior.
Designed to better model correlated and anti-correlated feature pairs.
"""
import torch as t
from typing import Optional, List, Dict, Any, Tuple
import torch.nn.functional as F
from collections import namedtuple

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn
from ..config import DEBUG

class VSAEMixtureTrainer(SAETrainer):
    """
    Variational Sparse Autoencoder with Gaussian Mixture Prior.
    
    This extends the isotropic Gaussian VSAE by using a mixture of Gaussians
    as the prior distribution to better model correlated and anti-correlated
    feature pairs.
    
    The prior means are structured as follows:
    - Correlated pairs: Both features have positive means
    - Anti-correlated pairs: One feature has positive mean, the other negative
    - Uncorrelated features: Zero mean (standard prior)
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
                 wandb_name: Optional[str] = 'VSAEMixtureTrainer',
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
            
        self.activation_dim = activation_dim
        self.dict_size = dict_size
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
            
        # Initialize encoder and decoder parameters
        self.W_enc = t.nn.Parameter(t.empty(activation_dim, dict_size, device=self.device))
        self.b_enc = t.nn.Parameter(t.zeros(dict_size, device=self.device))
        
        if self.var_flag == 1:
            self.W_enc_var = t.nn.Parameter(t.empty(activation_dim, dict_size, device=self.device))
            self.b_enc_var = t.nn.Parameter(t.zeros(dict_size, device=self.device))
            
        self.W_dec = t.nn.Parameter(t.empty(dict_size, activation_dim, device=self.device))
        self.b_dec = t.nn.Parameter(t.zeros(activation_dim, device=self.device))
        
        # Initialize parameters with Kaiming uniform
        t.nn.init.kaiming_uniform_(self.W_enc)
        t.nn.init.kaiming_uniform_(self.W_dec)
        
        if self.var_flag == 1:
            t.nn.init.kaiming_uniform_(self.W_enc_var)
            
        # Normalize decoder weights
        self.normalize_decoder()
        
        # Initialize the prior means based on correlation structure
        self.prior_means = self._initialize_prior_means()
        
        # For dead neuron detection and resampling
        self.resample_steps = resample_steps
        if self.resample_steps is not None:
            # How many steps since each neuron was last activated?
            self.steps_since_active = t.zeros(dict_size, dtype=int).to(self.device)
        else:
            self.steps_since_active = None
        
        # Create optimizer
        self.optimizer = t.optim.Adam(self.parameters(), lr=1e-4)
        
        # Create learning rate scheduler
        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps, sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        
        # Create sparsity warmup function
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)
        
        # Add tracking metrics for logging
        self.logging_parameters = ["kl_coeff", "var_flag", "n_correlated_pairs", "n_anticorrelated_pairs"]
    
    def _initialize_prior_means(self) -> t.Tensor:
        """
        Initialize the prior means for the latent variables based on 
        the specified correlation structure.
        
        Returns:
            prior_means: Tensor of shape [dict_size] with means for the prior distribution
        """
        means = t.zeros(self.dict_size, device=self.device)
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
        
        # The remaining features have zero mean (standard prior)
        return means
    
    @t.no_grad()
    def normalize_decoder(self):
        """Normalize decoder weights to have unit norm"""
        norm = self.W_dec.norm(dim=-1, keepdim=True)
        self.W_dec.data = self.W_dec.data / norm.clamp(min=1e-6)
    
    def reparameterize(self, mu: t.Tensor, log_var: t.Tensor) -> t.Tensor:
        """
        Apply the reparameterization trick:
        z = mu + eps * sigma, where eps ~ N(0, 1)
        
        Args:
            mu: Mean of latent distribution, shape [batch_size, dict_size]
            log_var: Log variance of latent distribution, shape [batch_size, dict_size]
            
        Returns:
            Sampled latent variable z, shape [batch_size, dict_size]
        """
        std = t.exp(0.5 * log_var)
        eps = t.randn_like(std)
        return mu + eps * std
    
    def compute_kl_divergence(self, mu: t.Tensor, log_var: t.Tensor) -> t.Tensor:
        """
        Compute KL divergence between q(z|x) = N(mu, sigma^2) and 
        the mixture prior distribution p(z) with structured means.
        
        For a Gaussian with non-zero mean prior:
        KL(N(mu, sigma^2) || N(prior_mu, 1)) = 
            0.5 * [log(1/sigma^2) + sigma^2 + (mu-prior_mu)^2 - 1]
        
        Args:
            mu: Mean of latent distribution, shape [batch_size, dict_size]
            log_var: Log variance of latent distribution, shape [batch_size, dict_size]
            
        Returns:
            KL divergence (scalar)
        """
        # Expand prior_means to match batch dimension [1, dict_size] -> [batch_size, dict_size]
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
    
    def loss(self, x, step: int, logging=False, **kwargs):
        """
        Compute the VSAE loss with mixture prior
        
        Args:
            x: Input activations [batch_size, activation_dim]
            step: Current training step
            logging: Whether to return extended information for logging
            
        Returns:
            Loss value or namedtuple with extended information
        """
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Center the input using the decoder bias
        x_cent = x - self.b_dec
        
        # Encode to get mean of latent distribution
        mu = F.relu(x_cent @ self.W_enc + self.b_enc)
        
        # Get log variance of latent distribution
        if self.var_flag == 1:
            log_var = F.relu(x_cent @ self.W_enc_var + self.b_enc_var)
        else:
            # Fixed variance when var_flag=0
            log_var = t.zeros_like(mu)
        
        # Sample from the latent distribution using reparameterization trick
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x_hat = z @ self.W_dec + self.b_dec
        
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
    
    def resample_neurons(self, deads, activations):
        """
        Resample dead neurons using activations from the batch.
        
        Args:
            deads: Boolean tensor indicating dead neurons
            activations: Tensor of activations to sample from
            
        Returns:
            None
        """
        with t.no_grad():
            if deads.sum() == 0: return
            print(f"resampling {deads.sum().item()} neurons")

            # compute loss for each activation
            losses = (activations - self(activations)).norm(dim=-1)

            # sample input to create encoder/decoder weights from
            n_resample = min([deads.sum(), losses.shape[0]])
            indices = t.multinomial(losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]

            # get norm of the living neurons
            alive_norm = self.W_enc[:, ~deads].norm(dim=0).mean()

            # resample first n_resample dead neurons
            deads[deads.nonzero()[n_resample:]] = False
            self.W_enc.data[:, deads] = (sampled_vecs - self.b_dec).T * alive_norm * 0.2
            self.W_dec.data[deads, :] = ((sampled_vecs - self.b_dec) / (sampled_vecs - self.b_dec).norm(dim=-1, keepdim=True)).T
            self.b_enc.data[deads] = 0.
            
            if self.var_flag == 1:
                self.W_enc_var.data[:, deads] = 0.
                self.b_enc_var.data[deads] = 0.

            # reset Adam parameters for dead neurons
            state_dict = self.optimizer.state_dict()['state']
            ## encoder weight
            for param_idx, param in enumerate(self.optimizer.param_groups[0]['params']):
                if param.shape == self.W_enc.shape:
                    state_dict[param_idx]['exp_avg'][:, deads] = 0.
                    state_dict[param_idx]['exp_avg_sq'][:, deads] = 0.
                elif param.shape == self.b_enc.shape:
                    state_dict[param_idx]['exp_avg'][deads] = 0.
                    state_dict[param_idx]['exp_avg_sq'][deads] = 0.
                elif self.var_flag == 1 and param.shape == self.W_enc_var.shape:
                    state_dict[param_idx]['exp_avg'][:, deads] = 0.
                    state_dict[param_idx]['exp_avg_sq'][:, deads] = 0.
                elif self.var_flag == 1 and param.shape == self.b_enc_var.shape:
                    state_dict[param_idx]['exp_avg'][deads] = 0.
                    state_dict[param_idx]['exp_avg_sq'][deads] = 0.
                elif param.shape == self.W_dec.shape:
                    state_dict[param_idx]['exp_avg'][deads, :] = 0.
                    state_dict[param_idx]['exp_avg_sq'][deads, :] = 0.
    
    def update(self, step, activations):
        """
        Perform a single training step
        
        Args:
            step: Current training step
            activations: Input activations
            
        Returns:
            None
        """
        activations = activations.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass and compute loss
        loss = self.loss(activations, step=step)
        
        # Backward pass
        loss.backward()
        
        # Remove gradients parallel to the decoder directions
        with t.no_grad():
            self._remove_parallel_component_of_grads()
        
        # Update parameters
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        # Normalize decoder
        self.normalize_decoder()
        
        # Resample dead neurons if needed
        if self.resample_steps is not None and step % self.resample_steps == 0:
            self.resample_neurons(self.steps_since_active > self.resample_steps / 2, activations)
    
    @t.no_grad()
    def _remove_parallel_component_of_grads(self):
        """
        Remove the parallel component of gradients for decoder weights
        This maintains the unit norm constraint during gradient descent
        """
        if self.W_dec.grad is None:
            return
            
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
    
    def forward(self, x):
        """
        Forward pass through the VAE
        
        Args:
            x: Input activations
            
        Returns:
            Reconstructed activations
        """
        x_cent = x - self.b_dec
        
        # Encode 
        mu = F.relu(x_cent @ self.W_enc + self.b_enc)
        
        # For inference we use the mean without sampling
        x_hat = mu @ self.W_dec + self.b_dec
        
        return x_hat
    
    def encode(self, x, deterministic=True):
        """
        Encode inputs to sparse features
        
        Args:
            x: Input tensor
            deterministic: If True, return mean without sampling
            
        Returns:
            Encoded features
        """
        x_cent = x - self.b_dec
        mu = F.relu(x_cent @ self.W_enc + self.b_enc)
        
        if deterministic:
            return mu
        
        # Get log variance 
        if self.var_flag == 1:
            log_var = F.relu(x_cent @ self.W_enc_var + self.b_enc_var)
        else:
            log_var = t.zeros_like(mu)
        
        # Sample from the latent distribution
        return self.reparameterize(mu, log_var)
    
    def decode(self, z):
        """
        Decode sparse features back to inputs
        
        Args:
            z: Sparse features
            
        Returns:
            Reconstructed inputs
        """
        return z @ self.W_dec + self.b_dec
    
    @property
    def config(self):
        """Return configuration for logging and saving"""
        return {
            'dict_class': 'VSAEMixtureTrainer',
            'trainer_class': 'VSAEMixtureTrainer',
            'activation_dim': self.activation_dim,
            'dict_size': self.dict_size,
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
