"""
Implements the variational autoencoder (VSAE) training scheme with isotropic Gaussian prior.
"""
import torch as t
from typing import Optional, List, Dict, Any, Tuple
import torch.nn.functional as F
from collections import namedtuple

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn, ConstrainedAdam
from ..config import DEBUG
from ..dictionary import Dictionary


class VSAEIsoGaussian(Dictionary, t.nn.Module):
    """
    Variational Sparse Autoencoder with Isotropic Gaussian Prior
    
    This extends the vanilla SAE by treating the bottleneck as a 
    probabilistic latent variable with an isotropic Gaussian prior.
    
    When var_flag=0, this behaves similarly to a vanilla SAE with a
    different regularization term (KL divergence instead of L1).
    
    When var_flag=1, the model learns both mean and variance of the latent
    distribution, allowing for more complex representations.
    """
    
    def __init__(self, activation_dim, dict_size, device="cuda"):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        
        # Main parameters
        self.W_enc = t.nn.Parameter(t.empty(activation_dim, dict_size, device=device))
        self.W_dec = t.nn.Parameter(t.empty(dict_size, activation_dim, device=device))
        self.b_enc = t.nn.Parameter(t.zeros(dict_size, device=device))
        self.b_dec = t.nn.Parameter(t.zeros(activation_dim, device=device))
        
        # Initialize parameters
        t.nn.init.kaiming_uniform_(self.W_enc)
        t.nn.init.kaiming_uniform_(self.W_dec)
        
        # Normalize decoder weights
        self.normalize_decoder()
    
    def encode(self, x, output_log_var=False):
        """
        Encode a vector x in the activation space.
        
        Args:
            x: Input tensor of shape [batch_size, activation_dim]
            output_log_var: Whether to output the log_var (always False for var_flag=0)
            
        Returns:
            Encoded features of shape [batch_size, dict_size]
            (and log_var if output_log_var=True)
        """
        x_cent = x - self.b_dec
        z = F.relu(x_cent @ self.W_enc + self.b_enc)
        
        if output_log_var:
            log_var = t.zeros_like(z)  # Fixed variance when var_flag=0
            return z, log_var
        return z
    
    def decode(self, f):
        """
        Decode a dictionary vector f
        
        Args:
            f: Dictionary vector of shape [batch_size, dict_size]
            
        Returns:
            Decoded vector of shape [batch_size, activation_dim]
        """
        return f @ self.W_dec + self.b_dec
    
    def forward(self, x, output_features=False):
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape [batch_size, activation_dim]
            output_features: Whether to return features as well
            
        Returns:
            Reconstructed input (and features if output_features=True)
        """
        f = self.encode(x)
        x_hat = self.decode(f)
        
        if output_features:
            return x_hat, f
        else:
            return x_hat
    
    def normalize_decoder(self):
        """Normalize decoder weights to have unit norm"""
        with t.no_grad():
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


class VSAEIsoGaussianLearned(VSAEIsoGaussian):
    """
    VSAE with learned variance
    """
    
    def __init__(self, activation_dim, dict_size, device="cuda"):
        super().__init__(activation_dim, dict_size, device)
        
        # Variance parameters
        self.W_enc_var = t.nn.Parameter(t.empty(activation_dim, dict_size, device=device))
        self.b_enc_var = t.nn.Parameter(t.zeros(dict_size, device=device))
        
        # Initialize variance parameters
        t.nn.init.kaiming_uniform_(self.W_enc_var)
    
    def encode(self, x, output_log_var=False):
        """
        Encode a vector x in the activation space with learned variance
        
        Args:
            x: Input tensor of shape [batch_size, activation_dim]
            output_log_var: Whether to output the log_var
            
        Returns:
            Encoded features of shape [batch_size, dict_size]
            (and log_var if output_log_var=True)
        """
        x_cent = x - self.b_dec
        z = F.relu(x_cent @ self.W_enc + self.b_enc)
        
        if output_log_var:
            log_var = F.relu(x_cent @ self.W_enc_var + self.b_enc_var)
            return z, log_var
        return z
    
    def scale_biases(self, scale: float):
        """Scale biases by a factor"""
        super().scale_biases(scale)
        self.b_enc_var.data *= scale


class VSAEIsoTrainer(SAETrainer):
    """
    Variational Sparse Autoencoder training scheme with isotropic Gaussian prior.
    
    This extends the standard SAE by treating the bottleneck as a probabilistic 
    latent variable with an isotropic Gaussian prior.
    
    When var_flag=0, this behaves similarly to a vanilla SAE with a 
    different regularization term (KL divergence instead of L1).
    
    When var_flag=1, the model learns both mean and variance of the latent
    distribution, allowing for more complex representations.
    """
    
    def __init__(self,
                 steps: int, # total number of steps to train for
                 activation_dim: int,
                 dict_size: int,
                 layer: int,
                 lm_name: str,
                 var_flag: int = 0, # whether to learn variance (0: fixed, 1: learned)
                 kl_coeff: float = 3e-4, # coefficient for KL divergence (like l1_penalty)
                 warmup_steps: int = 1000, # lr warmup period at start of training
                 sparsity_warmup_steps: Optional[int] = 2000, # sparsity warmup period
                 decay_start: Optional[int] = None, # decay learning rate after this many steps
                 resample_steps: Optional[int] = 3000, # how often to resample neurons
                 lr: float = 1e-4, # learning rate
                 dead_neuron_threshold: float = 1e-8, # threshold for identifying dead neurons
                 dead_neuron_window: int = 400, # window for checking dead neurons
                 resample_scale: float = 0.2, # scale for resampled neurons
                 seed: Optional[int] = None,
                 device = None,
                 wandb_name: Optional[str] = 'VSAEIsoTrainer',
                 submodule_name: Optional[str] = None,
    ):
        super().__init__(seed)
        
        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        
        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)
        
        # Initialize dictionary based on var_flag
        if var_flag == 0:
            self.ae = VSAEIsoGaussian(activation_dim, dict_size)
        else:
            self.ae = VSAEIsoGaussianLearned(activation_dim, dict_size)
        
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.var_flag = var_flag
        self.wandb_name = wandb_name
        
        # Setup for dead neuron detection
        self.dead_neuron_threshold = dead_neuron_threshold
        self.dead_neuron_window = dead_neuron_window
        self.resample_steps = resample_steps
        self.resample_scale = resample_scale
        
        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.ae.to(self.device)
        
        # For tracking neuron activations
        if self.resample_steps is not None:
            self.activation_history = []
        
        # Initialize optimizer with constrained decoder
        self.optimizer = ConstrainedAdam(self.ae.parameters(), [self.ae.W_dec], lr=lr)
        
        # Setup learning rate and sparsity schedules
        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps, sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)
        
        # Add logging parameters
        self.logging_parameters = ["kl_coeff", "var_flag"]
    
    def reparameterize(self, mu, log_var):
        """
        Apply the reparameterization trick: z = mu + eps * sigma, where eps ~ N(0, 1)
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            Sampled latent variable z
        """
        std = t.exp(0.5 * log_var)
        eps = t.randn_like(std)
        return mu + eps * std
    
    def compute_kl_divergence(self, mu, log_var):
        """
        Compute KL divergence between N(mu, sigma^2) and N(0, 1)
        
        Formula: KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        
        Args:
            mu: Mean tensor
            log_var: Log variance tensor
            
        Returns:
            KL divergence
        """
        return -0.5 * t.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()
    
    def loss(self, x, step: int, logging=False, **kwargs):
        """
        Compute the VSAE loss with isotropic Gaussian prior
        
        Args:
            x: Input tensor
            step: Current step
            logging: Whether to return extended log info
            
        Returns:
            Loss or LossLog namedtuple with extended information
        """
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        # Get mean and log variance
        if self.var_flag == 0:
            mu = self.ae.encode(x)
            log_var = t.zeros_like(mu)
        else:
            mu, log_var = self.ae.encode(x, output_log_var=True)
        
        # Sample using reparameterization trick
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x_hat = self.ae.decode(z)
        
        # Compute losses
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        kl_loss = self.compute_kl_divergence(mu, log_var)
        
        # Update activation history for dead neuron tracking
        if hasattr(self, 'activation_history'):
            # Track the fraction of batch samples where each neuron is active
            frac_active = (mu.abs() > self.dead_neuron_threshold).float().mean(0)
            self.activation_history.append(frac_active.detach().cpu())
            
            # Keep only the most recent window
            if len(self.activation_history) > self.dead_neuron_window:
                self.activation_history.pop(0)
        
        # Total loss with sparsity scale
        loss = recon_loss + self.kl_coeff * sparsity_scale * kl_loss
        
        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, z,
                {
                    'l2_loss' : l2_loss.item(),
                    'mse_loss' : recon_loss.item(),
                    'kl_loss' : kl_loss.item(),
                    'loss' : loss.item()
                }
            )
    
    def resample_neurons(self, activations):
        """
        Resample dead neurons with high loss activations
        
        Args:
            activations: Batch of activations
            
        Returns:
            Number of resampled neurons
        """
        with t.no_grad():
            if not hasattr(self, 'activation_history') or not self.activation_history:
                return 0
            
            # Stack activation history
            activation_window = t.stack(self.activation_history, dim=0)
            
            # Detect dead neurons (never activated in window)
            dead_mask = (activation_window.sum(0) < self.dead_neuron_threshold)
            dead_indices = t.where(dead_mask)[0]
            n_dead = len(dead_indices)
            
            if n_dead == 0:
                return 0
            
            print(f"Resampling {n_dead} dead neurons")
            
            # Get reconstruction loss for each example
            if self.var_flag == 0:
                z = self.ae.encode(activations)
            else:
                z, _ = self.ae.encode(activations, output_log_var=True)
                
            x_recon = self.ae.decode(z)
            losses = (activations - x_recon).pow(2).sum(dim=-1)
            
            # Sample from highest loss examples
            n_resample = min(n_dead, len(losses))
            _, indices = t.topk(losses, k=n_resample)
            
            # Get norm of alive neurons for scaling
            alive_mask = ~dead_mask
            if alive_mask.any():
                W_enc_norm_alive_mean = self.ae.W_enc[:, alive_mask].norm(dim=0).mean().item()
            else:
                W_enc_norm_alive_mean = 1.0
            
            # Resample each dead neuron
            for i, dead_idx in enumerate(dead_indices[:n_resample]):
                # Get a high-loss activation for replacement
                repl_vec = activations[indices[i % len(indices)]] - self.ae.b_dec
                
                # Normalize and scale
                repl_vec = repl_vec / repl_vec.norm().clamp(min=1e-6)
                repl_vec = repl_vec * W_enc_norm_alive_mean * self.resample_scale
                
                # Update weights and biases
                self.ae.W_enc.data[:, dead_idx] = repl_vec
                self.ae.b_enc.data[dead_idx] = 0.0
                
                # Also reset variance params if needed
                if self.var_flag == 1 and hasattr(self.ae, 'W_enc_var'):
                    self.ae.W_enc_var.data[:, dead_idx] = 0.0
                    self.ae.b_enc_var.data[dead_idx] = 0.0
            
            # Reset Adam parameters for dead neurons
            self._reset_optimizer_stats(dead_indices[:n_resample])
            
            # Clear activation history to avoid immediate resampling
            self.activation_history = []
            
            return n_dead
    
    def _reset_optimizer_stats(self, dead_indices):
        """Reset optimizer state for resampled neurons"""
        if not hasattr(self.optimizer, 'state'):
            return
            
        state_dict = self.optimizer.state_dict()['state']
        
        # For each parameter in the optimizer
        for param_idx, param in enumerate(self.optimizer.param_groups[0]['params']):
            if param_idx not in state_dict:
                continue
                
            # Reset momentum and variance for dead neurons
            if param.shape == self.ae.W_enc.shape:
                for idx in dead_indices:
                    state_dict[param_idx]['exp_avg'][:, idx] = 0.0
                    state_dict[param_idx]['exp_avg_sq'][:, idx] = 0.0
                    
            elif param.shape == self.ae.b_enc.shape:
                for idx in dead_indices:
                    state_dict[param_idx]['exp_avg'][idx] = 0.0
                    state_dict[param_idx]['exp_avg_sq'][idx] = 0.0
                    
            elif self.var_flag == 1 and hasattr(self.ae, 'W_enc_var') and param.shape == self.ae.W_enc_var.shape:
                for idx in dead_indices:
                    state_dict[param_idx]['exp_avg'][:, idx] = 0.0
                    state_dict[param_idx]['exp_avg_sq'][:, idx] = 0.0
                    
            elif self.var_flag == 1 and hasattr(self.ae, 'b_enc_var') and param.shape == self.ae.b_enc_var.shape:
                for idx in dead_indices:
                    state_dict[param_idx]['exp_avg'][idx] = 0.0
                    state_dict[param_idx]['exp_avg_sq'][idx] = 0.0
    
    def update(self, step, activations):
        """
        Update the autoencoder with a batch of activations
        
        Args:
            step: Current training step
            activations: Batch of activations
        """
        activations = activations.to(self.device)
        
        # Forward and backward pass
        self.optimizer.zero_grad()
        loss = self.loss(activations, step=step)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        # Resample dead neurons periodically
        if self.resample_steps is not None and step > 0 and step % self.resample_steps == 0:
            self.resample_neurons(activations)
    
    @property
    def config(self):
        """Return configuration for logging"""
        return {
            'dict_class': 'VSAEIsoGaussian' if self.var_flag == 0 else 'VSAEIsoGaussianLearned',
            'trainer_class': 'VSAEIsoTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr': self.lr,
            'kl_coeff': self.kl_coeff,
            'warmup_steps': self.warmup_steps,
            'sparsity_warmup_steps': self.sparsity_warmup_steps,
            'steps': self.steps,
            'decay_start': self.decay_start,
            'resample_steps': self.resample_steps,
            'dead_neuron_threshold': self.dead_neuron_threshold,
            'dead_neuron_window': self.dead_neuron_window,
            'resample_scale': self.resample_scale,
            'var_flag': self.var_flag,
            'seed': self.seed,
            'device': self.device,
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
        }
