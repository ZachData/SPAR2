"""
Implements a Variational Sparse Autoencoder with multivariate Gaussian prior.
This implementation is designed to handle general correlation structures in the latent space.
"""
import torch as t
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from collections import namedtuple

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn, ConstrainedAdam
from ..config import DEBUG
from ..dictionary import Dictionary


class VSAEMultiGaussian(Dictionary, t.nn.Module):
    """
    Variational Sparse Autoencoder with multivariate Gaussian prior
    Designed to handle general correlation structures in the latent space
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
        
        # Initialize parameters with Kaiming uniform
        t.nn.init.kaiming_uniform_(self.W_enc)
        t.nn.init.kaiming_uniform_(self.W_dec)
        
        # Normalize decoder weights
        self.normalize_decoder()
    
    def encode(self, x, output_log_var=False):
        """
        Encode a vector x in the activation space.
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


class VSAEMultiGaussianLearned(VSAEMultiGaussian):
    """
    VSAE with multivariate Gaussian prior and learned variance
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


class VSAEMultiGaussianTrainer(SAETrainer):
    """
    Trainer for VSAE with multivariate Gaussian prior
    """
    
    def __init__(
        self,
        steps: int,                           # total number of steps to train for
        activation_dim: int,                  # dimension of input activations
        dict_size: int,                       # dictionary size
        layer: int,                           # layer to train on
        lm_name: str,                         # language model name
        corr_rate: float = 0.5,               # correlation rate for prior
        corr_matrix: Optional[t.Tensor] = None, # custom correlation matrix
        var_flag: int = 0,                    # whether to use fixed (0) or learned (1) variance
        lr: float = 1e-3,                     # learning rate
        kl_coeff: float = 3e-4,               # KL coefficient for variational loss
        warmup_steps: int = 1000,             # LR warmup steps
        sparsity_warmup_steps: Optional[int] = 2000, # sparsity warmup steps
        decay_start: Optional[int] = None,    # when to start LR decay
        resample_steps: Optional[int] = 3000, # how often to resample neurons
        dead_neuron_threshold: float = 1e-8,  # threshold for identifying dead neurons
        dead_neuron_window: int = 400,        # window for dead neuron detection
        resample_scale: float = 0.2,          # scale for resampled neurons
        seed: Optional[int] = None,
        device = None,
        wandb_name: Optional[str] = 'VSAEMultiGaussianTrainer',
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
        
        # Initialize dictionary
        if var_flag == 0:
            self.ae = VSAEMultiGaussian(activation_dim, dict_size)
        else:
            self.ae = VSAEMultiGaussianLearned(activation_dim, dict_size)
        
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name
        self.var_flag = var_flag
        
        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.ae.to(self.device)
        
        # Setup correlation matrix for prior
        self.corr_rate = corr_rate
        self.corr_matrix = corr_matrix
        self.prior_covariance = self._build_prior_covariance()
        self.prior_precision = self._compute_prior_precision()
        self.prior_cov_logdet = self._compute_prior_logdet()
        
        # Setup for dead neuron tracking
        self.resample_steps = resample_steps
        self.dead_neuron_threshold = dead_neuron_threshold
        self.dead_neuron_window = dead_neuron_window
        self.resample_scale = resample_scale
        
        if self.resample_steps is not None:
            # Track neuron activations over time
            self.activation_history = []
        
        # Initialize optimizer
        self.optimizer = ConstrainedAdam(self.ae.parameters(), [self.ae.W_dec], lr=lr)
        
        # Setup learning rate and sparsity schedules
        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps, sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        
        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)
        
        # Add logging parameters
        self.logging_parameters = ["kl_coeff", "var_flag", "corr_rate"]
    
    def _build_prior_covariance(self):
        """Build the prior covariance matrix"""
        d_hidden = self.ae.dict_size
        
        if self.corr_matrix is not None and self.corr_matrix.shape[0] == d_hidden:
            corr_matrix = self.corr_matrix
        elif self.corr_rate is not None:
            # Create a matrix with uniform correlation
            corr_matrix = t.full((d_hidden, d_hidden), self.corr_rate, device=self.device)
            t.diagonal(corr_matrix)[:] = 1.0
        else:
            # Default to identity (no correlation)
            return t.eye(d_hidden, device=self.device)
        
        # Ensure the correlation matrix is valid
        if not t.allclose(corr_matrix, corr_matrix.t()):
            print("Warning: Correlation matrix not symmetric, symmetrizing it")
            corr_matrix = 0.5 * (corr_matrix + corr_matrix.t())
        
        # Add small jitter to ensure positive definiteness
        corr_matrix = corr_matrix + t.eye(d_hidden, device=self.device) * 1e-4
        
        return corr_matrix.to(self.device)
    
    def _compute_prior_precision(self):
        """Compute the precision matrix (inverse of covariance)"""
        try:
            return t.linalg.inv(self.prior_covariance)
        except:
            # Add jitter for numerical stability
            jitter = t.eye(self.ae.dict_size, device=self.device) * 1e-4
            return t.linalg.inv(self.prior_covariance + jitter)
    
    def _compute_prior_logdet(self):
        """Compute log determinant of prior covariance"""
        return t.logdet(self.prior_covariance)
    
    def loss(self, x, step: int, logging=False, **kwargs):
        """
        Compute the VSAE loss with multivariate Gaussian prior
        """
        sparsity_scale = self.sparsity_warmup_fn(step)
        
        if self.var_flag == 0:
            # Fixed variance version
            z = self.ae.encode(x)
            log_var = t.zeros_like(z)
        else:
            # Learned variance version
            z, log_var = self.ae.encode(x, output_log_var=True)
        
        x_hat = self.ae.decode(z)
        
        # Reconstruction loss
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        
        # KL divergence with multivariate Gaussian prior
        kl_loss = self._compute_kl_divergence(z, log_var)
        
        # Total loss
        loss = recon_loss + self.kl_coeff * sparsity_scale * kl_loss
        
        # Update activation history for dead neuron detection
        if self.resample_steps is not None:
            frac_active = (z.abs() > self.dead_neuron_threshold).float().mean(0)
            self.activation_history.append(frac_active.detach().cpu())
            
            # Keep only the most recent window
            if len(self.activation_history) > self.dead_neuron_window:
                self.activation_history.pop(0)
        
        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, z,
                {
                    'mse_loss': recon_loss.item(),
                    'kl_loss': kl_loss.item(), 
                    'loss': loss.item()
                }
            )
    
    def _compute_kl_divergence(self, mu, log_var):
        """
        Compute KL divergence between approximate posterior and prior
        """
        # For efficiency, compute on batch mean
        mu_avg = mu.mean(0)  # [dict_size]
        var_avg = log_var.exp().mean(0)  # [dict_size]
        
        # Trace term: tr(Σp^-1 * Σq)
        trace_term = (self.prior_precision.diagonal() * var_avg).sum()
        
        # Quadratic term: μ^T * Σp^-1 * μ
        quad_term = mu_avg @ self.prior_precision @ mu_avg
        
        # Log determinant term: ln(|Σp|/|Σq|)
        log_det_q = log_var.sum(1).mean()
        log_det_term = self.prior_cov_logdet - log_det_q
        
        # Combine terms
        kl = 0.5 * (trace_term + quad_term - self.ae.dict_size + log_det_term)
        
        # Ensure non-negative
        kl = t.clamp(kl, min=0.0)
        
        return kl
    
    def resample_neurons(self, activations):
        """
        Resample dead neurons with high loss activations
        """
        with t.no_grad():
            if not self.activation_history:
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
                if self.var_flag == 1:
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
        
        # Reset encoder weight stats
        if 1 in state_dict:  # W_enc is usually param 1
            if 'exp_avg' in state_dict[1]:
                for idx in dead_indices:
                    state_dict[1]['exp_avg'][:, idx] = 0.0
                    state_dict[1]['exp_avg_sq'][:, idx] = 0.0
                    
        # Reset encoder bias stats
        if 2 in state_dict:  # b_enc is usually param 2
            if 'exp_avg' in state_dict[2]:
                for idx in dead_indices:
                    state_dict[2]['exp_avg'][idx] = 0.0
                    state_dict[2]['exp_avg_sq'][idx] = 0.0
        
        # Reset variance params if needed
        if self.var_flag == 1:
            if 3 in state_dict:  # W_enc_var
                if 'exp_avg' in state_dict[3]:
                    for idx in dead_indices:
                        state_dict[3]['exp_avg'][:, idx] = 0.0
                        state_dict[3]['exp_avg_sq'][:, idx] = 0.0
            
            if 4 in state_dict:  # b_enc_var
                if 'exp_avg' in state_dict[4]:
                    for idx in dead_indices:
                        state_dict[4]['exp_avg'][idx] = 0.0
                        state_dict[4]['exp_avg_sq'][idx] = 0.0
    
    def update(self, step, activations):
        """
        Update the autoencoder with a batch of activations
        """
        activations = activations.to(self.device)
        
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
            'dict_class': 'VSAEMultiGaussian' if self.var_flag == 0 else 'VSAEMultiGaussianLearned',
            'trainer_class': 'VSAEMultiGaussianTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr': self.lr,
            'kl_coeff': self.kl_coeff,
            'warmup_steps': self.warmup_steps,
            'sparsity_warmup_steps': self.sparsity_warmup_steps,
            'resample_steps': self.resample_steps,
            'dead_neuron_threshold': self.dead_neuron_threshold,
            'dead_neuron_window': self.dead_neuron_window,
            'resample_scale': self.resample_scale,
            'corr_rate': self.corr_rate,
            'var_flag': self.var_flag,
            'steps': self.steps,
            'decay_start': self.decay_start,
            'seed': self.seed,
            'device': self.device,
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
        }
