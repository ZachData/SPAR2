"""
Implements the variational autoencoder (VSAE) training scheme with isotropic Gaussian prior.
"""
import torch as t
from typing import Optional, List
from collections import namedtuple

from ..trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn, ConstrainedAdam
from ..config import DEBUG
from ..dictionary import AutoEncoder

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
                 dict_class=AutoEncoder,
                 lr:float=1e-3,
                 kl_coeff:float=1e-1, # equivalent to l1_penalty in standard trainer
                 warmup_steps:int=1000, # lr warmup period at start of training and after each resample
                 sparsity_warmup_steps:Optional[int]=2000, # sparsity warmup period at start of training
                 decay_start:Optional[int]=None, # decay learning rate after this many steps
                 resample_steps:Optional[int]=None, # how often to resample neurons
                 var_flag:int=0, # whether to learn variance (0: fixed, 1: learned)
                 seed:Optional[int]=None,
                 device=None,
                 wandb_name:Optional[str]='VSAEIsoTrainer',
                 submodule_name:Optional[str]=None,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # initialize dictionary
        self.ae = dict_class(activation_dim, dict_size)

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

        # Additional parameters for VAE variance
        if self.var_flag == 1:
            # Initialize parameters for variance when var_flag is 1
            self.ae.W_enc_var = t.nn.Parameter(t.nn.init.xavier_normal_(t.empty((activation_dim, dict_size), device=self.device)))
            self.ae.b_enc_var = t.nn.Parameter(t.zeros(dict_size, device=self.device))

        self.resample_steps = resample_steps
        if self.resample_steps is not None:
            # how many steps since each neuron was last activated?
            self.steps_since_active = t.zeros(self.ae.dict_size, dtype=int).to(self.device)
        else:
            self.steps_since_active = None 

        self.optimizer = ConstrainedAdam(self.ae.parameters(), [self.ae.decoder.weight], lr=lr)

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps, sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)

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
        
    def resample_neurons(self, deads, activations):
        with t.no_grad():
            if deads.sum() == 0: return
            print(f"resampling {deads.sum().item()} neurons")

            # compute loss for each activation
            losses = (activations - self.ae(activations)).norm(dim=-1)

            # sample input to create encoder/decoder weights from
            n_resample = min([deads.sum(), losses.shape[0]])
            indices = t.multinomial(losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]

            # get norm of the living neurons
            alive_norm = self.ae.encoder.weight[~deads].norm(dim=-1).mean()

            # resample first n_resample dead neurons
            deads[deads.nonzero()[n_resample:]] = False
            self.ae.encoder.weight[deads] = sampled_vecs * alive_norm * 0.2
            self.ae.decoder.weight[:,deads] = (sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)).T
            self.ae.encoder.bias[deads] = 0.

            # If variance is learned, also reset those parameters
            if self.var_flag == 1:
                self.ae.W_enc_var[deads] = t.zeros_like(self.ae.W_enc_var[deads])
                self.ae.b_enc_var[deads] = 0.

            # reset Adam parameters for dead neurons
            state_dict = self.optimizer.state_dict()['state']
            ## encoder weight
            state_dict[1]['exp_avg'][deads] = 0.
            state_dict[1]['exp_avg_sq'][deads] = 0.
            ## encoder bias
            state_dict[2]['exp_avg'][deads] = 0.
            state_dict[2]['exp_avg_sq'][deads] = 0.
            ## decoder weight
            state_dict[3]['exp_avg'][:,deads] = 0.
            state_dict[3]['exp_avg_sq'][:,deads] = 0.
    
    def loss(self, x, step: int, logging=False, **kwargs):
        sparsity_scale = self.sparsity_warmup_fn(step)

        # Get activations - added code for VAE
        x_cent = x - self.ae.bias
        mu = t.nn.functional.relu(x_cent @ self.ae.encoder.weight.T + self.ae.encoder.bias)
        
        # Compute log variance if var_flag=1, otherwise fixed variance
        if self.var_flag == 1:
            log_var = t.nn.functional.relu(x_cent @ self.ae.W_enc_var + self.ae.b_enc_var)
        else:
            log_var = t.zeros_like(mu)
        
        # Sample from the latent distribution using reparameterization trick
        f = self.reparameterize(mu, log_var)
        
        # Decode the sampled features
        x_hat = f @ self.ae.decoder.weight.T + self.ae.bias
        
        # Compute losses
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        
        # KL divergence between N(mu, sigma^2) and N(0, 1)
        # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * t.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()
        
        loss = recon_loss + self.kl_coeff * sparsity_scale * kl_loss

        if self.steps_since_active is not None:
            # update steps_since_active
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
        
        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'l2_loss' : t.linalg.norm(x - x_hat, dim=-1).mean().item(),
                    'mse_loss' : recon_loss.item(),
                    'kl_loss' : kl_loss.item(),
                    'loss' : loss.item()
                }
            )

    def update(self, step, activations):
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations, step=step)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if self.resample_steps is not None and step % self.resample_steps == 0:
            self.resample_neurons(self.steps_since_active > self.resample_steps / 2, activations)

    @property
    def config(self):
        return {
            'dict_class': 'AutoEncoder',
            'trainer_class' : 'VSAEIsoTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr' : self.lr,
            'kl_coeff' : self.kl_coeff,
            'warmup_steps' : self.warmup_steps,
            'resample_steps' : self.resample_steps,
            'sparsity_warmup_steps' : self.sparsity_warmup_steps,
            'steps' : self.steps,
            'decay_start' : self.decay_start,
            'seed' : self.seed,
            'device' : self.device,
            'layer' : self.layer,
            'lm_name' : self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'var_flag': self.var_flag,
        }
