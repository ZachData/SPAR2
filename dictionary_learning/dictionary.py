"""
Defines the dictionary classes
"""

from abc import ABC, abstractmethod
import torch as t
import torch.nn as nn
import torch.nn.init as init
import einops


class Dictionary(ABC, nn.Module):
    """
    A dictionary consists of a collection of vectors, an encoder, and a decoder.
    """

    dict_size: int  # number of features in the dictionary
    activation_dim: int  # dimension of the activation vectors

    @abstractmethod
    def encode(self, x):
        """
        Encode a vector x in the activation space.
        """
        pass

    @abstractmethod
    def decode(self, f):
        """
        Decode a dictionary vector f (i.e. a linear combination of dictionary elements)
        """
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path, device=None, **kwargs) -> "Dictionary":
        """
        Load a pretrained dictionary from a file.
        """
        pass


class AutoEncoder(Dictionary, nn.Module):
    """
    A one-layer autoencoder.
    
    Can be configured in two ways:
    1. Standard mode: Uses bias in the encoder input (old approach from Towards Monosemanticity)
    2. April update mode: Uses bias in both encoder and decoder (newer approach)
    """

    def __init__(self, activation_dim, dict_size, use_april_update_mode=False):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.use_april_update_mode = use_april_update_mode
        
        # Initialize encoder and decoder
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=use_april_update_mode)
        
        # Initialize weights
        w = t.randn(activation_dim, dict_size)
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())
        
        # Initialize biases
        init.zeros_(self.encoder.bias)
        if use_april_update_mode:
            init.zeros_(self.decoder.bias)
        else:
            # In standard mode, we use a separate bias parameter
            self.bias = nn.Parameter(t.zeros(activation_dim))

    def encode(self, x):
        """
        Encode a vector x in the activation space.
        """
        if self.use_april_update_mode:
            return nn.ReLU()(self.encoder(x))
        else:
            return nn.ReLU()(self.encoder(x - self.bias))

    def decode(self, f):
        """
        Decode a dictionary vector f.
        """
        if self.use_april_update_mode:
            return self.decoder(f)
        else:
            return self.decoder(f) + self.bias

    def forward(self, x, output_features=False, ghost_mask=None):
        """
        Forward pass of an autoencoder.
        
        Args:
            x: activations to be autoencoded
            output_features: if True, return the encoded features as well as the decoded x
            ghost_mask: if not None, run this autoencoder in "ghost mode" where features are masked
        """
        if ghost_mask is None:  # normal mode
            f = self.encode(x)
            x_hat = self.decode(f)
            if output_features:
                return x_hat, f
            else:
                return x_hat

        else:  # ghost mode
            if self.use_april_update_mode:
                f_pre = self.encoder(x)
                f_ghost = t.exp(f_pre) * ghost_mask.to(f_pre)
                f = nn.ReLU()(f_pre)
                
                x_ghost = self.decoder(f_ghost)
                x_hat = self.decode(f)
            else:
                f_pre = self.encoder(x - self.bias)
                f_ghost = t.exp(f_pre) * ghost_mask.to(f_pre)
                f = nn.ReLU()(f_pre)
                
                x_ghost = self.decoder(f_ghost)
                x_hat = self.decode(f)
            
            if output_features:
                return x_hat, x_ghost, f
            else:
                return x_hat, x_ghost

    def scale_biases(self, scale: float):
        """
        Scale all bias parameters by a given factor.
        """
        self.encoder.bias.data *= scale
        if self.use_april_update_mode:
            self.decoder.bias.data *= scale
        else:
            self.bias.data *= scale

    def normalize_decoder(self):
        """
        Normalize decoder weights to have unit norm.
        """
        norms = t.norm(self.decoder.weight, dim=0).to(dtype=self.decoder.weight.dtype, device=self.decoder.weight.device)

        if t.allclose(norms, t.ones_like(norms)):
            return
        print("Normalizing decoder weights")

        test_input = t.randn(10, self.activation_dim)
        initial_output = self(test_input)

        self.decoder.weight.data /= norms

        new_norms = t.norm(self.decoder.weight, dim=0)
        assert t.allclose(new_norms, t.ones_like(new_norms))

        self.encoder.weight.data *= norms[:, None]
        self.encoder.bias.data *= norms

        new_output = self(test_input)

        # Errors can be relatively large in larger SAEs due to floating point precision
        assert t.allclose(initial_output, new_output, atol=1e-4)

    @classmethod
    def from_pretrained(cls, path, dtype=t.float, device=None, normalize_decoder=True):
        """
        Load a pretrained autoencoder from a file.
        
        Args:
            path: Path to the saved model
            dtype: Data type to convert model to
            device: Device to load model to
            normalize_decoder: Whether to normalize decoder weights
            
        Returns:
            Loaded autoencoder
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        
        # Determine mode based on state dict
        use_april_update_mode = "decoder.bias" in state_dict
        
        autoencoder = cls(activation_dim, dict_size, use_april_update_mode=use_april_update_mode)
        autoencoder.load_state_dict(state_dict)

        # This is useful for doing analysis where e.g. feature activation magnitudes are important
        # If training the SAE using the April update, the decoder weights are not normalized
        if normalize_decoder:
            autoencoder.normalize_decoder()

        if device is not None:
            autoencoder.to(dtype=dtype, device=device)

        return autoencoder


class IdentityDict(Dictionary, nn.Module):
    """
    An identity dictionary, i.e. the identity function.
    """

    def __init__(self, activation_dim=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = activation_dim

    def encode(self, x):
        return x

    def decode(self, f):
        return f

    def forward(self, x, output_features=False, ghost_mask=None):
        if output_features:
            return x, x
        else:
            return x

    @classmethod
    def from_pretrained(cls, path, dtype=t.float, device=None):
        """
        Load a pretrained dictionary from a file.
        """
        return cls(None)


class GatedAutoEncoder(Dictionary, nn.Module):
    """
    An autoencoder with separate gating and magnitude networks.
    """

    def __init__(self, activation_dim, dict_size, initialization="default", device=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.decoder_bias = nn.Parameter(t.empty(activation_dim, device=device))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=False, device=device)
        self.r_mag = nn.Parameter(t.empty(dict_size, device=device))
        self.gate_bias = nn.Parameter(t.empty(dict_size, device=device))
        self.mag_bias = nn.Parameter(t.empty(dict_size, device=device))
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False, device=device)
        if initialization == "default":
            self._reset_parameters()
        else:
            initialization(self)

    def _reset_parameters(self):
        """
        Default method for initializing GatedSAE weights.
        """
        # biases are initialized to zero
        init.zeros_(self.decoder_bias)
        init.zeros_(self.r_mag)
        init.zeros_(self.gate_bias)
        init.zeros_(self.mag_bias)

        # decoder weights are initialized to random unit vectors
        dec_weight = t.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)
        self.encoder.weight = nn.Parameter(dec_weight.clone().T)

    def encode(self, x: t.Tensor, return_gate:bool=False, normalize_decoder:bool=False):
        """
        Returns features, gate value (pre-Heavyside)
        """
        x_enc = self.encoder(x - self.decoder_bias)

        # gating network
        pi_gate = x_enc + self.gate_bias
        f_gate = (pi_gate > 0).to(self.encoder.weight.dtype)

        # magnitude network
        pi_mag = self.r_mag.exp() * x_enc + self.mag_bias
        f_mag = nn.ReLU()(pi_mag)

        f = f_gate * f_mag

        if normalize_decoder:
            # If the SAE is trained without ConstrainedAdam, the decoder vectors are not normalized
            # Normalizing after encode, and renormalizing before decode to enable comparability
            f = f * self.decoder.weight.norm(dim=0, keepdim=True)

        if return_gate:
            return f, nn.ReLU()(pi_gate)

        return f

    def decode(self, f: t.Tensor, normalize_decoder:bool=False):
        if normalize_decoder:
            # If the SAE is trained without ConstrainedAdam, the decoder vectors are not normalized
            # Normalizing after encode, and renormalizing before decode to enable comparability
            f = f / self.decoder.weight.norm(dim=0, keepdim=True)
        return self.decoder(f) + self.decoder_bias

    def forward(self, x:t.Tensor, output_features:bool=False, normalize_decoder:bool=False):
        f = self.encode(x)
        x_hat = self.decode(f)

        if normalize_decoder:
            f = f * self.decoder.weight.norm(dim=0, keepdim=True)

        if output_features:
            return x_hat, f
        else:
            return x_hat

    def scale_biases(self, scale: float):
        self.decoder_bias.data *= scale
        self.mag_bias.data *= scale
        self.gate_bias.data *= scale

    @classmethod
    def from_pretrained(cls, path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = GatedAutoEncoder(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class JumpReluAutoEncoder(Dictionary, nn.Module):
    """
    An autoencoder with jump ReLUs.
    """

    def __init__(self, activation_dim, dict_size, device="cpu"):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.W_enc = nn.Parameter(t.empty(activation_dim, dict_size, device=device))
        self.b_enc = nn.Parameter(t.zeros(dict_size, device=device))
        self.W_dec = nn.Parameter(
            t.nn.init.kaiming_uniform_(t.empty(dict_size, activation_dim, device=device))
        )
        self.b_dec = nn.Parameter(t.zeros(activation_dim, device=device))
        self.threshold = nn.Parameter(t.ones(dict_size, device=device) * 0.001)  # Appendix I

        self.apply_b_dec_to_input = False

        self.W_dec.data = self.W_dec / self.W_dec.norm(dim=1, keepdim=True)
        self.W_enc.data = self.W_dec.data.clone().T

    def encode(self, x, output_pre_jump=False):
        if self.apply_b_dec_to_input:
            x = x - self.b_dec
        pre_jump = x @ self.W_enc + self.b_enc

        f = nn.ReLU()(pre_jump * (pre_jump > self.threshold))

        if output_pre_jump:
            return f, pre_jump
        else:
            return f

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x, output_features=False):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features (and their pre-jump version) as well as the decoded x
        """
        f = self.encode(x)
        x_hat = self.decode(f)
        if output_features:
            return x_hat, f
        else:
            return x_hat

    def scale_biases(self, scale: float):
        self.b_dec.data *= scale
        self.b_enc.data *= scale
        self.threshold.data *= scale

    @classmethod
    def from_pretrained(
        cls,
        path: str | None = None,
        load_from_sae_lens: bool = False,
        dtype: t.dtype = t.float32,
        device: t.device | None = None,
        **kwargs,
    ):
        """
        Load a pretrained autoencoder from a file.
        If sae_lens=True, then pass **kwargs to sae_lens's
        loading function.
        """
        if not load_from_sae_lens:
            state_dict = t.load(path)
            activation_dim, dict_size = state_dict["W_enc"].shape
            autoencoder = JumpReluAutoEncoder(activation_dim, dict_size)
            autoencoder.load_state_dict(state_dict)
            autoencoder = autoencoder.to(dtype=dtype, device=device)
        else:
            from sae_lens import SAE

            sae, cfg_dict, _ = SAE.from_pretrained(**kwargs)
            assert (
                cfg_dict["finetuning_scaling_factor"] == False
            ), "Finetuning scaling factor not supported"
            dict_size, activation_dim = cfg_dict["d_sae"], cfg_dict["d_in"]
            autoencoder = JumpReluAutoEncoder(activation_dim, dict_size, device=device)
            autoencoder.load_state_dict(sae.state_dict())
            autoencoder.apply_b_dec_to_input = cfg_dict["apply_b_dec_to_input"]

        if device is not None:
            device = autoencoder.W_enc.device
        return autoencoder.to(dtype=dtype, device=device)