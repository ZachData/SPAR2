# gelu_1l_model.py
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.in_features = d_model
        self.out_features = d_mlp
        self.up_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.act_fn = nn.GELU()
        self.down_proj = nn.Linear(d_mlp, d_model, bias=False)
        
        # Additional attributes needed for ActivationBuffer
        self.inputs = [None]  # Will store input activation
        self.output = None    # Will store output activation
        
    def forward(self, x):
        self.inputs[0] = x    # Store input for "in" access
        x = self.up_proj(x)
        x = self.act_fn(x)
        self.output = x       # Store output for "out" access
        x = self.down_proj(x)
        return x

class GeluBlock(nn.Module):
    def __init__(self, d_model, d_mlp, n_heads, ln_eps=1e-5):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=ln_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model, eps=ln_eps)
        self.mlp = MLP(d_model, d_mlp)
        
    def forward(self, x, attention_mask=None):
        # Self-attention
        residual = x
        x = self.ln1(x)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
            x, _ = self.attn(x, x, x, key_padding_mask=~attention_mask)
        else:
            x, _ = self.attn(x, x, x)
        x = residual + x
        
        # MLP
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        return x

class GeluModel(nn.Module):
    def __init__(self, d_vocab=48262, d_model=512, d_mlp=2048, n_heads=8, n_layers=1, ln_eps=1e-5):
        super().__init__()
        self.d_vocab = d_vocab
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ln_eps = ln_eps
        
        self.embed = nn.Embedding(d_vocab, d_model)
        self.blocks = nn.ModuleList([
            GeluBlock(d_model, d_mlp, n_heads, ln_eps) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model, eps=ln_eps)
        self.unembed = nn.Linear(d_model, d_vocab, bias=False)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.ln_f(x)
        logits = self.unembed(x)
        return logits
    
    def generate(self, input_ids, max_length=20):
        # Simple generation function for convenience
        device = next(self.parameters()).device
        context = input_ids.to(device)
        for _ in range(max_length):
            with t.no_grad():
                logits = self(context)
                next_token = t.argmax(logits[:, -1], dim=-1).unsqueeze(-1)
                context = t.cat([context, next_token], dim=-1)
        return context

def load_gelu_1l(weights_path=None, device='cuda'):
    # Create model with default GELU-1L configuration
    model = GeluModel(
        d_vocab=48262,
        d_model=512,
        d_mlp=2048,
        n_heads=8,
        n_layers=1,
        ln_eps=1e-5
    )
    
    # Create a simple config object for reference
    class Config:
        def __init__(self, model):
            self.d_vocab = model.d_vocab
            self.d_model = model.d_model
            self.d_mlp = model.d_mlp
            self.n_heads = model.n_heads
            self.n_layers = model.n_layers
            self.ln_eps = model.ln_eps
    
    config = Config(model)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("NeelNanda/gpt-neox-tokenizer-digits")
    
    # Load weights if provided
    if weights_path:
        state_dict = t.load(weights_path, map_location=device)
        # You might need to remap keys here if state dict structure doesn't match
        model.load_state_dict(state_dict)
        
    model.to(device)
    model.eval()
    
    # Add tokenizer to model
    model.tokenizer = tokenizer
    
    return model, config