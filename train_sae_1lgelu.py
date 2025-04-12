# train_sae_1lgelu.py
import torch as t
from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.training import trainSAE
from gelu_1l_model import load_gelu_1l

# 1. Load model (without weights for now, just the structure)
model, config = load_gelu_1l(device="cuda")

# 2. Get the submodule (MLP layer)
submodule = model.blocks[0].mlp

# 3. Set up data generator - using Neel's pile-10k dataset
data_gen = hf_dataset_to_generator("NeelNanda/pile-10k", split="train")

# 4. Create activation buffer
buffer = ActivationBuffer(
    data=data_gen,
    model=model,
    submodule=submodule,
    d_submodule=2048,  # Explicit MLP output dimension
    io="out",  # Use "out" instead of "output"
    n_ctxs=30000,
    ctx_len=128,
    refresh_batch_size=32,
    out_batch_size=1024,
    device="cuda"
)

# 5. Configure trainer
trainer_config = {
    "trainer": StandardTrainer,
    "steps": 20000,
    "activation_dim": 2048,  # MLP output dimension
    "dict_size": 8192,
    "layer": 0,
    "lm_name": "NeelNanda/GELU_1L512W_C4_Code",
    "lr": 1e-3,
    "l1_penalty": 1e-3,
    "warmup_steps": 1000,
    "sparsity_warmup_steps": 2000,
    "resample_steps": 3000,
    "device": "cuda"
}

# 6. Run training
trainSAE(
    data=buffer,
    trainer_configs=[trainer_config],
    steps=20000,
    save_dir="./trained_sae",
    log_steps=100,
    verbose=True
)