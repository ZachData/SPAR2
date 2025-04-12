import torch as t
from nnsight import LanguageModel
import dictionary_learning as dl
from dictionary_learning.trainers.vsae_iso import VSAEIsoTrainer
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator

# Set random seed for reproducibility
t.manual_seed(42)
if t.cuda.is_available():
    t.cuda.manual_seed_all(42)

# 1. Load the NeelNanda/GELU_1L model
model = LanguageModel("NeelNanda/GELU_1L512W_C4_Code", device_map="auto")

# 2. Get the MLP submodule (note: model architecture may differ from other models)
submodule = model.gpt_neox.layers[0].mlp

# Print model architecture to verify the correct submodule
print(f"Model loaded: {model._model_key}")
print(f"Submodule selected: {submodule}")
print(f"Activation dimension: {submodule.out_features}")

# 3. Set up data generator
# You can use the C4 dataset for consistency with the model's training data
data_gen = hf_dataset_to_generator("c4", config="en", split="train")

# 4. Create activation buffer
buffer = ActivationBuffer(
    data=data_gen,
    model=model,
    submodule=submodule,
    io="out",  # collect outputs of the MLP
    n_ctxs=1e4,  # number of contexts to store (reduced for faster training)
    ctx_len=128,  # context length
    refresh_batch_size=16,  # batch size for collecting new activations
    out_batch_size=512,  # batch size for training
    device="cuda" if t.cuda.is_available() else "cpu"
)

# 5. Configure VSAE trainer
vsae_config = {
    "trainer": VSAEIsoTrainer,
    "steps": 15000,
    "activation_dim": submodule.out_features,  # automatically use the correct dimension
    "dict_size": submodule.out_features * 4,  # 4x the input dimension
    "layer": 0,
    "lm_name": "NeelNanda/GELU_1L512W_C4_Code",
    "lr": 1e-3,
    "kl_coeff": 3e-4,  # KL divergence coefficient (replaces l1_penalty)
    "warmup_steps": 1000,
    "sparsity_warmup_steps": 2000,
    "resample_steps": 3000,
    "var_flag": 1,  # Use learned variance (0 for fixed variance, more like standard SAE)
    "device": "cuda" if t.cuda.is_available() else "cpu",
    "wandb_name": "VSAE_GELU1L"
}

# 6. Run training
dl.training.trainSAE(
    data=buffer,
    trainer_configs=[vsae_config],
    steps=15000,
    save_dir="./trained_vsae_gelu1l",
    log_steps=100,
    verbose=True
)

# 7. Load the trained VSAE and evaluate
print("\nTraining complete! Evaluating VSAE...")
sae_path = "./trained_vsae_gelu1l/trainer_0/ae.pt"
vsae = dl.dictionary.AutoEncoder.from_pretrained(sae_path, device=vsae_config["device"])

# Perform evaluation
metrics = dl.evaluation.evaluate(
    dictionary=vsae,
    activations=buffer,
    max_len=128,
    batch_size=16,
    io="out",
    device=vsae_config["device"],
    n_batches=5
)

# Print evaluation metrics
print("\nEvaluation Results:")
print(f"L2 Loss: {metrics['l2_loss']:.4f}")
print(f"Variance Explained: {metrics['frac_variance_explained']:.4f}")
print(f"Cosine Similarity: {metrics['cossim']:.4f}")
print(f"Loss Recovery: {metrics['frac_recovered']:.4f}")
print(f"Alive Features: {metrics['frac_alive']:.4f}")

# 8. Optional: Visualize a few random features
try:
    from dictionary_learning.interp import examine_dimension, feature_umap
    import random
    
    print("\nExamining random features...")
    for _ in range(3):
        feature_idx = random.randint(0, vsae_config["dict_size"]-1)
        print(f"\nExamining feature {feature_idx}:")
        
        feature_profile = examine_dimension(
            model=model,
            submodule=submodule,
            buffer=buffer,
            dictionary=vsae,
            dim_idx=feature_idx,
            k=5  # Show top 5 examples
        )
        
        print("Top activating tokens:")
        for token, act in feature_profile.top_tokens[:5]:
            print(f"  {token}: {act:.4f}")
        
        print("Top affected tokens:")
        for token, prob in feature_profile.top_affected[:5]:
            print(f"  {token}: {prob:.4f}")
    
    # Generate UMAP visualization of feature space
    print("\nGenerating UMAP visualization of feature space...")
    feature_plot = feature_umap(vsae, n_components=2)
    feature_plot.write_html("vsae_feature_space.html")
    print("UMAP visualization saved to 'vsae_feature_space.html'")
    
except Exception as e:
    print(f"Visualization failed with error: {e}")

print("\nTraining and evaluation completed!")