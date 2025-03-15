img_dir = r"datasets/FinalDataset_128"
prompt_path = r"datasets/FinalDatasetPrompts.txt"
vae_path = "stabilityai/sd-vae-ft-ema"
clip_path = "openai/clip-vit-base-patch32"
lpips_path = "vgg_weights/vgg.pth"



unet_params = {
    "in_channels": 4,  # 4 would be input/output channels due to VAE, 3 if operating in pixel space
    "channels": (128, 256, 512, 1024),
    "n_groups": 16,  # Number of normalization groups for groupnorm
    "dropout": 0.1,  # Dropout probability
    "T": 1000,  # Number of timesteps
    "t_embd": 128,  # Embedding dimension to timesteps
    "n_embd": 512,  # Fixed at 512 for CLIP Embeddings
    "n_heads": 16,  # Number of attention heads
    "n_layers": 3,  # Number of layers per Encoder/Mid/Decoder blocks
}


diffusion_params = {
    "beta1": 0.00085,  # Lowest noise level
    "beta2": 0.01200,  # Highest noise level
    "guidance_scale": 4.5,  # CFG Scale
}


training_params = {
    "batch_size": 64,
    "grad_accum_steps": 4,  # Number of batches to accumulate
    "epochs": 500,
    "t_max": 250,  # Dictates over how many epochs to adjust optimization lr. If t_max < epochs, usually the case, then the final (epochs - t_max) epochs would be at minimum lr
    "lr": 1e-4,  # Starting lr
    "min_lr": 1e-6,  # Ending (minimum) lr
    "pl_coeff": 0.25,  # Perceptual loss weight
    "epoch_save_interval": 5,  # Model checkpointing interval
    "model_dir": "saved_models"  # Checkpoint folder name
}
