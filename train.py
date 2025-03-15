import os
import time
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from config import unet_params, diffusion_params, training_params, img_dir, prompt_path, vae_path, clip_path, lpips_path
from dataset_loader import DatasetLoader
from unet import DiffusionModel
from vgg import LPIPS
from helper_functions import eval_model, get_prompt_tensor



# Setting to 'high' uses TF32 rather than FP32, which makes the training process faster (varies on machines)
# Can set to 'medium' for even faster training, though will be loss in performance
# Check out the documentations https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("high")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Currently using {device=}\n")


batch_size = training_params["batch_size"]
grad_accum_steps = training_params["grad_accum_steps"]
epochs = training_params["epochs"]
t_max = training_params["t_max"]
lr = training_params["lr"]
min_lr = training_params["min_lr"]
epoch_save_interval = training_params["epoch_save_interval"]


cfg = diffusion_params["guidance_scale"]


model_dir = training_params["model_dir"]
os.makedirs(model_dir, exist_ok=True)
if len(os.listdir(model_dir)) > 0:
    clear = input("There are models saved in saved_models dir. Clear? [Y/N]: ")

    if clear.lower() == "y":
        for filename in os.listdir(model_dir):
            file_path = os.path.join(model_dir, filename)
            os.remove(file_path)
    else:
        exit()


print(f"{unet_params['channels']=}")
print(f"{unet_params['T']=}")
print(f"{cfg=}")
print(f"{batch_size=}")
print(f"{grad_accum_steps=}")
print(f"{epochs=}")
print(f"{t_max=}")
print(f"{lr=}")
print(f"{min_lr=}")
print(f"{epoch_save_interval=}")


confirmation = input("\nConfirm above hyperparameters [Y/N]: ").lower()
if confirmation != "y":
    exit()



loss_file = "saved_losses.txt"
# Delete previous loss file if exists
if os.path.exists(loss_file):
    os.remove(loss_file)

val_loss_file = "saved_val_losses.txt"
if os.path.exists(val_loss_file):
    os.remove(val_loss_file)

epoch_loss_file = "epoch_loss.txt"
if os.path.exists(epoch_loss_file):
    os.remove(epoch_loss_file)


# Initialization
dataset_loader = DatasetLoader(img_dir=img_dir,
                               prompt_path=prompt_path,
                               batch_size=batch_size,
                               device=device,
                               train_split=0.95
                               )
print(f"\n{len(dataset_loader.train_pairs)=}")
print(f"{len(dataset_loader.val_pairs)=}\n")

model = DiffusionModel(unet_params, diffusion_params).to(device)
compiled_model = torch.compile(model)
# input("NOT USING TORCH COMPILE: Enter: ")

vae: AutoencoderKL = AutoencoderKL.from_pretrained(vae_path).to(device)
vae.requires_grad_(False)
vae.eval()

# Load CLIP tokenizer and text model
clip_tokenizer = CLIPTokenizer.from_pretrained(clip_path, clean_up_tokenization_spaces=False)
clip_text_model = CLIPTextModel.from_pretrained(clip_path).to(device)
clip_text_model.requires_grad_(False)
clip_text_model.eval()

lpips = LPIPS(lpips_path, device).to(device)
lpips.requires_grad_(False)
lpips.eval()



optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.MSELoss()
training_scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)



print("\n")
n_params = sum([p.numel() for p in model.parameters()])
if n_params < 1000:
    print(f"There are {n_params} parameters in the Diffusion Model")
elif 1000 <= n_params < 1e6:
    print(f"There are {n_params / 1000:.1f}K parameters in the Diffusion Model")
else:
    print(f"There are {n_params / 1e6:.1f}M parameters in the Diffusion Model")


n_params = sum([p.numel() for p in vae.parameters()])
print(f"There are {n_params / 1e6:.1f}M parameters in the VAE Model")


n_params = sum([p.numel() for p in clip_text_model.parameters()])
print(f"There are {n_params / 1e6:.1f}M parameters in the CLIP Text Model")


n_params = sum([p.numel() for p in lpips.parameters()])
print(f"There are {n_params / 1e6:.1f}M parameters in the LPIPS Model")
print("\n")



# Training Loop
losses = []
mse_losses = []
percept_losses = []
for epoch in range(epochs):
    step = 0
    start = time.time()
    while dataset_loader.train_epoch == epoch:
        prompt_list, clean_images = dataset_loader.get_batch(train=True)
        clean_images = clean_images.to(device)

        with torch.no_grad():
            latents = vae.encode(clean_images).latent_dist.sample() * 0.18215  # Scale factor from SD

        prompt_tensor = get_prompt_tensor(clip_tokenizer, clip_text_model, prompt_list, device)

        # Sample a random timestep for each latent
        timesteps = torch.randint(0, unet_params["T"], (batch_size,)).long()

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            pred_noise, true_noise = compiled_model(latents, prompt_tensor, timesteps, cfg)
            mse_loss = criterion(pred_noise, true_noise)

            decoded_pred = vae.decode(pred_noise / 0.18215).sample
            decoded_true = vae.decode(true_noise / 0.18215).sample

            decoded_pred = decoded_pred.clamp(-1, 1)
            decoded_true = decoded_true.clamp(-1, 1)

            percept_loss = training_params["pl_coeff"] * lpips(decoded_true, decoded_pred).mean()

            loss = (mse_loss + percept_loss) / grad_accum_steps

        loss.backward()
        losses.append(loss.item() * grad_accum_steps)
        mse_losses.append(mse_loss.item())
        percept_losses.append(percept_loss.item())

        # Update the model parameters with the optimizer
        if (step+1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevents unstable learning
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        with open(loss_file, "a") as f:
            f.write(f"{loss.item() * grad_accum_steps} {mse_loss.item()} {percept_loss.item()}\n")

        step += 1

    step = min(step, 100)  # If there are hundreds or thousands of steps per epoch, using last 100 is good reflection

    if epoch < t_max:
        training_scheduler.step()

    loss_last_epoch = sum(losses[-step:]) / len(losses[-step:])
    mse_loss_last_epoch = sum(mse_losses[-step:]) / len(mse_losses[-step:])
    percept_loss_last_epoch = sum(percept_losses[-step:]) / len(percept_losses[-step:])
    avg_val_loss, all_val_losses = eval_model(model, vae, clip_tokenizer, clip_text_model, lpips, criterion, dataset_loader, batch_size, unet_params["T"], cfg, device)
    print(f"Epoch:{epoch+1}, "
          f"train_loss: {loss_last_epoch:.4f}, "
          f"mse_loss: {mse_loss_last_epoch:.4f}, "
          f"percept_loss: {percept_loss_last_epoch:.4f}, "
          f"val_loss: {avg_val_loss:.4f}, "
          f"lr: {optimizer.param_groups[0]['lr']}, "
          f"time: {int(time.time() - start)}s"
          )

    with open(epoch_loss_file, "a") as f:
        f.write(f"Epoch:{epoch+1}, train_loss: {loss_last_epoch:.4f}, mse_loss: {mse_loss_last_epoch:.4f}, percept_loss: {percept_loss_last_epoch:.4f}, val_loss: {avg_val_loss:.4f}, lr: {optimizer.param_groups[0]['lr']}, time: {int(time.time() - start)}\n")

    with open(val_loss_file, "a") as f:
        for loss in all_val_losses:
            f.write(f"{loss}\n")

    if (epoch+1) % epoch_save_interval == 0:
        avg_train_loss = int(loss_last_epoch * 1000)
        avg_val_loss = int(avg_val_loss * 1000)
        save_output = {"model_state_dict": model.state_dict(),
                       "optimizer_state_dict": optimizer.state_dict(),
                       "training_params": training_params,
                       "unet_params": unet_params,
                       "diffusion_params": diffusion_params,
                       }
        torch.save(save_output, f"{model_dir}/epoch_{epoch+1}_model_tl_{avg_train_loss}_vl_{avg_val_loss}.pth")


avg_loss = int(loss_last_epoch * 1000)
avg_val_loss = int(avg_val_loss * 1000)
save_output = {"model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(),
               "training_params": training_params,
               "unet_params": unet_params,
               "diffusion_params": diffusion_params,
               }
torch.save(save_output, f"{model_dir}/final_model_{avg_loss}_vl_{avg_val_loss}.pth")
