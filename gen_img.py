import os
# from unet_clip import DiffusionModel
import time
import torch
from PIL import Image
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from helper_functions import get_prompt_tensor
from unet import DiffusionModel
from config import clip_path


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Currently using {device=}\n")


def use_test_prompts():
    with open("temp.txt", 'r') as f:
        d = f.read().split("\n")
        d = [i for i in d if len(i) > 0]


    return d


prompts = use_test_prompts()
model_path = r"inference/diffusion_model_weights/epoch_375_model_tl_132_vl_155.pth"
img_dim = 128
num_images = 1



save_img_dir = "eval_images"
os.makedirs(save_img_dir, exist_ok=True)

if len(os.listdir(save_img_dir)) > 0:
    clear = input("There are images in eval_images dir. Clear? [Y/N]: ")

    if clear.lower() == "y":
        for filename in os.listdir(save_img_dir):
            file_path = os.path.join(save_img_dir, filename)
            os.remove(file_path)
    else:
        exit()



def tensor_to_img(tensor: torch.Tensor, output_dir):
    assert len(tensor.shape) == 4

    if tensor.max().item() <= 1 and tensor.min().item() >= 0:
        tensor *= 255
        print("[0, 1]")
    elif tensor.max().item() <= 1 and tensor.min().item() >= -1:
        tensor += 1
        tensor *= (255 / 2)
        print("[-1, 1]")
    else:
        raise ValueError(f"{tensor.max()}, {tensor.min()}")


    for i in range(tensor.shape[0]):
        prompt_idx = i//num_images
        prompt = prompts[prompt_idx]

        prediction = tensor[i]

        prediction = prediction.cpu()  # Move back to CPU

        # Permute tensor from (C, H, W) -> (W, H, C) for easier processing
        prediction = prediction.permute(2, 1, 0)  # Shape: (H, W, C)

        prediction = prediction.to(torch.uint8)
        prediction = torch.clamp(prediction, 0, 255)

        print(os.path.join(output_dir, f"img_{i}_{prompt[:40]}.png"))

        # Convert the mask array to a PIL.Image
        Image.fromarray(prediction.numpy()).save(os.path.join(output_dir, f"img_{i}_{prompt[:40]}.png"))




model_data = torch.load(model_path, weights_only=False, map_location=device)
unet_params = model_data["unet_params"]
diffusion_params = model_data["diffusion_params"]

model = DiffusionModel(unet_params, diffusion_params).to(device)
model.load_state_dict(model_data["model_state_dict"])
# model = torch.load(model_path, weights_only=False, map_location=device).to(device)


vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
vae.requires_grad_(False)

clip_tokenizer = CLIPTokenizer.from_pretrained(clip_path, clean_up_tokenization_spaces=False)
clip_text_model = CLIPTextModel.from_pretrained(clip_path).to(device)



prompts_list = [[p] * num_images for p in prompts]




n_params = sum([p.numel() for p in model.parameters()])
print(f"There are {n_params / 1e6:.1f}M parameters in the Diffusion Model")




print("Starting")
start = time.time()
result = None
for p in prompts_list:
    prompt_tensor = get_prompt_tensor(clip_tokenizer, clip_text_model, p, device)
    img_tensor = model.generate(vae, prompt_tensor, num_images, img_dim, 1000, 4.5, device)

    print(round(time.time() - start, 2))
    start = time.time()
    print(p[0])
    print("\n---------------\n")

    result = img_tensor if result is None else torch.cat((result, img_tensor), dim=0)


tensor_to_img(result, save_img_dir)

