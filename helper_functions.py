import os
import torch
from diffusers import AutoencoderKL
from PIL import Image
from config import training_params


def save_image_tensor(image_tensor: torch.Tensor, tensor_range: tuple | list, directory: str):
    assert len(image_tensor.shape) == 4
    assert len(tensor_range) == 2

    if tensor_range[0] == 0 and tensor_range[1] == 1:
        image_tensor *= 255
    elif tensor_range[0] == -1 and tensor_range[1] == 1:
        image_tensor += 1
        image_tensor *= (255 / 2)
    else:
        raise ValueError(f"Only range [-1, 1] and [0, 1] allowed\nGiven Range: {tensor_range}")

    assert image_tensor.min() >= tensor_range[0] and image_tensor.max() <= tensor_range[1], \
        "Image tensor values are outside the specified range."

    os.makedirs(directory, exist_ok=True)
    for i in range(image_tensor.shape[0]):
        prediction = image_tensor[i].cpu()  # Move back to CPU

        # Permute tensor from (C, H, W) -> (W, H, C) for easier processing
        prediction = prediction.permute(2, 1, 0)

        # Change to uint8
        prediction = prediction.to(torch.uint8)

        # Convert the mask array to a PIL.Image and save

        Image.fromarray(prediction.numpy()).save(os.path.join(directory, f"img_{i}.png"))


@torch.no_grad()
def eval_model(model, vae: AutoencoderKL, clip_tokenizer, clip_text_model, lpips, criterion: torch.nn.Module, dataset_loader, batch_size: int, T: int, cfg: float, device: str):
    initial_val_epoch = dataset_loader.val_epoch
    all_losses = []

    model.eval()

    while dataset_loader.val_epoch <= initial_val_epoch:  # Go over the entire validation DS
        prompt_list, clean_images = dataset_loader.get_batch(train=False)
        clean_images = clean_images.to(device)

        latents = vae.encode(clean_images).latent_dist.mean * 0.18215  # Scale factor from SD

        prompt_tensor = get_prompt_tensor(clip_tokenizer, clip_text_model, prompt_list, device)

        # Sample a random timestep for each image
        timesteps = torch.randint(0, T, (batch_size,)).long()
        pred_noise, true_noise = model(latents, prompt_tensor, timesteps, cfg)

        mse_loss = criterion(pred_noise, true_noise)

        decoded_pred = vae.decode(pred_noise / 0.18215).sample
        decoded_true = vae.decode(true_noise / 0.18215).sample

        decoded_pred = decoded_pred.clamp(-1, 1)
        decoded_true = decoded_true.clamp(-1, 1)

        percept_loss = training_params["pl_coeff"] * lpips(decoded_true, decoded_pred).mean()

        loss = mse_loss + percept_loss


        all_losses.append(loss.item())


    model.train()

    avg_loss = sum(all_losses) / len(all_losses)
    return avg_loss, all_losses



@torch.no_grad()
def get_prompt_tensor(clip_tokenizer, clip_text_model, prompts, device):
    """

    :param clip_tokenizer:
    :param clip_text_model:
    :param prompts: A list of strings
    :return: Torch tensor of shape (b, 77, 512)
    """

    tokens = clip_tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True).to(device)
    with torch.no_grad():
        outputs = clip_text_model(**tokens)

    return outputs.last_hidden_state




if __name__ == "__main__":
    from transformers import CLIPTokenizer, CLIPTextModel
    import torch
    import time


    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", clean_up_tokenization_spaces=False)
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    result = get_prompt_tensor(tokenizer, text_model, ["hello, how are you?", "hello, I'm good."], "cpu")

    print(type(result))
    print(result.shape)
    print(result)



