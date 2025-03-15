import os
import cv2
import random
import json
import torch
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from unet import DiffusionModel


def get_prompts(categories: list, use_training_prompts: bool):
    rand_category = random.choice(categories).lower()

    if use_training_prompts:
        with open("training_prompts.json", "r") as f:
            training_prompts_dict = json.load(f)

        return random.choice(training_prompts_dict[rand_category])


    with open("test_prompts.json", "r") as f:
        new_prompts_dict = json.load(f)

    return random.choice(new_prompts_dict[rand_category])



def get_vocab(consecutive_words: int) -> list[str]:
    mapping = {
        1: "consecutive_words/single_word_f10.txt",
        2: "consecutive_words/double_word_f10.txt",
        3: "consecutive_words/triple_word_f10.txt",
        4: "consecutive_words/quad_word_f10.txt",
        5: "consecutive_words/penta_word_f10.txt",
    }

    with open(mapping[consecutive_words], "r") as f:
        vocab = f.read().split("\n")
        vocab = [" ".join(w.split(" ")[:-1]) for w in vocab if len(w) > 0]

    return vocab


def load_esrgan(esrgan_type: str, device: str):
    mapping = {
        "2x": "real_esrgan_weights/RealESRGAN_x2plus.pth",
        "4x": "real_esrgan_weights/RealESRGAN_x4plus.pth",
    }
    esr_scale = 2 if esrgan_type == "2x" else 4

    RRDBNet_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=esr_scale)
    upsampler = RealESRGANer(
        scale=esr_scale,
        model_path=mapping[esrgan_type],
        model=RRDBNet_model,
        half=True if device == "cuda" else False  # Use FP16 on CUDA
    )

    return upsampler


def load_diffusion_model(initial_load: bool, diffusion_model_path: str, device: str):
    if initial_load:  # Meaning this is the initial loading of model (Default load)
        weights = [p for p in os.listdir("diffusion_model_weights") if p.endswith(".pth")]
        try:
            # Checks if model files still follows the expected name format of 'epoch_<ep#>_model_tl_<tl#>_vl_<vl#>.pth'
            # For example, 'epoch_150_model_tl_165_vl_183.pth'
            weights = sorted(weights, key=lambda x: int(x.split("_")[1]))
            diffusion_model_path = weights[-1]  # Use the final ("best") one
        except Exception as e:
            # Incase users changed the filename, just randomly select one as default, until user manually selects another one
            diffusion_model_path = random.choice(weights)
            print("Error in model weights selection due to incorrect filename formatting. Selecting random model"
                  f"\n{diffusion_model_path=}")


    diffusion_model_path = os.path.join("diffusion_model_weights", diffusion_model_path)

    diffusion_model_dict = torch.load(diffusion_model_path, weights_only=False, map_location=device)
    unet_params = diffusion_model_dict["unet_params"]
    diffusion_params = diffusion_model_dict["diffusion_params"]

    diffusion_model = DiffusionModel(unet_params=unet_params, diffusion_params=diffusion_params)
    diffusion_model.load_state_dict(diffusion_model_dict["model_state_dict"])
    diffusion_model.to(device)

    return diffusion_model, diffusion_model_path


def save_images(img_tensors) -> None:
    assert len(img_tensors.shape) == 4, f"Expected img_tensors to be of shape (b, w, h, c) instead got {img_tensors.shape=}"

    downloads_path = "./flask_outputs/batch_gen"

    for it in img_tensors:
        # Iterating over tensor like this would result in (w, h, c)
        img = Image.fromarray(it.to(torch.uint8).numpy())
        filename = get_unique_filename(downloads_path, "custom_img.png")
        img.save(os.path.join(downloads_path, filename))


def create_denoising_video(frame_rate):
    # Set parameters
    image_dir = "./flask_outputs/denoising_temp"
    output_dir = "./flask_outputs/denoising_outputs"
    output_path = get_unique_filename(output_dir, "output.mp4")
    output_path = os.path.join(output_dir, output_path)

    # Get list of images sorted by filename
    images = sorted([img for img in os.listdir(image_dir)], key=lambda x: int(x.split(".")[0]), reverse=True)
    images = [os.path.join(image_dir, img) for img in images]

    # Read the first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
    video = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    # Add images to video
    for img_path in images:
        frame = cv2.imread(img_path)
        video.write(frame)

    # Release video writer
    video.release()
    cv2.destroyAllWindows()


def get_unique_filename(directory, filename):
    # Prevent overwriting previous files. Appends like (x) suffix for filenames as commonly seen
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename

    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base} ({counter}){ext}"
        counter += 1

    return new_filename


if __name__ == "__main__":
    ...
