import io
import os
import base64
import random
import torch
from PIL import Image
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from flask import Flask, request, jsonify, render_template
from utils import get_prompts, get_vocab, load_esrgan, load_diffusion_model, save_images, create_denoising_video



app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html',
                           vocab=vocab,
                           diffusion_model_list=diffusion_model_list,
                           diffusion_model_path=diffusion_model_path)


@app.route('/help')
def help_page():
    return render_template('help.html')  # Renders the help page


@app.route('/editor')
def editor():
    return render_template('editor.html')


@app.route('/random_prompt', methods=['GET'])
def random_prompt():
    categories = request.args.getlist('categories')
    use_training_prompts = request.args.get('use_training_prompts', 'false').lower() == 'true'
    prompt = get_prompts(categories, use_training_prompts)
    return jsonify({"prompt": prompt})


@app.route('/get_vocab', methods=['GET'])
def get_vocab_route():
    # Return the updated global vocab as JSON.
    return jsonify(vocab)


@app.route('/progress', methods=['GET'])
def get_progress():
    return jsonify({"progress": current_progress})



@app.route('/generate', methods=['POST'])
def generate():
    global current_consecutive_words, vocab, esrgan_type, upsampler, diffusion_model, diffusion_model_path, current_progress

    # Reset current progress
    current_progress = 0

    # Parse inputs
    prompt = str(request.form.get('prompt'))

    batch_size = int(request.form.get('batch_size'))
    denoising_steps = int(request.form.get('denoising_steps'))
    cfg_scale = float(request.form.get('cfg_scale', 4.5))
    img2img = False
    img_strength = float(request.form.get('strength', 0.5))
    noise_img2img = request.form.get('noise_image') is not None
    inpaint = False
    mask = None

    seed_str = request.form.get('seed', '').strip()  # Get the seed as a string and remove whitespace
    chosen_model = request.form.get('diffusion_model_type')
    sampling_method = request.form.get('sampling_method')
    real_esrgan = request.form.get('esrgan')
    consecutive_words = int(request.form.get('consecutiveWords'))
    realtime_denoise = request.form.get('realtime_denoise') is not None
    video_fps = int(request.form.get('video_fps', 60))


    if consecutive_words != current_consecutive_words:
        current_consecutive_words = consecutive_words
        vocab = get_vocab(current_consecutive_words)

    if real_esrgan != esrgan_type and real_esrgan != "None":
        print(f"Now using upsampler type as : {real_esrgan}")
        esrgan_type = real_esrgan
        upsampler = load_esrgan(esrgan_type, device)

    if real_esrgan == "None":
        print(f"Now using Upsampler=None")
        upsampler = None

    if chosen_model != diffusion_model_path:
        diffusion_model_path = chosen_model
        diffusion_model, _ = load_diffusion_model(initial_load=False, diffusion_model_path=diffusion_model_path, device=device)
        print(f"Now using {diffusion_model_path=}")

    if seed_str.isnumeric():
        seed = int(seed_str)
    else:
        seed = random.randint(-10_000, 10_000)

    # Handle image uploads (for img2img and inpainting)
    if 'image' in request.files and request.files['image'].filename != '':
        image_file = request.files['image']
        image = Image.open(image_file).convert("RGB")
        img2img = True

    if 'mask' in request.files and request.files['mask'].filename != '':
        mask_file = request.files['mask']
        mask = Image.open(mask_file).convert("L")  # Grayscale mask
        inpaint = True
        noise_img2img = True  # If inpainting, this should be True

    # Debugging: Print parsed inputs
    print(f"Prompt: {prompt}")
    print(f"BS: {batch_size}")
    print(f"{real_esrgan=}")
    print(f"{esrgan_type=}")
    print(f"Seed: {seed}")
    print(f"Realtime denoise: {realtime_denoise}")


    gen.manual_seed(seed)
    use_ddim = sampling_method == "ddim"

    # Clear the temp_dir if realtime_denoising is checked
    if realtime_denoise:
        for files in os.listdir("./flask_outputs/denoising_temp"):
            os.remove(os.path.join("./flask_outputs/denoising_temp", files))


    # The starting timestep of the diffusion process. Default is 1000 (999 for 0 index), adjusted to other values if using img2img
    starting_step = 999

    # Init with None values
    img_warning = None
    msk_warning = None
    clean_latents = None

    # First, generate the latent image (Gaussian Noise if not using img2img/inpainting else use provided image)
    # Scale by 0.18215 for VAE
    if img2img:
        try:
            # .getdata() returns a list of tuples, where len of list == width * height, len of each tuple is 3
            # Essentially shape == (w*h, 3), then reshape to (1, w, h, 3) and permute to (1, 3, h, w)
            # If no noise added, the first decoded latent in realtime denoising should == input image
            image = torch.tensor(image.getdata(), dtype=torch.float32).reshape(1, image_size, image_size, 3).permute(0, 3, 2, 1)
            image = ((image / 255) - 0.5) * 2
            clean_latents = vae.encode(image).latent_dist.sample() * 0.18215  # Scale factor for VAE from SD
            clean_latents = clean_latents.repeat(batch_size, 1, 1, 1)  # Repeat it across batch dimension

            # Next, we add noise according to the img_strength slider

            # So if img_strength == 0.1, then starting_step = 1000 - 100 = 900, start at t=900
            # meaning it won't be as strongly conditioned on the given image
            # Conversely if img_strength == 0.9, then starting_step = 100, which would introduce minor changes to the img
            starting_step = 1000 - (1000 * img_strength)
            latents = clean_latents.clone().to(device)

            if noise_img2img:
                timesteps = torch.tensor([starting_step] * batch_size, device=device).long()
                # Apply noise using the forward diffusion method and calculated timesteps
                latents, _ = diffusion_model._forward_process(latents, timesteps)

        except Exception as e:  # Exception is most likely due to img not being 128x128 in shape
            latents = torch.randn((batch_size, 4, image_size // 8, image_size // 8), generator=gen,
                                  device=device) * 0.18215
            img_warning = f"Warning: The provided image failed to be processed (likely due to size mismatch). Generating image from noise instead.\nWarning: {e=}"
    else:
        latents = torch.randn((batch_size, 4, image_size // 8, image_size // 8), generator=gen, device=device) * 0.18215


    if inpaint:
        try:
            # Assert an image is actually given
            assert img2img, "Need to submit an image for img2img to inpaint"

            # Assert a valid image was given to condition on
            assert img_warning is None, "Invalid image was given, no inpainting will be done."

            # Assert shape
            assert mask.size == (image_size, image_size), f"Image should be of shape ({image_size}, {image_size}), instead got {mask.size=}"

            # Resize into shape of VAE, and make sure to use nearest neighbor, otherwise error will be thrown
            mask = mask.resize((image_size//8, image_size//8), Image.Resampling.NEAREST)

            # Convert into tensor
            mask = torch.tensor(mask.getdata(), dtype=torch.int32)

            # Assert valid values
            unique_vals = set(mask.flatten().tolist())
            assert len(unique_vals) == 2 and (sorted(unique_vals)[0] == 0 and sorted(unique_vals)[1] == 255), \
                f"This should only be a binary mask of 0s and 255s! Got {unique_vals=}"


            # Reshape into (b=1, w=image_size, h=image_size, c=1) then permute to (b, c, h, w)
            mask = mask.reshape(1, image_size//8, image_size//8, 1).permute(0, 3, 2, 1)

            # Replace all 255s with 1s
            mask = mask.masked_fill(mask == 255, 1)

            mask = mask.repeat(batch_size, 1, 1, 1)  # Repeat it across batch dimension

        except Exception as e:
            msk_warning = f"Warning: The provided mask failed to be processed (likely due to size mismatch or invalid values). Mask will not be used.\nWarning: {e=}"

    if msk_warning is not None:
        mask = None

    # gen_image returns a tuple of image tensors of shape (w, h, c) and (b, w, h, c) respectively
    generated_image, remaining_img = gen_image(latents, clean_latents, mask, prompt, starting_step, denoising_steps, cfg_scale, use_ddim, realtime_denoise)

    # Save the additional images (when batch_size > 1) if any
    save_images(remaining_img)

    generated_image = Image.fromarray(generated_image.to(torch.uint8).numpy())

    # Convert image to base64 for display
    buffered = io.BytesIO()
    generated_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    if realtime_denoise:  # Create the video
        create_denoising_video(video_fps)


    response_data = {"image": img_str}

    if img_warning is not None:
        response_data["img_warning"] = img_warning

    if msk_warning is not None:
        response_data["msk_warning"] = msk_warning

    return jsonify(response_data)


def gen_image(latents, clean_latents, mask, prompt, starting_step, denoising_steps, cfg_scale, use_ddim, realtime_denoise):
    prompt_tensor = get_prompt_tensor(prompts=[prompt], batch_size=latents.shape[0])
    img_tensor = diffusion_model.generate(vae=vae,
                                          latents=latents,
                                          clean_latents=clean_latents,
                                          inpainting_mask=mask,
                                          prompt_tensor=prompt_tensor,
                                          t=starting_step,
                                          num_steps=denoising_steps,
                                          cfg_scale=cfg_scale,
                                          use_ddim=use_ddim,
                                          device=device,
                                          realtime_denoise=realtime_denoise,
                                          progress_callback=update_progress,
                                          )

    # Remove the batch dimension
    if upsampler is not None:
        img_tensor = upscale_tensor(img_tensor)
    else:
        # Need manually permute here if not using upsampler
        # (b, c, h, w) -> (b, w, h, c) for PIL.Image conversion
        img_tensor = torch.clip(img_tensor.permute(0, 3, 2, 1) * 255, min=0, max=255)

    display_img = img_tensor[0]  # Image to display on Flask Server
    remaining_img = img_tensor[1:]  # Remaining images to be saved (only occurs when batch_size > 1)

    # Shape would be (w, h, c) and (b-1, w, h, c) respectively
    return display_img, remaining_img


def upscale_tensor(input_tensor: torch.Tensor):
    assert len(input_tensor.shape) == 4, f"Expected input_tensor to be of shape (b, c, h, w) instead got {input_tensor.shape=}"
    # input_tensor = input_tensor.to(device)  # Move to device

    # Convert tensor to numpy array after permuting to (h, w, c)
    # input_img = input_tensor.permute(1, 2, 0).cpu().numpy()
    input_img = input_tensor.permute(0, 2, 3, 1).cpu().numpy()
    input_img = (input_img * 255.0).clip(0, 255).astype('uint8')

    # Since upsampler doesn't take in multiple images, need to iterate over
    temp_holder = []
    for tensor in input_img:
        output_img, _ = upsampler.enhance(tensor)
        temp_holder.append(torch.tensor(output_img))

    # Then stack along batch dim
    output_img = torch.stack(temp_holder, dim=0)

    # Convert it into (w, h, c) which is the format that PIL.Image expects to convert into Image object
    # output_tensor = torch.tensor(output_img).permute(1, 0, 2).float()
    output_tensor = torch.tensor(output_img).permute(0, 2, 1, 3).float()

    return output_tensor


def update_progress(value):
    global current_progress
    current_progress = 1000 - value
    print(current_progress)


@torch.no_grad()
def get_prompt_tensor(prompts, batch_size):
    """
    :param prompts: A list of strings
    :return: Torch tensor of shape (b, 77, 512)
    """

    tokens = clip_tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True).to(device)
    with torch.no_grad():
        outputs = clip_text_model(**tokens)

    return outputs.last_hidden_state.repeat(batch_size, 1, 1)


@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    """Returns up to 5 matching words from the vocab_list based on the query."""
    query = request.args.get('q', '').strip().lower()
    if not query:
        return jsonify([])  # Return empty list if no query

    # Filter vocab for words that start with the query
    matches = [word for word in vocab if word.lower().startswith(query)]
    # Limit to top 5
    matches = matches[:5]

    return jsonify(matches)



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae_path = "stabilityai/sd-vae-ft-ema"
    clip_path = "openai/clip-vit-base-patch32"

    image_size = 128  # Image dimension. Default value is 128 (base 2 square images)
    current_progress = 0  # Value for progress bar

    assert os.path.isdir("diffusion_model_weights")  # Make sure that this folder exists, as it holds the diffusion model weights

    diffusion_model_list = [p for p in os.listdir("diffusion_model_weights") if p.endswith(".pth")]
    assert len(diffusion_model_list), ".pth file containing model weights should exist in the folder!"

    # Checks if model files still follows the expected name format of 'epoch_<ep#>_model_tl_<tl#>_vl_<vl#>.pth'
    # For example, 'epoch_150_model_tl_165_vl_183.pth'
    try:
        diffusion_model_list = sorted(diffusion_model_list, key=lambda x: int(x.split("_")[1]))
    except Exception as e:  # Otherwise just sort it lexicographically
        diffusion_model_list = sorted(diffusion_model_list)


    # Next, create the outputs folder
    os.makedirs("flask_outputs", exist_ok=True)  # Holds outputs when running Flask
    os.makedirs("./flask_outputs/batch_gen", exist_ok=True)  # Stores output image when setting batch size > 1
    os.makedirs("./flask_outputs/denoising_outputs", exist_ok=True)  # Stores denoising output videos
    os.makedirs("./flask_outputs/denoising_temp", exist_ok=True)  # Stores temp denoising images used to compile final video


    # Create a Generator for seed setting by users if desired
    gen = torch.Generator()


    diffusion_model, diffusion_model_path = load_diffusion_model(initial_load=True, diffusion_model_path=None, device=device)
    print(f"Initial Diffusion Model Path: {diffusion_model_path}")

    n_params = sum([p.numel() for p in diffusion_model.parameters()])
    print(f"There are {n_params / 1e6:.1f}M parameters in the Diffusion Model")


    vae: AutoencoderKL = AutoencoderKL.from_pretrained(vae_path).to(device)
    vae.requires_grad_(False)
    vae.eval()

    # Load CLIP tokenizer and text model
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_path, clean_up_tokenization_spaces=False)
    clip_text_model = CLIPTextModel.from_pretrained(clip_path).to(device)
    clip_text_model.requires_grad_(False)
    clip_text_model.eval()

    # Load in Real-ESRGAN
    esrgan_type = "2x"
    upsampler = load_esrgan(esrgan_type, device)

    current_consecutive_words = 1
    vocab = get_vocab(current_consecutive_words)


    app.run(debug=True)
