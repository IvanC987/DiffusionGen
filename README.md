# Diffusion Model Interface

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Observations](#observations)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

(At the moment, this is the draft/informal version of the readme. Will update later on once final is completed.)


## Introduction
This repository contains a web-based interface for interacting with a Latent Diffusion Model (LDM). Users can generate images from text prompts, upload images for transformation (Img2Img), perform inpainting, and apply (pseudo) real-time denoising.

![GUI Image](readme_images/GUI_SS1.png)



## Features
- Text-to-Image generation using Latent Diffusion Models
- **Img2Img:** Modify existing images using diffusion processes
- **Inpainting:** Remove and fill missing parts of an image with AI-generated content
- **Real-Time Denoising:** View intermediate diffusion steps in near-real-time
- Configurable advanced parameters (Batch Generation, CFG Scale, Denoising Steps, Sampling Methods (DDPM/DDIM), among many others)
- Light/Dark mode UI toggle
- Progress tracking with a live updating bar
- Download and save generated images

---

--To Adjust Later--
--------------------------------------

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch
- torchvision
- Flask
- Diffusers library
- Transformers (for CLIP text embeddings)

### Steps
1. **Clone the repository**
   ```sh
   git clone https://github.com/IvanC987/DiffusionGen
   ```

2. Due to GitHub restrictions, I have separated the Diffusion model weights and stored it in my HuggingFace Repo, which can be accessed at:
`https://huggingface.co/DL-Hobbyist/DiffusionGen/tree/main/inference/diffusion_model_weights`
Since each file is ~4.7GB, I would recommend choosing the 'best' version, epoch 375.
Download whichever ones you would like to play around with and place them within the `DiffusionGen\inference\diffusion_model_weights` folder

3. Install all the required packages in requirments.txt

4. CD into `DiffusionGen` and run `python3 inference/app.py`


Have Fun!


*Important Notes- 
1. You may need to authenticate and log into your huggingface account via `from huggingface_hub import login` as the OpenAI CLIP model and Stable Diffusion's VAE is pulled from HuggingFace
2. It is likely that the error `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'` will pop up. It's due to deprecation of the package, to fix, simply navigate to the file `basicsr\data\degradations.py`
And change the import from:
`from torchvision.transforms.functional_tensor import rgb_to_grayscale`
to
`from torchvision.transforms.functional import rgb_to_grayscale`




--------------------------------------


## Usage

### Running the Interface
1. Enter a **text prompt** to generate an image.
2. Adjust **CFG scale**, **Denoising Steps**, and **Batch Size** as needed.
3. Click **Generate** to create an image.

### Using Img2Img
- Upload an image and set the **strength** (higher values introduce more changes).
- Click **Generate** to transform the image.

### Real-Time Denoising
- Check the **Real-Time Denoising** box.
- The intermediate steps of the denoising process will be saved within the `inference\flask_outputs\denoising_temp` folder as individual images, where it would automatically compiled into a video of chosen frame rate.

### Inpainting
- Upload an image and a binary mask to specify areas to fill.
- Click **Generate** for inpainting.

## Configuration
- **Batch Size:** Number of images to generate at once.
- **CFG Scale:** Classifier-Free Guidance Scale (higher = more influence from prompt).
- **Denoising Steps:** Number of steps in the reverse diffusion process.
- **Strength (Img2Img):** Determines how much of the original image is preserved.
- **Sampling Method:** Choose between `DDPM` (stochastic) and `DDIM` (deterministic).


## Additional Feature
Users can:
- Choose to generate random prompts, which can be chosen from the training data or unseen data by the model (through the `Use Training Prompts` checkbox)
- Select/Deselect certain categories for random prompt generation
- Select different variations of downloaded diffusion models
- Fix random seed (Make DDIM completely deterministic)
- Create custom binary mask for inpainting
- Choose whether or not to use the `Real-Time Denoising` feature and specify the frame rate of the final compiled video (60 fps by default)
- And others.


### Notes:
- This is a fairly small Latent Diffusion Model (~370M Parameters), hence it would be able to run fairly well even on a CPU (Though limited to DDIM. GPU is recommended for DDPM)
- Regarding RealTime-Denoise: I originally intended for it to be actually real time in the GUI, but couldn't quite get the front end HTML/JS to work properly. Had to settle by saving the intermediate images to a temp folder and create a video out of it. So...'semi' RealTime-Denoise
- If using RealTime-Denoise, would recommend using the full DDPM with 1000 steps or DDIM with sufficient steps. As the final compiled video's length is directly dependent upon the specified frame rate and # of denoising steps
- There are certain limitations of this model, which is detailed in the [Observations](#observations) section below





## Model Training

This will be a very comprehensive and detailed explanation of the backend. 
I've split it into multiple sections, namely
1. Dataset procurement/composition
2. Model architecture/unet&diffusion hyperparameters
3. Training process
4. Intermediate and final results


Here goes. 



1. Dataset Procurement and Composition

Initially when I first started this project, one of the first things I thought of was the dataset. 
What kind of dataset should I use? 
Where do I get it? 
The size of dataset? (e.g. number of text-images in the dataset)
Quality? (This is of utmost concern)
Resolution? (For this project, all image resolution should be fixed at 1:1 ratio)
Style? (Realism, Abstract, Concept Art, Anime, etc.)
Among others. 


I was looking through various dataset that HuggingFace offers, and although there are a lot of them, I couldn't quite decide on a particular dataset. 
Mainly due to the following criterias: 
1. Limited Computational resources. To train a model of this caliber, it must have at least ~10k or so images (my estimate), but should be less than 1 million images (Too large is computationally infeasible for me)
2. Quality. The dataset shouldn't be mix and match type, where it have images of various resolution. Of course, using the PIL library to resize the image is an alternative, but that would likely degrade the quality of the dataset. Image resolution can't be too low either, as that will be detrimental to the model considering the limited number of training samples
3. Style. While looking through various datasets, I realized that there are various styles of images. Like realism (images taken in real life), Concept Art, Anime, Cartoon, etc. I would need a dataset that focuses on a particular style, as having multiple styles would likely confuse the model during training (again, this likely wouldn't be a problem using a sufficiently powerful model/large dataset, but it's a problem here)
4. Diversity. Limited by the number of training images, I would prefer a dataset that's as diverse as possible, but still have sufficient number of images per category (e.g. Humans like Man, Woman, Girl, Boy, etc., and Animals like Dogs, Cats, Fishes, etc.) so that the model can generalize well enough
Among various other filters/considerations I had. 

In the end, I realized it wasn't very feasible to find my "ideal" dataset. 
So the alternative? Stable Diffusion. 

Thank God for SD being open source. 

I realized that although it may cost a bit, renting a GPU and pulling `stabilityai/stable-diffusion-3.5-large` from HF then generate images from it is possible. 
A synthetic dataset. It may just work...

After some consideration, I decided to proceed with that route due to the many benefit it provides. 
1. I have control of what kind of images it produces and can specify things like category, resolution, # of images generated, and such.
2. Can choose the style of generated images. So I was wondering whether or not realistic images may impact the model's performance.
           - This mainly stems back when I was testing out DALLE from OpenAI. When I requested it to generate images, the resulting images were always of something like a cartoon-ish style. Where the image looks smooth with flowing gradients. Why not generate realistic images rather than ones like these? 
          - My personal speculation was that it was because perhaps training a generative model on images like those is easier than realistic images. As in, with realisitc images, there are problems of things like JEPG compression, Noises, Blurs, etc, in addition to it being very "complex". Where it has sharp lines, contrastive edges, etc., However cartoon-ish/anime like images doesn't quite habe that problem. Not only does it look somewhat better at times (subjective opinion), it would make sense that it's easier to train on.
          - Anyways, that's just my speculation. Taking that into account, that was another reason why I chose a syntehtic dataset using SD


Anyways. Now that I decided to use SD, my next question would be what kind of images do I want. 

It'll take too long to go over everything I was thinking of, but in short, I chose the following categories/subcategories

Humans- 10k Images (Man, Woman, Boy, Girl, Teen, Guy, Kid)
Animals- 5k Images (Dog, Cat, Fish, Bird, Horse, Tiger, Wolf, Panda, Rhino, Whale)
Mythical Beings- 5k Images (Qilin, Leviathan, Dragon, Fairy, Phoenix, Mermaid)
Scenery- 5k Images (Desert, Rainforest, Mountain Lake, Snowy Mountain, Tropical Island, Deep Sea, Night Sky, Glacier, Volcano, Aurora Borealis, Underwater Cave, Savannah)
 

Above number of images per categories are approximates. 
In actuality, there's around 23.5k images total, not exactly 25k. I won't go into the details. 

Anyways, the ratio of Humans compared to other three is twice the amount. 
That's because I was thinking,
1. Generation of human figures might be more common compared to other categories
2. Human figures are hard to get correct

And through testing, it seems that the latter is correct (Though the former depends on the user)
Will touch more on this later on. 
There's still a lot of more details that was omitted for the sake of conciseness. 
Next is model architecture. 



2. Model Architecture
After going through multiple sources and a few weeks of research of Diffusion Models and how they work, I decided to implement a variation of my own (Which is definitely worse than production levels, but just copying and pasting defeats the purpose of this project)

(Note that the core of the Diffusion Model is the UNet and that the following explaination may be techical. It would take too long to explain eveyrthing in detail)

I gathered everything together and built the UNet with the following blocks: 
TimeEncoder- This class returns the encoding of the chosen timestep. Since it's quite similar to the positional encoding used in `Attention Is All You Need`, I just decided to use that 
EncoderBlock- This contains 
BottleNeck- 
DecoderBlock-

At the very end there's a sequential layer made of GroupNorm, SiLU, and Conv2D that returns the image (latent) tensor back into the original input tensor shape


Hyperparameters used here is fairly common, won't go into that. 
Located in `config.py`


3. Training Process
The training hyperparameters are also located in `config.py`

Before I started the official training, I artificially increased the dataset via data augmentation, primarily using horizontal flip and color adjustments. 
So the final dataset was 4x the original size, composed of.
1. Original Image
2. Original Image + Color Adjustment
3. Horizontal Flip
4. Horizontal Flip + Color Adjustment

The corresponding prompt was copied 3 times. The two primary benefits of this is that
1. Dataset is increased by 3x. Instead of ~23.5k images, it is now ~94k images. Considering the loss function is primarily based on MSE, this greatly helped the training
2. There is now a 1-to-4 relationship between the text prompt and the images. For example, if a prompt is "A dog running across a bright green lawn", then there would be 4 images that corresponds to that. This would help the model generalize better, since all four images are valid for that prompt. Note that horizontal flip can only be used where images are not directionally dependent (e.g. Images containing texts)


During the training process, the model would get a batch of randomly chosen images and encode via SD's VAE, where the tensor dimensions are (batch, channel, height, width)
For example, input tensor shape might be (128, 3, 1024, 1024)
output would then be (128, 4, 128, 128)

Here's an example (Since VAE encodes image from 3 channels, RGB, to 4 latent channels, it doesn't quite do it justice to try to represent the latent channels as RGB, but this just kind of "shows" what it does)



![Original Image](readme_images/original_img.png)

![VAE Encoded Image](readme_images/vae_encoded.png)



During the training, I have stored the outputs below:


I have included the three loss files.
`epoch_loss.txt`- Details the epoch number, training loss, mse_loss, perceptual loss, validation loss, learning rate, and time taken (in seconds)
`final_custom_losses.txt`- Stores space separated loss values for training loss, mse_loss, perceptual loss per training step
`final_custom_val_losses.txt`- Stores the validation loss per training step

(For mse_loss and perceptual loss, I've accidentally scaled it by 4x. However all other losses are correct.)


After training the model for about 320 GPU hours on a single A40 (~14 days), here are the results: 


Plot of Training and Validation Loss:
![Train and Val Loss Image](readme_images/train_val_loss.png)



Overall, the training progress was surprisingly stable. 
Like most models first few epochs drastically reduces the loss. 
Then for the next ~200 or so epochs it stabilizes and loss decreases linearly, which is surprising.
At around epoch 50, the loss starts to diverge, however they are still proportionally decreasing. 
After ~200 epoch mark, it starts to plateau, which is to be expected. Although it's hard to tell, however both losses are slighly decreasing for the next ~180 epochs.
However due to time and computational limitations, I had to stop training and cut it off at epoch 380, where the most recent version, epoch 375 is currently the best in both losses.


This graph was plotted using the two final loss files, where I averaged every 19 step's losses into a single value (since ratio of training to validation was approximately 19:1)
However the lines still fluctuate too much and so I further averaged 25 values into a single scalar. 

(The graph seemed to be right-shifted by a slight margin, not sure why)





## Observations

When testing out the final model, I noticed the following: 

1. Output quality varies across categories 
When users enter a prompt in the 'Scenery' category, usually the generated image would look decently better than those of other categories (Humans, Animals, and Mythical Beings)
I attributed this to our internal bias and model objective. 
Whether or not an image looks "good" is inherently subjective to us. If a given prompt is something like "A horse galloping across a grassland...", then we would expect to see an image where the main subject is a horse galloping, and grassland as background. 
However if an image is generated, one would notice that the 'horse' would be somewhat blurry/lower quality, relative to the background.
Recall that the training objective of the model is the MSE loss. The way that the loss is calculated would make the model weight all the pixels equally. The "subject" would take up less area of the image compared to the background. 
Hence generally the background, when relatively uniform, would look better compared to the main subject. 
This is esepcially the case when comparing text prompts of small subjects like cat vs larger subjects like rhino. The former is noticeably more blurred compared to the latter. 

This is the primary reason why the I made the 'Humans' category of the dataset double the size of other categories. 
Although people take up a fair chunk of the total pixels, majority of the time it is around ~20%. The rest is all common background. 
Assuming 20%, we ourselves would be especially biased towards certain parts of the generated person in question. For example, it is commonly the case that we would scrutinize the facial feature of the person in the image compared to other body parts (like arms, legs, body, etc.)
And the final result proved that this was indeed the case. 
The model would weight the facial feature of the humans equally with all other pixels, like background and such, and so it would be especially blurred. 
...


2. Diversity of output images
In general, when generating images via models like Stable diffusion, the output would be quite diverse given the same prompt. 
This is due to how the initial image is created and process of image generation. 
However this model's output is relatively fixed. When giving the model the same prompt, the output would very different, but highly similar. 

I suspect this is due to the dataset, rather than model architecture/training itself. 
Assume the dataset has 25k images for simplicity. 
10k is Humans
5k for each of the remaining categories (Animals, Mythical Beings, Scenery)

Take a look at for example, scenery. 
There are 12 subcategories, (Desert, Rainforest, Mountain Lake, Snowy Mountain, Tropical Island, Deep Sea, Night Sky, Glacier, Volcano, Aurora Borealis, Underwater Cave, Savannah),

The prompts are selected at random through uniform distribution, and so it's fair to assume that in the entire dataset would have
approximately 420 images per subcategory. This is extremely low, nearly the bare minimum needed to even train a diffusion model I would say.
(There's also the problem of diversity of descriptive adjectives in textual prompts, but that will take a while to explain)

Anyways, 420 image per category. That is very low. 
And so you can imagine how limited the diversity of the dataset is. 
Stable Diffusion is trained on the LAION dataset, in the magnitude of billions of images, whereas this uses 100k (with augmentation). So it's nowhere near comparable, in terms of dataset size. 

TLDR: Limited dataset -> Limited Output


3. Img2Img
This feature is very interesting, however due to the problem mentioned above, the output is often blurry and of low quality. 
Unfortunately that can't be helped. This stems from the training dataset, rather than from implementation, so...yeah


4. Inpaininting
Likewise, this is also somewhat forced. Viewed in real-time denoising it's fairly intriguing, but outputs are quite....bad. 
Welp. 


Think there's a few more. Will add those later on. 


And so this concludes the informal version of readme. The final version should be completed in another day or two. 





## Acknowledgments
This project utilizes:
- [Variational Autoencoder (VAE)](https://huggingface.co/stabilityai/sd-vae-ft-ema) from Stable Diffusion for encoding and decoding images in the latent space.
- [CLIP (Contrastive Language-Image Pretraining)](https://huggingface.co/openai/clip-vit-base-patch32) from OpenAI for text-to-image embedding.
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for upscaling.
- [Perceptual Loss (LPIPS)](https://github.com/richzhang/PerceptualSimilarity) for evaluating image similarity.


This project is inspired by and builds upon the foundational research from the following papers:

- **High-Resolution Image Synthesis with Latent Diffusion Models**  
  *Rombach et al.*  
  [🔗 arXiv:2112.10752](https://arxiv.org/abs/2112.10752)

- **Denoising Diffusion Probabilistic Models**  
  *Ho, Jain, Abbeel*  
  [🔗 arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

- **Denoising Diffusion Implicit Models**  
  *Song, Meng, Ermon*  
  [🔗 arXiv:2010.02502](https://arxiv.org/abs/2010.02502)


Finally, huge thanks to Umar Jamil with his [Stable Diffusion Video](https://www.youtube.com/watch?v=ZBKpAp_6TGI&t=9117s) and ExplainAI's [Stable Diffusion Video](https://www.youtube.com/watch?v=hEJjg7VUA8g&list=PL8VDJoEXIjpo2S7X-1YKZnbHyLGyESDCe)


## License
This project is licensed under the **MIT License**.


