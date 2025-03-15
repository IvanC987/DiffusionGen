import os
import cv2
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet



# Based on https://github.com/xinntao/Real-ESRGAN



# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = "external_model_weights/RealESRGAN_x2plus.pth"  # Ensure this exists
input_folder = "t1"  # Folder containing images
output_folder = "t3"  # Folder to save results
outscale = 2  # Upscaling factor
tile = 0  # Set to >0 if CUDA memory is an issue

# Create output directory if it doesn’t exist
os.makedirs(output_folder, exist_ok=True)

# Load Model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=outscale)
upsampler = RealESRGANer(
    scale=outscale,
    model_path=model_path,
    model=model,
    tile=tile,
    half=True if torch.cuda.is_available() else False  # Use FP16 on CUDA
)

# Process Images
for file_name in os.listdir(input_folder):
    if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue  # Skip non-image files

    img_path = os.path.join(input_folder, file_name)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    output, _ = upsampler.enhance(img, outscale=outscale)
    save_path = os.path.join(output_folder, f"upscaled_{file_name}")
    cv2.imwrite(save_path, output)
    print(f"Saved: {save_path}")


print("Upscaling complete!")
