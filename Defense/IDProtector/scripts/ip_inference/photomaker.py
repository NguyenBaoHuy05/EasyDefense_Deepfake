# %%
import torch
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
import argparse
import os

from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, DDIMScheduler
from huggingface_hub import hf_hub_download

from generation_methods.PhotoMaker.photomaker import PhotoMakerStableDiffusionXLPipeline

from IPython.display import display

# %%
base_model_path = 'SG161222/RealVisXL_V3.0'
device = "cuda"

# %%
parser = argparse.ArgumentParser()

parser.add_argument('--source_dir', type=str, required=True)
parser.add_argument('--destination_dir', type=str, required=True)
parser.add_argument('--num_inference_steps', type=int, default=50)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--prompt', type=str, default='A person img, realistic, high quality')

args = parser.parse_args()

# %%
if not os.path.exists(args.destination_dir):
    os.makedirs(args.destination_dir, exist_ok = True)

files = [file for file in os.listdir(args.source_dir) if file.lower().endswith(('.png', '.jpg'))]

# %%
from huggingface_hub import hf_hub_download

photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")

pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="fp16",
).to(device)
pipe.set_progress_bar_config(disable=True)

pipe.load_photomaker_adapter(
    os.path.dirname(photomaker_ckpt),
    subfolder="",
    weight_name=os.path.basename(photomaker_ckpt),
    trigger_word="img"
)
pipe.id_encoder.to(device)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.fuse_lora()

# %%
style_strength_ratio = 20
start_merge_step = int(float(style_strength_ratio) / 100 * args.num_inference_steps)
if start_merge_step > 30:
    start_merge_step = 30

# prompt = "a half-body portrait of a man img wearing the sunglasses in Iron man suit, best quality"
prompt = args.prompt
negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth, grayscale"
generator = torch.Generator(device=device).manual_seed(args.seed)

# %%
for file in tqdm(files):
    # Load image
    ip_image = Image.open(os.path.join(args.source_dir, file)).convert('RGB')

    # Generate
    generated_image = pipe(
        prompt=prompt,
        input_id_images=ip_image,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        num_inference_steps=args.num_inference_steps,
        start_merge_step=start_merge_step,
        generator=generator,
    ).images[0]

    # Save image
    save_path = os.path.join(args.destination_dir, file)
    generated_image.save(save_path)

# %%



