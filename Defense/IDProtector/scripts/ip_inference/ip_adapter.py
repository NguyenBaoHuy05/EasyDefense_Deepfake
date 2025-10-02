# %%
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
from tqdm import tqdm
import argparse
import os

from generation_methods.IP_Adapter.ip_adapter import IPAdapter, IPAdapterPlus

IP_ADAPTER_PATH = './generation_methods/IP_Adapter/'

# %%
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, choices=['IPAdapter', 'IPAdapterPlus'], default='IPAdapter')
parser.add_argument('--source_dir', type=str, required=True)
parser.add_argument('--destination_dir', type=str, required=True)
parser.add_argument('--num_inference_steps', type=int, default=50)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--prompt', type=str, default='High quality, best quality')

args = parser.parse_args()

# %%
if not os.path.exists(args.destination_dir):
    os.makedirs(args.destination_dir, exist_ok = True)

files = [file for file in os.listdir(args.source_dir) if file.lower().endswith('.png')]

# %%
device = 'cuda'

if args.model == 'IPAdapter':
    base_model_path = "stable-diffusion-v1-5/stable-diffusion-v1-5" # "runwayml/stable-diffusion-v1-5"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path = os.path.join(IP_ADAPTER_PATH, "models/image_encoder")
    ip_ckpt = os.path.join(IP_ADAPTER_PATH, "models/ip-adapter_sd15.bin")

elif args.model == 'IPAdapterPlus':
    base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path = os.path.join(IP_ADAPTER_PATH, "models/image_encoder")
    ip_ckpt = os.path.join(IP_ADAPTER_PATH, "models/ip-adapter-plus_sd15.bin")

else:
    raise ValueError(f'Invalid model: {args.model}.')

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

# load SD pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)
pipe.set_progress_bar_config(disable=True)

if args.model == 'IPAdapter':
    ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
elif args.model == 'IPAdapterPlus':
    ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)
else:
    raise ValueError(f'Invalid model: {args.model}.')

# %%
for file in tqdm(files):
    # Load image
    ip_image = Image.open(os.path.join(args.source_dir, file)).convert('RGB')
    
    # Generate
    generated_image = ip_model.generate(pil_image=ip_image, prompt=args.prompt, num_samples=1, num_inference_steps=args.num_inference_steps, seed=args.seed)[0]
    
    # Save image
    save_path = os.path.join(args.destination_dir, file)
    generated_image.save(save_path)

# %%



