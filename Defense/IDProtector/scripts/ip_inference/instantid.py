# %%
import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

import os
import sys
INSTANTID_PATH = './generation_methods/InstantID'
sys.path.append(INSTANTID_PATH)

from insightface.app import FaceAnalysis
from generation_methods.InstantID.pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

# %%
parser = argparse.ArgumentParser()

parser.add_argument('--source_dir', type=str, required=True)
parser.add_argument('--destination_dir', type=str, required=True)
parser.add_argument('--num_inference_steps', type=int, default=30)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--height', type=int, default=1024)
parser.add_argument('--width', type=int, default=1024)
parser.add_argument('--prompt', type=str, default='A person, realistic')

args = parser.parse_args()

# %%
if not os.path.exists(args.destination_dir):
    os.makedirs(args.destination_dir, exist_ok = True)

files = [file for file in os.listdir(args.source_dir) if file.lower().endswith('.png')]

# %%
# prepare 'antelopev2' under ./models
app = FaceAnalysis(name='antelopev2', root=INSTANTID_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# prepare models under ./checkpoints
face_adapter = os.path.join(INSTANTID_PATH, f'checkpoints/ip-adapter.bin')
controlnet_path = os.path.join(INSTANTID_PATH, f'checkpoints/ControlNetModel')

# load IdentityNet
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

base_model = 'stabilityai/stable-diffusion-xl-base-1.0'  # 'wangqixun/YamerMIX_v8'  # from https://civitai.com/models/84040?modelVersionId=196039
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe.set_progress_bar_config(disable=True)
pipe.cuda()

# load adapter
pipe.load_ip_adapter_instantid(face_adapter)

torch.autograd.set_grad_enabled(False)

# %%
# prompt
# prompt = "film noir style, ink sketch|vector, male man, highly detailed, sharp focus, ultra sharpness, monochrome, high contrast, dramatic shadows, 1940s style, mysterious, cinematic"
# negative_prompt = "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, vibrant, colorful"

# prompt = "analog film photo of a man. faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, masterpiece, best quality"
# negative_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"

prompt = args.prompt
negative_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"

# %%
for file in tqdm(files):
    # Load image
    ip_image = Image.open(os.path.join(args.source_dir, file)).convert('RGB')
    
    # prepare face emb
    face_info = app.get(cv2.cvtColor(np.array(ip_image), cv2.COLOR_RGB2BGR))

    if len(face_info) == 0:
        continue

    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # only use the maximum face
    face_emb = face_info['embedding']
    face_kps = draw_kps(ip_image, face_info['kps'])

    # generate image
    generated_image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=0.8,
        ip_adapter_scale=0.8,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=5,
        height=args.height,
        width=args.width
    ).images[0]

    # Save image
    save_path = os.path.join(args.destination_dir, file)
    generated_image.save(save_path)

# %%



