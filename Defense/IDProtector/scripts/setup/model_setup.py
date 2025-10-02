import torch
from diffusers import DiffusionPipeline
_ = DiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", torch_dtype=torch.float16)
_ = DiffusionPipeline.from_pretrained("SG161222/RealVisXL_V3.0")
_ = DiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16)
_ = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
_ = DiffusionPipeline.from_pretrained("TencentARC/PhotoMaker", torch_dtype=torch.bfloat16, use_safetensors=True, variant="fp16")

from diffusers.models import AutoencoderKL
_ = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

from huggingface_hub import hf_hub_download
photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")

from generation_methods.InstantID.pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
_ = StableDiffusionXLInstantIDPipeline.from_pretrained('wangqixun/YamerMIX_v8')