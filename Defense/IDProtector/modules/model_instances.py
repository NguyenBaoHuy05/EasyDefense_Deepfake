import torch
import os

# IPAdapter
from transformers import CLIPVisionModelWithProjection
from generation_methods.IP_Adapter.ip_adapter import IPAdapterPlus
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL

# Photomaker
from huggingface_hub import hf_hub_download
from generation_methods.PhotoMaker.photomaker import PhotoMakerStableDiffusionXLPipeline

class AttributeContainer:
    pass

# Paths (TODO: move these into functions/yaml configs)
USER_HOME_DIR = os.path.expanduser('~')

PATH_IPADAPTER_VAE = "stabilityai/sd-vae-ft-mse"
PATH_IPADAPTER_BASE = os.path.join(USER_HOME_DIR, '.cache/huggingface/hub/models--SG161222--Realistic_Vision_V4.0_noVAE/snapshots/1bd8c538b40236e642a1427ed154a50ef5bdd3df')
PATH_IPADAPTER_IMAGE_ENCODER = "./generation_methods/IP_Adapter/models/image_encoder"
PATH_IPADAPTER_IP_CKPT ="./generation_methods/IP_Adapter/models/ip-adapter-plus_sd15.bin"
PATH_PHOTOMAKER_CKPT = os.path.join(USER_HOME_DIR, '.cache/huggingface/hub/models--TencentARC--PhotoMaker/snapshots/f68f8e6309bf213d28d68230abff0ccc92de9f30/photomaker-v1.bin')
PATH_PHOTOMAKER_BASE = os.path.join(USER_HOME_DIR, '.cache/huggingface/hub/models--SG161222--RealVisXL_V3.0/snapshots/4a3f0e44d3abcc0c3ee48fe85e337d78075d1445')


def get_IPAdapter_models(device = 'cuda'):

    # base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE" # Base model changed
    vae_model_path = PATH_IPADAPTER_VAE
    base_model_path = PATH_IPADAPTER_BASE
    image_encoder_path = PATH_IPADAPTER_IMAGE_ENCODER
    ip_ckpt = PATH_IPADAPTER_IP_CKPT

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    # vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(vae_model_path, local_files_only = True).to(dtype=torch.float16)

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )

    ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

    IPAdapter_models = AttributeContainer()
    IPAdapter_models.clip_image_processor = ip_model.clip_image_processor
    IPAdapter_models.clip_vision_model = ip_model.image_encoder.vision_model
    IPAdapter_models.clip_proj_layer = ip_model.image_encoder.visual_projection
    IPAdapter_models.resampler_proj_layer = ip_model.image_proj_model

    return IPAdapter_models


def get_IPAdapter_embeddings(image_tensor: torch.Tensor, IPAdapter_models: AttributeContainer, device = 'cuda'):

    clip_raw_output = IPAdapter_models.clip_vision_model(image_tensor.to(device, dtype=torch.float16), output_hidden_states = True)

    clip_pooler_output = clip_raw_output.pooler_output
    clip_project_output = IPAdapter_models.clip_proj_layer(clip_pooler_output)

    clip_hidden_output = clip_raw_output.hidden_states[-2]
    resampler_project_output = IPAdapter_models.resampler_proj_layer(clip_hidden_output)

    IPAdapter_embeddings = AttributeContainer()
    IPAdapter_embeddings.IPAdapter = clip_project_output
    IPAdapter_embeddings.IPAdapterPlus = resampler_project_output

    return IPAdapter_embeddings


def CLIP_vision_IPAdapter():
    '''
    Returns CLIP vision model in fp32
    ''' 
    return CLIPVisionModelWithProjection.from_pretrained(PATH_IPADAPTER_IMAGE_ENCODER).vision_model


def CLIP_vision_PhotoMaker(return_id_encoder = False):
    # photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
    photomaker_ckpt = PATH_PHOTOMAKER_CKPT

    PhotoMaker_pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
        PATH_PHOTOMAKER_BASE, # 'SG161222/RealVisXL_V3.0',
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="fp16",
    )

    PhotoMaker_pipe.load_photomaker_adapter(
        os.path.dirname(photomaker_ckpt),
        subfolder="",
        weight_name=os.path.basename(photomaker_ckpt),
        trigger_word="img"
    )

    if return_id_encoder:
        return PhotoMaker_pipe.id_encoder
    
    return PhotoMaker_pipe.id_encoder.vision_model