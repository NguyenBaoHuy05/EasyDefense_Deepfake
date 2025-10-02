# %% [markdown]
#  - Environment: `IPAttack`
#  - Run this notebook to get attack result metrics, saved as a csv file under `training_results/date`

# %%

import argparse

parser = argparse.ArgumentParser()

# Paths
parser.add_argument('--clean_data_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--path_to_state_dict', type=str, required=True)
parser.add_argument('--metrics_save_path', type=str, required=True)

# Network
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--in_channels', type=int, required=True)

# Loss/metric
parser.add_argument('--epsilon', type=float, required=False, default=0.1)
parser.add_argument('--clip_epsilon', type=int, required=True)
parser.add_argument('--calculate_vgg_loss', type=int, default=1)
parser.add_argument('--calculate_lpips_loss', type=int, default=1)

# Runs
parser.add_argument('--cuda', type=int, required=True)

args = parser.parse_args()

# %%

clean_data_dir = args.clean_data_dir
save_dir = args.save_dir

path_to_state_dict = args.path_to_state_dict
metrics_save_path = args.metrics_save_path

batch_size = args.batch_size
arcface_affine_size = 112
network_input_size = 224
in_channels = args.in_channels

emb_magnitude_factor = 0.
epsilon = args.epsilon
clip_epsilon = args.clip_epsilon
calculate_vgg_loss = args.calculate_vgg_loss
calculate_lpips_loss = args.calculate_lpips_loss

# %%
import torch
torch.cuda.set_device(args.cuda)
torch.set_grad_enabled(False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

# %%
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import kornia
import pyiqa

import sys
sys.path.append('./utils/PerceptualSimilarity/src/loss')
from utils.PerceptualSimilarity.src.loss.loss_provider import LossProvider

# %%
import numpy as np
from diffusers.utils import load_image
from typing import Callable
from utils.MetricLogger import MetricLogger
import time
from tqdm import tqdm

from transformers import CLIPImageProcessor
from modules.instantid_prep_pipe import arcface_align_pipeline_simplified, get_arcface_model, get_insightface_app, clip_preprocessing_pipeline_v3, CLIPImageDenormaliser, suppress_output
from modules.model_instances import get_IPAdapter_models, get_IPAdapter_embeddings, CLIP_vision_PhotoMaker

from modelling.NoContextDiT import NoContextDiT
from utils.ImageOnlyWithPriorsDataset import ImageOnlyWithPriorsDataset

# %%
logger = MetricLogger(metrics_save_path)

torch.set_grad_enabled(False)

# %% [markdown]
# # Dataset and Dataloader

# %%
with suppress_output():
    arcface_app = get_insightface_app()

# %%
dataset = ImageOnlyWithPriorsDataset(clean_data_dir, network_input_size, arcface_app)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)

# %% [markdown]
# # Adv model

# %%
ddp_state_dict = torch.load(path_to_state_dict)
non_ddp_state_dict = {}
for key in ddp_state_dict.keys():
    new_key = key.replace('module.', '')
    non_ddp_state_dict[new_key] = ddp_state_dict[key]

adv_model = NoContextDiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, input_size=network_input_size, learn_sigma=False, in_channels=in_channels, out_channels=3)
adv_model.load_state_dict(non_ddp_state_dict)
adv_model = adv_model.to(device).eval()

# %% [markdown]
# # Load face models

# %%
with suppress_output():
    clip_image_processor = CLIPImageProcessor()
    clip_image_denormaliser = CLIPImageDenormaliser()
    clip_vision_model_PhotoMaker = CLIP_vision_PhotoMaker().to(device)
    IPAdapter_models = get_IPAdapter_models(device = device)
    arcface_model = get_arcface_model().to(device)

if calculate_lpips_loss:
    lpips_model = pyiqa.create_metric('lpips', device = device, as_loss = True)
if calculate_vgg_loss:
    vgg_loss_model = LossProvider().get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum').to(device)

# %%
def face_prep_pipe(
    image_tensor: torch.Tensor, 
    arcface_affine_M: torch.Tensor, 
    arcface_affine_size: int, 

    input_augmentation: Callable = None, 
    arcface_augmentation: Callable = None,
    clip_augmentation: Callable = None
):

    if input_augmentation is not None:
        image_tensor = input_augmentation(image_tensor)
    
    with suppress_output():

        arcface_image = arcface_align_pipeline_simplified(image_tensor, arcface_affine_M, arcface_affine_size)
        if arcface_augmentation is not None:
            arcface_image = arcface_augmentation(arcface_image)

        clip_image = clip_preprocessing_pipeline_v3(image_tensor, clip_image_denormaliser).to(dtype = torch.float16)
        if clip_augmentation is not None:
            clip_image = clip_augmentation(clip_image)

    return arcface_image, clip_image

# %% [markdown]
# # Inference Loop

# %%
def compute_embeds(arcface_image: torch.Tensor, clip_image: torch.Tensor):
    pred_IPAdapter_series = get_IPAdapter_embeddings(clip_image, IPAdapter_models, device)
    pred_IPAdapter = pred_IPAdapter_series.IPAdapter
    pred_IPAdapterPlus = pred_IPAdapter_series.IPAdapterPlus
    pred_PhotoMaker = clip_vision_model_PhotoMaker(clip_image).pooler_output
    pred_InstantID = arcface_model(arcface_image)
    return pred_IPAdapter, pred_IPAdapterPlus, pred_PhotoMaker, pred_InstantID

# %%
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# %%
for images_clean, dataset_indices in tqdm(dataloader):
    images_clean = images_clean.to(device)
    
    # Step 1 - Compute adversarial perturbations
    '''
    Input  ---adv_model--->  Output

        - Input: CelebA in-the-wild images, resized to [b, 3, 224, 224] normalised to [-1, 1] + additional [b, 1, 224, 224] one-hot prior
        - Output: [b, 3, 224, 224] adversarial perturbations
    '''
    start_time = time.time()
    adv_perturbations = adv_model(images_clean)
    end_time = time.time()
    l_infty_loss = (torch.nn.ReLU()(-1 - adv_perturbations) + torch.nn.ReLU()(adv_perturbations - 1)).mean()
    l_2_loss = adv_perturbations.pow(2).mean()

    if clip_epsilon:
        adv_perturbations = torch.clamp(adv_perturbations, -1, 1) * 127.5 * epsilon
    else:
        adv_perturbations = adv_perturbations * 127.5 * epsilon
    # images_clean = images_clean[:, :(3 - in_channels)]

    # TODO: Implement OOB loss
    

    # Step 2 - Perturb and align images at arcface/clip inputs
    arcface_image_clean_list, clip_image_clean_list, arcface_image_adv_list, clip_image_adv_list = [], [], [], []

    for batch_index in range(images_clean.shape[0]):
        dataset_index = dataset_indices[batch_index].item()
        
        # Load the original high-res image, also load the corresponding M matrix
        image_clean = dataset.get_original_image(dataset_index).convert('RGB')
        original_size = (image_clean.size[1], image_clean.size[0])
        arcface_affine_M_clean = dataset.get_affine_M(image_clean).to(device)
        image_clean = torch.tensor(np.array(image_clean)).permute(2, 0, 1)[None, :].float().to(device)

        # Upscale the adversarial perturbation to the original iamge size, and then perturb onto the image
        adv_perturbation = adv_perturbations[batch_index][None, ...]
        adv_perturbation = kornia.geometry.transform.resize(adv_perturbation, original_size, interpolation = 'bicubic', antialias = True)
        adv_perturbation = torch.clamp(adv_perturbation, -127.5 * epsilon, 127.5 * epsilon) # Clamp back to epsilon ball again after interpolation (which could double the epsilon ball)
        image_adv = torch.clamp(image_clean + adv_perturbation, 0, 255)     # [b, 3, 224, 224]

        # Save image and quantise
        image_adv_pil = Image.fromarray(np.clip(image_adv[0].permute(1, 2, 0).detach().cpu().numpy(), 0, 255).astype(np.uint8))
        # save_name = os.path.join(save_dir, dataset.get_original_image_filename(dataset_index).split('/')[-1])    
        save_name = os.path.join(save_dir, os.path.splitext(os.path.basename(dataset.get_original_image_filename(dataset_index)))[0])
        image_adv_pil.save(save_name + '.png')
        torch.save(adv_perturbation.detach().cpu().numpy(), save_name + '.pt')
        image_adv_quantised = torch.tensor(np.array(image_adv_pil)).permute(2, 0, 1)[None, ...].float().to(image_adv.device).to(image_adv.dtype)
        arcface_affine_M_adv = dataset.get_affine_M(image_adv_pil)


        # Log l2 and l_infty 
        logger.collect('filename', dataset.get_original_image_filename(dataset_index).split('/')[-1])
        logger.collect('l_infty', torch.max(torch.abs(image_adv_quantised - image_clean)).item())
        logger.collect('l_2', (image_adv_quantised - image_clean).pow(2).mean().item())
        logger.collect('time', end_time - start_time)

        if arcface_affine_M_clean is None:
            raise RuntimeError(f'Insightface face detection failed for image {dataset.get_original_image_filename(dataset_index)}. ')
        if arcface_affine_M_adv is None:
            logger.collect('arcface_adv_detected', 0)
            arcface_affine_M_adv = arcface_affine_M_clean
        else:
            arcface_affine_M_adv = arcface_affine_M_adv.to(device)
            logger.collect('arcface_adv_detected', 1)


        # Compute and log vgg and lpips
        if calculate_lpips_loss:
            lpips_loss = lpips_model(image_clean / 255, image_adv_quantised / 255).mean().item()
            logger.collect('lpips', lpips_loss)
        if calculate_vgg_loss:
            vgg_loss = vgg_loss_model(image_clean / 255, image_adv_quantised / 255).mean().item()
            logger.collect('vgg', vgg_loss)


        # Align the image to arcface/clip inputs
        arcface_image_clean, clip_image_clean = face_prep_pipe(
            image_clean,
            arcface_affine_M_clean, # + torch.randn_like(arcface_affine_M) * 1e-3,
            arcface_affine_size
        )
        arcface_image_adv, clip_image_adv = face_prep_pipe(
            image_adv_quantised,
            arcface_affine_M_adv, # + torch.randn_like(arcface_affine_M) * 1e-3,
            arcface_affine_size
        )

        arcface_image_clean_list.append(arcface_image_clean)
        clip_image_clean_list.append(clip_image_clean)
        arcface_image_adv_list.append(arcface_image_adv)
        clip_image_adv_list.append(clip_image_adv)

    arcface_images_clean = torch.cat(arcface_image_clean_list, dim = 0)
    clip_images_clean = torch.cat(clip_image_clean_list, dim = 0)
    arcface_images_adv = torch.cat(arcface_image_adv_list, dim = 0)
    clip_images_adv = torch.cat(clip_image_adv_list, dim = 0)



    # Step 3 - Compute embeddings
    pred_clean_IPAdapter, pred_clean_IPAdapterPlus, pred_clean_PhotoMaker, pred_clean_InstantID = compute_embeds(arcface_images_clean, clip_images_clean)
    pred_adv_IPAdapter, pred_adv_IPAdapterPlus, pred_adv_PhotoMaker, pred_adv_InstantID = compute_embeds(arcface_images_adv, clip_images_adv)

    pred_clean_IPAdapterPlus = pred_clean_IPAdapterPlus.flatten(start_dim = -2)
    pred_adv_IPAdapterPlus = pred_adv_IPAdapterPlus.flatten(start_dim = -2)



    # Step 4 - Compute metrics

    for value in torch.nn.functional.mse_loss(pred_clean_IPAdapter * emb_magnitude_factor, pred_adv_IPAdapter, reduction = 'none').mean(dim = 1).tolist():
        logger.collect('IPAdapter_mse', value)
    for value in torch.nn.functional.mse_loss(pred_clean_IPAdapterPlus * emb_magnitude_factor, pred_adv_IPAdapterPlus, reduction = 'none').mean(dim = 1).tolist():
        logger.collect('IPAdapterPlus_mse', value)
    for value in torch.nn.functional.cosine_similarity(pred_clean_PhotoMaker, pred_adv_PhotoMaker, dim = 1).tolist():
        logger.collect('PhotoMaker_cossim', value)
    for value in torch.nn.functional.cosine_similarity(pred_clean_InstantID, pred_adv_InstantID, dim = 1).tolist():
        logger.collect('InstantID_cossim', value)


    loss_IPAdapter = torch.nn.functional.mse_loss(pred_clean_IPAdapter * emb_magnitude_factor, pred_adv_IPAdapter) # minimise
    loss_IPAdapterPlus = torch.nn.functional.mse_loss(pred_clean_IPAdapterPlus * emb_magnitude_factor, pred_adv_IPAdapterPlus) # minimise
    loss_PhotoMaker = torch.nn.functional.cosine_similarity(pred_clean_PhotoMaker, pred_adv_PhotoMaker, dim = 1) # minimise
    loss_InstantID = torch.nn.functional.cosine_similarity(pred_clean_InstantID, pred_adv_InstantID, dim = 1) # minimise


# %%
logger.save()

# %%



