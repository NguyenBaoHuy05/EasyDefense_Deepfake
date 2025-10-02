# %%
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from utils.MetricLogger import MetricLogger
import argparse
import os

from modules.instantid_prep_pipe import get_insightface_app, suppress_output
from utils.eval_insightface_metrics import get_insightface_metrics
from insightface.app import FaceAnalysis
from utils.calculate_ser_fiq import calculate_ser_fiq
from utils.FaceImageQuality.face_image_quality import SER_FIQ
from brisque import BRISQUE
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import pyiqa

# %%
parser = argparse.ArgumentParser()

parser.add_argument('--generated_image_dir', type=str, required=True, help="Directory containing generated images")
parser.add_argument('--ref_image_dir', type=str, required=True, help="Directory containing reference images")
parser.add_argument('--adv_image_dir', type=str, required=True, help="Directory containing adversarial examples")
parser.add_argument('--metric_savepath', type=str, default=None, help="Path to save the csv containing metric results")
parser.add_argument('--clip_iqa_model_name', type=str, default='clipiqa', choices=['clipiqa', 'clipiqa+', 'clipiqa+_rn50_512', 'clipiqa+_vitL14_512'])
parser.add_argument('--pyiqa_device', type=str, default='cuda', help='Load pyiqa models (e.g. CLIP-IQA) with cuda or CPU')

args = parser.parse_args()

if args.metric_savepath is None:
    args.metric_savepath = os.path.join(args.generated_image_dir, 'metric.csv')

# %% [markdown]
# # Utils

# %%
def cossim(array1: np.ndarray, array2: np.ndarray):
    return np.dot(array1, array2) / (np.linalg.norm(array1) * np.linalg.norm(array2))

def pad_and_resize(image: Image.Image):
    '''
    Function to pad a PIL image. This is in case insightface does not recognise big faces.
    '''
    original_size = image.size
    new_size = (original_size[0] * 2, original_size[1] * 2)
    padded_image = Image.new("RGB", new_size, (128, 128, 128))
    padded_image.paste(image, (original_size[0] // 2, original_size[1] // 2))
    resized_image = padded_image.resize(original_size)
    return resized_image

# %% [markdown]
# # Load models

# %%
with suppress_output():
    insightface_app = get_insightface_app()
    ser_fiq = SER_FIQ(gpu=0)
    brisque = BRISQUE(url=False)
clip_iqa_model = pyiqa.create_metric(args.clip_iqa_model_name, device = args.pyiqa_device) 

# %% [markdown]
# # Prepare files and loggers

# %%
generated_image_files = [file for file in os.listdir(args.generated_image_dir) if file.lower().endswith(('jpg', 'png', 'jpeg'))]
ref_image_files = [file for file in os.listdir(args.ref_image_dir) if file.lower().endswith(('jpg', 'png', 'jpeg'))]
adv_image_files = [file for file in os.listdir(args.adv_image_dir) if file.lower().endswith(('jpg', 'png', 'jpeg'))]
files = list(set(generated_image_files) & set(ref_image_files) & set(adv_image_files))

logger = MetricLogger(args.metric_savepath)

# %% [markdown]
# # Eval and save

# %%
for file in tqdm(files):
    logger.collect('filename', file)

    generated_image_path = os.path.join(args.generated_image_dir, file)
    ref_image_path = os.path.join(args.ref_image_dir, file)
    adv_image_path = os.path.join(args.adv_image_dir, file)

    generated_image = Image.open(generated_image_path).convert('RGB')
    ref_image = Image.open(ref_image_path).convert('RGB')
    adv_image = Image.open(adv_image_path).convert('RGB')

    # Evaluate FDFR (gen, ref) and Arcface cossim (gen <--> ref)
    detection_success, generated_arcface_emb = get_insightface_metrics(generated_image, insightface_app)
    if not detection_success:
        detection_success, generated_arcface_emb = get_insightface_metrics(pad_and_resize(generated_image), insightface_app)
    ref_detection_success, ref_arcface_emb = get_insightface_metrics(ref_image, insightface_app)
    if not ref_detection_success:
        ref_detection_success, ref_arcface_emb = get_insightface_metrics(pad_and_resize(ref_image), insightface_app)

    logger.collect('gen_detection_success', int(detection_success))
    logger.collect('ref_detection_success', int(ref_detection_success))
    if detection_success and ref_detection_success:
        if generated_arcface_emb is not None and ref_arcface_emb is not None:
            logger.collect('cossim', cossim(generated_arcface_emb, ref_arcface_emb))
        else:
            raise RuntimeError(f'Possible that `get_insightface_metrics` successfully detected a face but failed to provide a proper face embedding.')
    else:
        logger.collect('cossim', 'NA')
    

    # Evaluate SER-FIQ (gen, ref, adv)
    generated_image_bgr = cv2.cvtColor(np.array(generated_image), cv2.COLOR_RGB2BGR)
    ref_image_bgr = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
    adv_image_bgr = cv2.cvtColor(np.array(adv_image), cv2.COLOR_RGB2BGR)
    logger.collect('ser_fiq_gen', calculate_ser_fiq(generated_image_bgr, ser_fiq))
    logger.collect('ser_fiq_ref', calculate_ser_fiq(ref_image_bgr, ser_fiq))
    logger.collect('ser_fiq_adv', calculate_ser_fiq(adv_image_bgr, ser_fiq))

    # Evaluate BRISQUE (gen, ref, adv)
    logger.collect('brisque_gen', brisque.score(np.array(generated_image)))
    logger.collect('brisque_ref', brisque.score(np.array(ref_image)))
    logger.collect('brisque_adv', brisque.score(np.array(adv_image)))

    # Evaluate PSNR (ref <--> adv)
    psnr = peak_signal_noise_ratio(np.array(ref_image), np.array(adv_image))
    logger.collect('psnr', psnr)

    # Evaluate SSIM (ref <--> adv)
    ssim = structural_similarity(np.array(ref_image), np.array(adv_image), channel_axis = 2)
    logger.collect('ssim', ssim)

    # Evaluate CLIP_IQA (gen, ref, adv)
    logger.collect('clip_iqa_gen', clip_iqa_model(generated_image_path).item())
    logger.collect('clip_iqa_ref', clip_iqa_model(ref_image_path).item())
    logger.collect('clip_iqa_adv', clip_iqa_model(adv_image_path).item())

    # Evaluate l_2 and l_infty (ref <--> adv)
    logger.collect('l_infty', np.max(np.abs(np.array(ref_image).astype(float) - np.array(adv_image).astype(float))))
    logger.collect('l_2', np.mean((np.array(ref_image).astype(float) - np.array(adv_image).astype(float)) ** 2))

logger.save()

# %%



