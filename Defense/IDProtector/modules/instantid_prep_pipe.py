import importlib.util
import cv2
import torch
import numpy as np
from PIL import Image
import kornia
import torchvision

# Imports used during analysis
import skimage
# from insightface.app.common import Face

from diffusers.utils import load_image
from diffusers.models import ControlNetModel

# from insightface.app import FaceAnalysis
# from .pipeline_stable_diffusion_xl_instantid_relative import StableDiffusionXLInstantIDPipeline, draw_kps

# CLIP general imports
from diffusers.utils import load_image
from transformers import CLIPImageProcessor
from transformers.utils import constants

# General utils
import sys
from contextlib import contextmanager
import io

def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

def resize_img_tensor(input_tensor, max_side=1280, min_side=1024, size=None, pad_to_max_side=False, base_pixel_number=64):
    _, _, h, w = input_tensor.shape

    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w_resize, h_resize = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h_resize, w_resize)
        w_resize_new, h_resize_new = round(ratio * w_resize), round(ratio * h_resize)

    w_resize_new = (w_resize_new // base_pixel_number) * base_pixel_number
    h_resize_new = (h_resize_new // base_pixel_number) * base_pixel_number

    input_tensor = torch.nn.functional.interpolate(input_tensor.float(), size=(h_resize_new, w_resize_new), mode='bilinear', align_corners=False)

    if pad_to_max_side:
        pad_h = max_side - h_resize_new
        pad_w = max_side - w_resize_new
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        input_tensor = F.pad(input_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=1.0)

    return input_tensor

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

def estimate_norm(lmk, image_size=112,mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    tform = skimage.transform.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

def norm_crop_torch(img: torch.Tensor, landmark, device, image_size=112, mode='arcface'):
    _, _, h, w = img.shape
    if h != w:
        raise NotImplementedError(f'Implementation not validated for non-square images; got image with shape {img.shape}.')
    M = np.append(estimate_norm(landmark, image_size, mode), np.array([[0, 0, 1.]]), axis = 0)
    D = np.array([[2/h, 0, -1], [0, 2/w, -1], [0, 0, 1]])
    Theta = np.linalg.inv(D @ M @ np.linalg.inv(D))
    grid = torch.nn.functional.affine_grid(torch.tensor(Theta[:2, :]).unsqueeze(0).float(), torch.Size([1, 3, h, w]))
    return torch.nn.functional.grid_sample(img, grid.to(device))[:, :, :image_size, :image_size]

'''
[Log] 20240724
Update on `get_arcface_model`
'''

import onnx
from onnx2torch import convert

def get_arcface_model(path = './generation_methods/InstantID/models/antelopev2/glintr100.onnx'):
    model = onnx.load(path)
    torch_model = convert(model)
    net = torch_model
    return net.eval()


def get_arcface_model_old(return_app = False):

    raise DeprecationWarning(f'The function `get_arcface_model_old` is deprecated.')

    path_to_module = './generation_methods/insightface'
    module_name = 'generation_methods.insightface.app'
    spec = importlib.util.spec_from_file_location(module_name, f"{path_to_module}/face_analysis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # from insightface.app import FaceAnalysis
    FaceAnalysis = module.FaceAnalysis

    #1. ONNX 模型推理：
    # image = load_image("/Users/yiren/projects/stable_signature2/stable_signature/hidden/imgs/face1000-align/000112.jpg")
    # download 'antelopev2' under ./models
    app = FaceAnalysis(name='antelopev2', root='./generation_methods/InstantID', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))


    # _ = app.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))[-1]

    if return_app:
        return app

    #2. ONNX 转torch 模型
    ArcFaceONNX = app.get2()
    model = ArcFaceONNX # 通过实例调用

    return model

def get_insightface_app(path = './generation_methods/InstantID'):
    from insightface.app import FaceAnalysis
    insightface_app = FaceAnalysis(name='antelopev2', root=path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    insightface_app.prepare(ctx_id=0, det_size=(640, 640))
    return insightface_app


def arcface_align_pipeline(image_tensor: torch.Tensor, image_pil: Image, device):

    image_tensor = resize_img_tensor(image_tensor.float())

    image_bgr = cv2.cvtColor(np.array(resize_img(image_pil)), cv2.COLOR_RGB2BGR)
    app = get_arcface_model(return_app = True)
    bboxes, kpss = app.det_model.detect(image_bgr, max_num = 0, metric = 'default')
    image_warped_tensor = norm_crop_torch(image_tensor, landmark=kpss[0], image_size=app.models['recognition'].input_size[0], device=device)

    image_warped_tensor = torch.clamp(image_warped_tensor, 0, 255)
    # return Image.fromarray(image_warped_tensor[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
    image_warped_tensor = image_warped_tensor / 127.5 - 1
    image_warped_tensor_bgr = image_warped_tensor[:, [2, 1, 0]] # RGB -> BGR conversion

    return image_warped_tensor_bgr

def get_affine_grid(img: torch.Tensor, landmark, image_size=112, mode='arcface'):
    _, _, h, w = img.shape
    if h != w:
        raise NotImplementedError(f'Implementation not validated for non-square images; got image with shape {img.shape}.')
    M = np.append(estimate_norm(landmark, image_size, mode), np.array([[0, 0, 1.]]), axis = 0)
    D = np.array([[2/h, 0, -1], [0, 2/w, -1], [0, 0, 1]])
    Theta = np.linalg.inv(D @ M @ np.linalg.inv(D))
    grid = torch.nn.functional.affine_grid(torch.tensor(Theta[:2, :]).unsqueeze(0).float(), torch.Size([1, 3, h, w]))
    return grid

# def get_affine_params(image_tensor: torch.Tensor, image_pil: Image):

#     '''Deprecated implementation (using torch.nn.functional); used in PGD_u_20240502.1-7'''

#     image_tensor = resize_img_tensor(image_tensor.float())
#     image_bgr = cv2.cvtColor(np.array(resize_img(image_pil)), cv2.COLOR_RGB2BGR)
#     app = get_arcface_model(return_app = True)

#     bboxes, kpss = app.det_model.detect(image_bgr, max_num = 0, metric = 'default')
#     bboxes = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
#     kpss = kpss[np.argsort(np.array(bboxes))]

#     image_size = app.models['recognition'].input_size[0]
#     return get_affine_grid(img = image_tensor, landmark = kpss[-1], image_size = image_size), image_size

# def arcface_align_pipeline_simplified(image_tensor: torch.Tensor, affine_grid: torch.Tensor, affine_image_size: int):

#     '''Deprecated implementation (using torch.nn.functional); used in PGD_u_20240502.1-7'''

#     image_tensor = resize_img_tensor(image_tensor.float())
#     image_warped_tensor = torch.nn.functional.grid_sample(image_tensor, affine_grid.to(image_tensor.device))[:, :, :affine_image_size, :affine_image_size]

#     image_warped_tensor = torch.clamp(image_warped_tensor, 0, 255)
#     image_warped_tensor = image_warped_tensor / 127.5 - 1
#     image_warped_tensor_bgr = image_warped_tensor[:, [2, 1, 0]] # RGB -> BGR conversion

#     return image_warped_tensor_bgr

def get_affine_params(image_tensor: torch.Tensor, image_pil:Image):

    raise DeprecationWarning(f'The function `get_affine_params` is deprecated.')

    image_tensor = resize_img_tensor(image_tensor.float())
    image_bgr = cv2.cvtColor(np.array(resize_img(image_pil)), cv2.COLOR_RGB2BGR)
    app = get_arcface_model_old(return_app = True)

    bboxes, kpss = app.det_model.detect(image_bgr, max_num = 0, metric = 'default')
    bboxes = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
    kpss = kpss[np.argsort(np.array(bboxes))]

    image_size = app.models['recognition'].input_size[0]

    M = torch.tensor(estimate_norm(kpss[-1], image_size, 'arcface')).repeat(image_tensor.shape[0], 1, 1).float().to(image_tensor.device)

    return M, image_size

def get_affine_params_with_app(image_tensor: torch.Tensor, image_pil:Image, app):

    image_tensor = resize_img_tensor(image_tensor.float())
    image_bgr = cv2.cvtColor(np.array(resize_img(image_pil)), cv2.COLOR_RGB2BGR)
    # app = get_arcface_model_old(return_app = True)

    bboxes, kpss = app.det_model.detect(image_bgr, max_num = 0, metric = 'default')
    if len(bboxes) == 0:
        return None, None
    bboxes = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
    kpss = kpss[np.argsort(np.array(bboxes))]

    image_size = app.models['recognition'].input_size[0]

    M = torch.tensor(estimate_norm(kpss[-1], image_size, 'arcface')).repeat(image_tensor.shape[0], 1, 1).float().to(image_tensor.device)

    return M, image_size

def arcface_align_pipeline_simplified(image_tensor: torch.Tensor, M: torch.Tensor, affine_image_size: int):

    image_tensor = resize_img_tensor(image_tensor.float())
    image_warped_tensor = kornia.geometry.transform.warp_affine(image_tensor, M, tuple(image_tensor.shape[-2:]))[:, :, :affine_image_size, :affine_image_size]

    image_warped_tensor = torch.clamp(image_warped_tensor, 0, 255)
    image_warped_tensor = image_warped_tensor / 127.5 - 1

    return image_warped_tensor

    # image_warped_tensor_bgr = image_warped_tensor[:, [2, 1, 0]] # RGB -> BGR conversion
    return image_warped_tensor_bgr

def normalize(img_tensor):
    # Normalize the image tensor using the pre-defined mean and standard deviation for each channel (yiren's implementation)
    mean_tensor = torch.tensor([127.5, 127.5, 127.5]).view(1, 3, 1, 1)
    std_tensor = torch.tensor([127.5, 127.5, 127.5]).view(1, 3, 1, 1)
    img_tensor = img_tensor[:, [2, 1, 0], :, :]
    return (img_tensor - mean_tensor) / std_tensor

class CLIPImageDenormaliser:
    def __init__(self, mean = constants.OPENAI_CLIP_MEAN, std = constants.OPENAI_CLIP_STD):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.denormalise(tensor)

    def denormalise_scale_tensor(self, tensor):
        return tensor * torch.tensor(self.std)[None, :, None, None].to(tensor.device)

    def normalise_scale_tensor(self, tensor):
        return tensor / torch.tensor(self.std)[None, :, None, None].to(tensor.device)
    
    def denormalise_tensor(self, tensor):
        return tensor * torch.tensor(self.std)[None, :, None, None].to(tensor.device) + torch.tensor(self.mean)[None, :, None, None].to(tensor.device)
    
    def normalise_tensor(self, tensor):
        return (tensor - torch.tensor(self.mean)[None, :, None, None].to(tensor.device)) / torch.tensor(self.std)[None, :, None, None].to(tensor.device)
    
    def denormalise(self, tensor):
        denormalised_tensor = tensor.detach().cpu().numpy() * np.array(self.std)[None, :, None, None] + np.array(self.mean)[None, :, None, None]
        denormalised_images = np.clip(denormalised_tensor * 255, 0, 255).astype(np.uint8).transpose((0, 2, 3, 1))
        return [Image.fromarray(this_image) for this_image in denormalised_images]

def clip_preprocessing_pipeline(image_tensor: torch.Tensor, denormaliser: CLIPImageDenormaliser):
    image_tensor = torch.nn.functional.interpolate(image_tensor.float(), size=(224, 224), mode='bilinear', align_corners=False)
    image_tensor = image_tensor / 255
    return denormaliser.normalise_tensor(image_tensor)

'''
[Log] 20240505
Update on `clip_preprocessing_pipeline_transforms`
 - Second version of `clip_preprocessing_pipeline`
 - This update is with `20240505.20.ipynb`
 - Implementing bicubic resampling better adapts the adversarial perturbations to the CLIP preprocessing pipeline (given the images are square)
 - Please consider citing our paper if the insight of adaptation inspires you :)
'''

clip_resize = torchvision.transforms.Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)

def clip_preprocessing_pipeline_transforms(image_tensor: torch.Tensor, denormaliser: CLIPImageDenormaliser, transforms: torch.nn.Module = clip_resize):
    image_tensor = image_tensor / 255
    image_tensor = transforms(image_tensor)
    return denormaliser.normalise_tensor(image_tensor)

'''
[Log] 20240527
Update on `get_resizing_size` and `clip_preprocessing_pipeline_v3`: enables centre cropping for non-squaure inputs (previous version `clip_preprocessing_pipeline_transforms` resizes non-square inputs to 224*224)
'''

def get_resizing_size(size):
    '''Get CLIP resizing size: resize the shorter edge to 224*224 and longer edge to keep the aspect ratio'''
    height, weight = size
    smaller, larger = sorted((height, weight))
    new_smaller = 224
    new_larger = int(224 * larger / smaller)
    updated_values = (new_smaller, new_larger) if height == smaller else (new_larger, new_smaller)
    return updated_values

def clip_preprocessing_pipeline_v3(image_tensor: torch.Tensor, denormaliser: CLIPImageDenormaliser):
    image_tensor = image_tensor / 255

    # Get the original height and width
    current_size = image_tensor.shape[-2], image_tensor.shape[-1]

    # Resize the shorter edge to 224 and maintain aspect ratio for the longer edge
    image_tensor = torchvision.transforms.Resize(get_resizing_size(current_size), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)(image_tensor)

    # Perform center crop
    crop_height, crop_width = 224, 224

    # Calculate the top, bottom, left, and right coordinates for the center crop
    top = (image_tensor.shape[-2] - crop_height) // 2
    bottom = top + crop_height
    left = (image_tensor.shape[-1] - crop_width) // 2
    right = left + crop_width

    image_tensor = image_tensor[:, :, top:bottom, left:right]

    if denormaliser is None:
        return image_tensor
    return denormaliser.normalise_tensor(image_tensor)

@contextmanager
def suppress_stdout():
    """A context manager to suppress stdout."""
    # Create a new stdout object that writes to devnull
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    try:
        yield
    finally:
        sys.stdout = sys.__stdout__

@contextmanager
def suppress_output():
    """A context manager to suppress both stdout and stderr."""
    # Save the original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create new stdout and stderr objects that write to devnull
    new_stdout = io.StringIO()
    new_stderr = io.StringIO()
    
    sys.stdout = new_stdout
    sys.stderr = new_stderr
    try:
        yield
    finally:
        # Restore the original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr