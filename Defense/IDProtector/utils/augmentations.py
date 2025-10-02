import torch
import kornia
import random
from io import BytesIO
from PIL import Image
import numpy as np
from typing import Union

class InterpolationErrorSimulator:
    def __init__(self, affine_randomness: int = None):
        self.affine_randomness = affine_randomness

    def __call__(self, x):
        # Step 1: Pad the image
        b, _, h, w = x.shape
        x_dtype = x.dtype
        x_padded = torch.nn.functional.pad(x, (w, w, h, h)).float()

        # Step 2: Get random affine params
        translations = torch.zeros(b, 2)
        center = torch.tensor([int(3*h/2), int(3*w/2)])[None, :].repeat(b, 1).float()
        scale = torch.tensor([random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)])[None, :].repeat(b, 1)
        angle = torch.tensor([random.uniform(-30, 30)]).repeat(b)
        sx = torch.tensor([random.uniform(-0.5, 0.5)]).repeat(b)
        sy = torch.tensor([random.uniform(-0.5, 0.5)]).repeat(b)

        # Step 3: Perform random affine transformation
        affine_matrix = kornia.geometry.get_affine_matrix2d(translations = translations, center = center, scale = scale, angle = angle, sx = sx, sy = sy).to(x.device)

        x_transformed = kornia.geometry.transform.warp_affine(x_padded, affine_matrix[:, :2, :], (3*h, 3*w))

        # Step 4: Perform inverse affine transformation
        inv_affine_matrix = torch.inverse(affine_matrix)
        if self.affine_randomness is not None:
            inv_affine_matrix += torch.randn_like(inv_affine_matrix) * self.affine_randomness
        x_restored = kornia.geometry.transform.warp_affine(x_transformed, inv_affine_matrix[:, :2, :], (3*h, 3*w))

        # Step 5: Crop the image back
        x_cropped = x_restored[:, :, h:2*h, w:2*w]

        return x_cropped.to(x_dtype)

class DiffJPEG:
    def __init__(self, min_quality: int, max_quality: int, p: float = 0.8):
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.p = p

    def jpeg_compress(self, image_tensor: torch.Tensor, quality: int):
        if image_tensor.shape[0] != 1:
            raise NotImplementedError(f'`jpeg_compress` only implemented for a batch of 1 image; got tensor: {image_tensor.shape}.')
        bytes_io = BytesIO()
        Image.fromarray((image_tensor)[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)).save(bytes_io, format='JPEG', quality=quality)
        # return Image.open(bytes_io)
        return torch.tensor(np.array(Image.open(bytes_io))).permute(2, 0, 1).float().to(image_tensor.device, dtype = image_tensor.dtype).unsqueeze(0)

    def diff_jpeg_compress(self, image_tensor: torch.Tensor, quality: int):
        with torch.no_grad():
            image_tensor_compressed = self.jpeg_compress(image_tensor, quality)
            difference = (image_tensor - image_tensor_compressed).detach()
        return image_tensor - difference
    
    def __call__(self, x: torch.Tensor):
        if random.random() < self.p:
            return self.diff_jpeg_compress(x, random.randint(self.min_quality, self.max_quality))
        return x
    
class RandAugmentation:
    def __init__(self, sigma: int = 1):
        self.sigma = sigma

    def __call__(self, x):
        return x + torch.randn_like(x) * self.sigma / 127.5

class UpscaleAugmentation:
    def __init__(self, lower_bound: int = 5, upper_bound: int = 15):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, x):
        _, _, h, w = x.shape
        upscaled = kornia.geometry.transform.resize(x, (h * np.random.randint(self.lower_bound, self.upper_bound), w * np.random.randint(self.lower_bound, self.upper_bound)))
        return kornia.geometry.transform.resize(upscaled, (h, w))

class RandomAffine:
    def __init__(self, randomness: float, device: Union[torch.device, str, None] = None):
        self.randomness = randomness
        self.base_affine_matrix = torch.eye(2, 3, dtype = torch.float32)
        self.device = device
        if self.device is not None:
            self.base_affine_matrix = self.base_affine_matrix.to(device)

    def __call__(self, image_tensor: torch.Tensor):
        b, _, h, w = image_tensor.shape
        affine_matrix = self.base_affine_matrix.repeat(b, 1, 1)
        affine_matrix += torch.randn_like(affine_matrix) * self.randomness
        if self.device is None:
            affine_matrix = affine_matrix.to(image_tensor.device)
        elif affine_matrix.device != image_tensor.device:
            raise RuntimeError(f'In RandomAffine augmentation: device mismatch between image tensor ({image_tensor.device}) and the affine matrix ({affine_matrix.device}).')
        return kornia.geometry.transform.warp_affine(image_tensor.to(affine_matrix.dtype), affine_matrix, (h, w)).to(image_tensor.dtype)