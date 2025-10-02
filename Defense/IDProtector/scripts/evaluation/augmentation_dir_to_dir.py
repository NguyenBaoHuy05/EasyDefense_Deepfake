import os
import cv2
import numpy as np
from PIL import Image
import random

import torch
from utils.augmentations import RandomAffine
from typing import Union

import argparse

def jpeg_compression(image, quality=85):
    """Apply JPEG compression."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def add_gaussian_noise(image, mean=0, var=10):
    """Add Gaussian noise to the image."""
    row, col, ch = image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255)  # Keep pixel values between 0 and 255
    return noisy_image.astype(np.uint8)

def crop_and_resize(image, scale_min=0.9, scale_max=1.0):
    """Randomly crop and resize the image."""
    h, w, _ = image.shape
    scale = random.uniform(scale_min, scale_max)
    new_h, new_w = int(h * scale), int(w * scale)
    
    start_row = random.randint(0, h - new_h)
    start_col = random.randint(0, w - new_w)
    
    cropped_image = image[start_row:start_row + new_h, start_col:start_col + new_w]
    resized_image = cv2.resize(cropped_image, (w, h))  # Resize back to original size
    return resized_image

# def crop_and_resize(image: np.ndarray, padding_factor: float = 0.2) -> np.ndarray:
#     h, w = image.shape[:2]
#     scale = 1 - padding_factor
#     new_h, new_w = int(h * scale), int(w * scale)
    
#     resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#     output = np.zeros((h, w, 3), dtype=np.uint8)
    
#     top = np.random.randint(0, h - new_h)
#     left = np.random.randint(0, w - new_w)
#     output[top:top+new_h, left:left+new_w] = resized_image
#     return output

@torch.no_grad()
def random_affine(image: np.ndarray, random_affine_instance: RandomAffine) -> np.ndarray:
    affine_transformer = random_affine_instance

    # Convert OpenCV image (HWC, BGR) to PyTorch tensor (BCHW, RGB)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # Apply random affine transformation
    transformed_tensor = affine_transformer(image_tensor)

    # Convert back to OpenCV format (HWC, BGR)
    transformed_image = np.clip(transformed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    
    return transformed_image

def process_images(source_dir, save_dir):
    """Process images in the source_dir and save augmented images to save_dir."""
    if os.path.exists(os.path.join(save_dir, 'compressed')) or os.path.exists(os.path.join(save_dir, 'noisy')) or os.path.exists(os.path.join(save_dir, 'cropped')) or os.path.exists(os.path.join(save_dir, 'affine')):
        raise RuntimeError('Non-empty save dirs. Remove existing save dirs first.')
    
    # Create save dirs
    os.makedirs(os.path.join(save_dir, 'compressed'))
    os.makedirs(os.path.join(save_dir, 'noisy'))
    os.makedirs(os.path.join(save_dir, 'cropped'))
    os.makedirs(os.path.join(save_dir, 'affine'))

    random_affine_instance = RandomAffine(randomness = 5e-2, device = 'cpu')

    for filename in os.listdir(source_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Load image
            image_path = os.path.join(source_dir, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Failed to load {image_path}")
                continue

            # Apply JPEG compression and save
            compressed_image = jpeg_compression(image, quality=85)
            compressed_save_path = os.path.join(save_dir, 'compressed', os.path.splitext(filename)[0] + '.png')
            cv2.imwrite(compressed_save_path, compressed_image)
            print(f"Saved JPEG compressed image to {compressed_save_path}")

            # Apply Gaussian noise and save
            noisy_image = add_gaussian_noise(image)
            noisy_save_path = os.path.join(save_dir, 'noisy', os.path.splitext(filename)[0] + '.png')
            cv2.imwrite(noisy_save_path, noisy_image)
            print(f"Saved noisy image to {noisy_save_path}")

            # Apply crop & resize and save
            cropped_image = crop_and_resize(image)
            cropped_save_path = os.path.join(save_dir, 'cropped', os.path.splitext(filename)[0] + '.png')
            cv2.imwrite(cropped_save_path, cropped_image)
            print(f"Saved cropped and resized image to {cropped_save_path}")

            # Apply random affine and save
            affine_transformed_image = random_affine(image, random_affine_instance)
            affine_save_path = os.path.join(save_dir, 'affine', os.path.splitext(filename)[0] + '.png')
            cv2.imwrite(affine_save_path, affine_transformed_image)
            print(f'Saved affine image to {affine_save_path}')

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--source_dir', type=str, required=True, help='Path to the source directory')
#     parser.add_argument('--save_dir', type=str, required=True, help='Path to the save directory')
    
#     args = parser.parse_args()

#     process_images(args.source_dir, args.save_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--source_dir', type=str, required=True, help='Path to the source directory')
parser.add_argument('--save_dir', type=str, required=True, help='Path to the save directory')

args = parser.parse_args()

process_images(args.source_dir, args.save_dir)
