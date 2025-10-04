import torch
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def gaussian_blur(x, size=5, sigma=1.0):
    """Applies Gaussian blur to each feature map independently using Float."""
    x = x.float()  # Ensure input is Float
    n = torch.arange(0, size, device=x.device).float() - (size - 1) / 2.0
    gauss_kernel = torch.exp(-n ** 2 / (2 * sigma ** 2)).float()
    gauss_kernel = gauss_kernel / gauss_kernel.sum()
    gauss_kernel = gauss_kernel[:, None] * gauss_kernel[None, :]
    gauss_kernel = gauss_kernel.view(1, 1, size, size).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, gauss_kernel, padding=size // 2, groups=x.shape[1])

def sobel_gradients(x):
    """Computes the Sobel gradient magnitudes for each feature map on GPU."""
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=x.device).view(1, 1, 3,
                                                                                                            3).repeat(
        x.shape[1], 1, 1, 1)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=x.device).view(1, 1, 3,
                                                                                                            3).repeat(
        x.shape[1], 1, 1, 1)
    grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.shape[1])
    grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.shape[1])
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    return grad_magnitude

def canny_edge_loss(x, y):
    """Calculates the MSE loss based on the Sobel gradient magnitudes for each pair of feature maps on GPU."""
    blurred_x = gaussian_blur(x)
    blurred_y = gaussian_blur(y)
    mag_x = sobel_gradients(blurred_x)
    mag_y = sobel_gradients(blurred_y)
    loss = F.mse_loss(mag_x, mag_y)
    return loss, mag_x, mag_y

def batch_process(input_x, input_y, save_images=False, save_dir=None, current_step=None):
    """Processes batches and sub-batches to compute the average edge detection loss across all feature maps on GPU.
    If save_images is True, visualizes and saves sub_input_x and sub_input_y."""
    total_loss = 0
    count = 0
    for batch_idx in range(input_x.shape[0]):
        for sub_batch_idx in range(input_x.shape[1]):
            sub_input_x = input_x[batch_idx, sub_batch_idx].unsqueeze(0).to(device)  # Move to GPU
            sub_input_y = input_y[batch_idx, sub_batch_idx].unsqueeze(0).to(device)  # Move to GPU
            #loss = canny_edge_loss(sub_input_x, sub_input_y)  # Shape: (1, C, H, W)
            loss, mag_x, mag_y = canny_edge_loss(sub_input_x, sub_input_y)
            total_loss += loss.item()
            count += 1
    return total_loss / count