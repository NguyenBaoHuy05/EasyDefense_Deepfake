from torch.utils.data import Dataset
import os
import torchvision
import kornia
from PIL import Image
import torch
from math import ceil, floor

class ImageAnnotatedWithPriorsWithPGDDataset(Dataset):
    def __init__(self, clean_data_dir, pgd_data_dir, network_input_size):
        self.clean_data_dir = clean_data_dir
        self.pgd_data_dir = pgd_data_dir
        self.image_paths = []
        self.label_paths = []
        self.exists_pgd = []
        self.pgd_image_paths = []
        self.pgd_pt_paths = []
        self.network_input_size = network_input_size

        # Filter out images without corresponding .pt files
        for filename in os.listdir(clean_data_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(clean_data_dir, filename)
                label_path = os.path.splitext(image_path)[0] + '.pt'
                if os.path.exists(label_path):
                    self.image_paths.append(image_path)
                    self.label_paths.append(label_path)
                    pgd_image_path = os.path.join(pgd_data_dir, filename)
                    pgd_image_path = os.path.splitext(pgd_image_path)[0] + '.png'
                    pgd_pt_path = os.path.splitext(pgd_image_path)[0] + '.pt'
                    if os.path.exists(pgd_image_path) and os.path.exists(pgd_pt_path):
                        self.exists_pgd.append(1)
                        self.pgd_image_paths.append(pgd_image_path)
                        self.pgd_pt_paths.append(pgd_pt_path)
                    else:
                        self.exists_pgd.append(0)
                        self.pgd_image_paths.append(None)
                        self.pgd_pt_paths.append(None)

        # Define transform to resize image and convert to tensor
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.network_input_size, self.network_input_size), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image_tensor = self.transform(image)
        # Returns clip-cropped images normalised to [0, 1]
        # image = torch.tensor(np.array(image)).permute(2, 0, 1)[None, :].float()
        # image = torch.clamp(clip_preprocessing_pipeline_v3(image, None)[0], 0, 1)

        width, height = image.size
        # Get arcface prior channel
        affine_M = torch.load(self.label_paths[idx], weights_only = False)
        affine_mask = self.get_affine_mask(height, width, affine_M, 112)
        # Get clip prior channel
        clip_prior = self.get_clip_prior(height, width, self.network_input_size)
        image_tensor = torch.cat([image_tensor, affine_mask[None, ...], clip_prior[None, ...]], dim = 0)

        # Return image and index as tuple
        return image_tensor, idx, self.exists_pgd[idx] # (M_matrix, original_size)
    
    def get_original_image(self, idx):
        return Image.open(self.image_paths[idx])
    
    def get_affine_M(self, idx):
        return torch.load(self.label_paths[idx])
    
    def get_pgd_image(self, idx):
        return Image.open(self.pgd_image_paths[idx])
    
    def get_pgd_perturbations(self, idx):
        return torch.load(self.pgd_pt_paths[idx])
    
    def get_affine_mask(self, height: int, width: int, affine_M: torch.Tensor, affine_size: int):

        max_side, min_side, base_pixel_number = 1280, 1024, 64
        ratio = min_side / min(height, width)
        w_resize, h_resize = round(ratio * width), round(ratio * height)
        ratio = max_side / max(h_resize, w_resize)
        w_resize_new, h_resize_new = round(ratio * w_resize), round(ratio * h_resize)
        w_resize_new = (w_resize_new // base_pixel_number) * base_pixel_number
        h_resize_new = (h_resize_new // base_pixel_number) * base_pixel_number

        y, x = torch.meshgrid(torch.arange(h_resize_new), torch.arange(w_resize_new), indexing='ij')
        coords = torch.stack([x, y], dim=-1).float()
        coords_homogeneous = torch.cat([coords, torch.ones_like(coords[..., :1])], dim=-1)
        transformed_coords = torch.matmul(affine_M, coords_homogeneous.view(-1, 3)[..., None]).squeeze(-1).view(h_resize_new, w_resize_new, 2)

        mask_x = (transformed_coords[..., 0] >= 0) & (transformed_coords[..., 0] < affine_size)
        mask_y = (transformed_coords[..., 1] >= 0) & (transformed_coords[..., 1] < affine_size)
        # return kornia.geometry.transform.resize((mask_x & mask_y).float(), (height, width), interpolation = 'bicubic', antialias = True)
        return kornia.geometry.transform.resize((mask_x & mask_y).float(), (self.network_input_size, self.network_input_size), interpolation = 'bicubic', antialias = True)
    
    @torch.no_grad()
    def get_clip_prior(self, h: int, w: int, network_input_size: int):
        if h > w:
            h_start = ceil((h - w) / 2 * network_input_size / h)
            h_end = floor((h - w) / 2 * network_input_size / h + (w * network_input_size) / h)
            w_start = 0
            w_end = network_input_size
        elif h < w:
            h_start = 0
            h_end = network_input_size
            w_start = ceil((w - h) / 2 * network_input_size / w)
            w_end = floor((w - h) / 2 * network_input_size / w + (h * network_input_size) / w)
        elif h == w:
            h_start = 0
            h_end = network_input_size
            w_start = 0
            w_end = network_input_size
        else:
            raise RuntimeError(f'Undefined branch in `get_clip_prior`.')

        clip_prior_channel = torch.zeros(network_input_size, network_input_size, dtype = torch.float32)
        clip_prior_channel[h_start : h_end, w_start : w_end] = 1
        return clip_prior_channel