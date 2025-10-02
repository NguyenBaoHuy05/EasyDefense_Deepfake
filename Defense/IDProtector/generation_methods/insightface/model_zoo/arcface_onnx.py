# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 

from __future__ import division
import numpy as npD
import cv2
import onnx
import onnxruntime
from ..utils import face_align

import torch
from onnx2torch import convert
import pdb


__all__ = [
    'ArcFaceONNX',
]


class ArcFaceONNX:
    def __init__(self, model_file=None, session=None):
        assert model_file is not None
        self.model_file = model_file
        self.session = session
        self.taskname = 'recognition'
        find_sub = False
        find_mul = False
        model = onnx.load(self.model_file)
        # pdb.set_trace()
        self.torch_model = convert(model)
        graph = model.graph
        for nid, node in enumerate(graph.node[:8]):
            #print(nid, node.name)
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
        if find_sub and find_mul:
            #mxnet arcface model
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        #print('input mean and std:', self.input_mean, self.input_std)
        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names)==1
        self.output_shape = outputs[0].shape
        
    def get_torch_model(self):
        return self.torch_model

    def prepare(self, ctx_id, **kwargs):
        if ctx_id<0:
            self.session.set_providers(['CPUExecutionProvider'])

    def get(self, img):
        raise DeprecationWarning(f'This function needs further auditing.')
        # aimg = face_align.norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
        embedding = self.get_feat(img).flatten()
        #face.embedding = self.forward(aimg).flatten()
        print("yes*********************************")
        # pdb.set_trace()
        return embedding

    def compute_sim(self, feat1, feat2):
        from numpy.linalg import norm
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim

    # def get_feat(self, imgs):
    #     if not isinstance(imgs, list):
    #         imgs = [imgs]
    #     input_size = self.input_size
    #     blob = cv2.dnn.blobFromImages(imgs, 1.0 / self.input_std, input_size,
    #                                   (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
    #     net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
    #     return net_out
    
    def normalize(self, img_tensor):
        # Normalize the image tensor using the pre-defined mean and standard deviation for each channel
        mean_tensor = torch.tensor([127.5, 127.5, 127.5]).view(1, 3, 1, 1)
        std_tensor = torch.tensor([127.5, 127.5, 127.5]).view(1, 3, 1, 1)
        img_tensor = img_tensor[:, [2, 1, 0], :, :]
        return (img_tensor - mean_tensor) / std_tensor
    
    def get_feat(self, img):
        raise DeprecationWarning(f'This function is deprecated. Originally implemented by yiren for debug purposes? Uncovered 20240917.')
        input_size = self.input_size
        # pdb.set_trace()
        # cv2.imwrite("/Users/yiren/projects/stable_signature2/stable_signature/IDprotector/test1.png",img)
        # Ensure img is in the correct format, assuming img is already loaded as a numpy array
        if img.ndim == 3:  # Single image with three dimensions (H, W, C)
            img = cv2.resize(img, input_size)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # Convert to torch tensor and reorder dimensions
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
            img_tensor = self.normalize(img_tensor)  # Apply normalization

        # Set the model to evaluation mode
        self.torch_model.eval()
        torch.save(img_tensor, './saved_tensor1.pt')
        with torch.no_grad():  # Disable gradient computation
            net_out = self.torch_model(img_tensor) 
        return net_out.numpy()  # Convert output to NumPy array if needed
    
    # def forward(self, batch_data):
    #     blob = (batch_data - self.input_mean) / self.input_std
    #     net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
    #     return net_out
    
    def forward(self, batch_data):
        # Assuming batch_data is a batch of images with shape (N, H, W, C) where N is the batch size
        # and images are in the standard HWC format (height, width, channels)

        # Resize and reorder dimensions to CHW format expected by PyTorch
        batch_imgs = [cv2.resize(img, self.input_size) for img in batch_data]
        batch_tensors = [torch.from_numpy(img).permute(2, 0, 1).float() for img in batch_imgs]  # Convert to tensors and reorder dimensions

        # Stack all image tensors to create a single batch tensor
        batch_tensor = torch.stack(batch_tensors)  # This combines all the image tensors into one batch tensor

        # Normalize the entire batch
        batch_tensor = self.normalize(batch_tensor)

        # Set the model to evaluation mode
        self.torch_model.eval()

        # Disable gradient computation since we are only doing forward pass (useful for inference and reduces memory usage)
        with torch.no_grad():
            net_out = self.torch_model(batch_tensor)  # Perform inference on the batch

        return net_out.numpy()  # Convert PyTorch tensor to NumPy array if needed for further processing

