from typing import Tuple, Union
from insightface.app import FaceAnalysis
from modules.instantid_prep_pipe import resize_img
from PIL import Image
import cv2
import numpy as np

def get_face_info(image: Image.Image, app: FaceAnalysis):
    emb_face = resize_img(image)
    emb_face_info = app.get(cv2.cvtColor(np.array(emb_face), cv2.COLOR_RGB2BGR))
    return emb_face_info

def get_face_feature(emb_face_info: list):
    return sorted(emb_face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]['embedding']

def get_insightface_metrics(pil_image: Image.Image, insightface_app: FaceAnalysis) -> Tuple[bool, Union[None, np.ndarray]]:
    '''
    Detect whether a face exists, and if so, calculate its arcface embedding.
    Inputs:
        `pil_image`: a PIL RGB image
        `insightface_app`: an instance of the FaceAnalysis app
    Output:
        A tuple containing:
            1. A bool value indicating whether a face can be successfully detected
            2. A length-512 arcface embedding; if face detection fails, returns None
    '''
    emb_face_info = get_face_info(pil_image, insightface_app)
    if len(emb_face_info) == 0:
        return False, None
    else:
        return True, get_face_feature(emb_face_info)