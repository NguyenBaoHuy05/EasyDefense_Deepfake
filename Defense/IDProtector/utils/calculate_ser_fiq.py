from .FaceImageQuality.face_image_quality import SER_FIQ
from PIL import Image
import numpy as np

def calculate_ser_fiq(cv2_image: np.ndarray, ser_fiq: SER_FIQ):

    # Align the image
    aligned_img = ser_fiq.apply_mtcnn(cv2_image)
    if aligned_img is None:
        return 'NA'
    
    # Calculate the quality score of the image
    # T=100 (default) is a good choice
    # Alpha and r parameters can be used to scale your
    # score distribution.
    score = ser_fiq.get_score(aligned_img, T=100)

    return score