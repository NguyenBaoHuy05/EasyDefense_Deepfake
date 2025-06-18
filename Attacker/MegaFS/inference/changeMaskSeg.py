import os
import cv2
import numpy as np

label = {
    "background": 0,
    "skin": 1,
    "l_brow": 2,
    "r_brow": 3,
    "l_eye": 4,
    "r_eye": 5,
    "eye_g": 6,
    "l_ear": 7,
    "r_ear": 8,
    "ear_r": 9,
    "nose": 10,
    "mouth": 11,
    "u_lip": 12,
    "l_lip": 13,
    "neck": 14,
    "neck_l": 15,
    "cloth": 16,
    "hair": 17,
    "hat": 18,
}
for files in os.listdir("mask"):
    if files.endswith(".png"):
        file_path = os.path.join("mask", files)
        face = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        segmentation = np.zeros_like(face, dtype=np.uint8)
        break
for files in os.listdir("mask"):
    if files.endswith(".png"):
        file_name = os.path.basename(files)
        split_pos = file_name.find("_")
        idx = file_name[:split_pos]
        att = file_name[split_pos + 1 :].split(".")[0]

        file_path = os.path.join("mask", files)
        imgs = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        segmentation[imgs == 255] = label[att]
cv2.imwrite(f"maskSeg/{int(idx)}.png", segmentation)
