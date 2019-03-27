import os
import cv2
import copy
import numpy as np


def increase_contrast(image, brightness=30, contrast=150):
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    brightness  = 100 - 0.5 * np.mean(gray)
    img = np.int16(image)
    img = img * (contrast/127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    return img


def center_and_crop(image, size=224):
    ret = image.copy()
    x = ret.shape[0]
    y = ret.shape[1]
    l = min([x // 2, y // 2])
    crop = ret[ x // 2 - l : x // 2  + l,  y//2 - l : y//2 + l]
    crop = np.array(crop, dtype=np.float32)
    crop = cv2.resize(crop, (size, size))
    crop = np.array(crop, dtype=np.uint8)
    return crop


def preprocess(image):
    ret = center_and_crop(image)
    ret = increase_contrast(ret)
    return ret

