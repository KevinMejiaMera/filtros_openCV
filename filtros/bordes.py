import cv2
import numpy as np

def aplicar_canny(img):
    return cv2.Canny(img, 100, 200)