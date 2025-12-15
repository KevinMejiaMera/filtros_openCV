import cv2
import numpy as np

def aplicar_blur(img):
    return cv2.blur(img, (9, 9))

def aplicar_gaussiano(img):
    return cv2.GaussianBlur(img, (9, 9), 0)

def aplicar_mediana(img):
    return cv2.medianBlur(img, 9)

def aplicar_bilateral(img):
    return cv2.bilateralFilter(img, 9, 75, 75)
