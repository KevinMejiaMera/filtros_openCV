import cv2
import numpy as np

def aplicar_rotacion_90(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def aplicar_rotacion_180(img):
    return cv2.rotate(img, cv2.ROTATE_180)

def aplicar_volteo_horizontal(img):
    return cv2.flip(img, 1)

def aplicar_volteo_vertical(img):
    return cv2.flip(img, 0)
