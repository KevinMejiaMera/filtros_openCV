import cv2
import numpy as np

def aplicar_erosion(img):
    kernel = np.ones((5,5), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def aplicar_dilatacion(img):
    kernel = np.ones((5,5), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)

def aplicar_apertura(img):
    kernel = np.ones((5,5), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def aplicar_cierre(img):
    kernel = np.ones((5,5), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def aplicar_gradiente(img):
    kernel = np.ones((5,5), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
