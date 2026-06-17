import cv2
import numpy as np

def aplicar_noise_salt_pepper(img, prob=0.02):
    """Ruido Sal y Pimienta."""
    output = np.copy(img)
    thres = 1 - prob
    output[np.random.random(img.shape[:2]) > thres] = 255
    output[np.random.random(img.shape[:2]) < prob] = 0
    return output

def aplicar_g_noise(img):
    """Ruido Gaussiano."""
    mean = 0
    var = 100
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, img.shape).astype('uint8')
    return cv2.add(img, gauss)
