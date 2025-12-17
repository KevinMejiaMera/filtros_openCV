import cv2
import numpy as np

def aplicar_sharpen(img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

def aplicar_high_pass(img):
    # High pass filter removes low frequency (simulated by blurring)
    blur = cv2.GaussianBlur(img, (21, 21), 0)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0) # Simple sharpening/high-pass approximation

def aplicar_emboss(img):
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    result = cv2.filter2D(img, -1, kernel)
    return cv2.add(result, 128) # Offset to make it visible grey

def aplicar_unsharp_mask(img):
    gaussian = cv2.GaussianBlur(img, (9, 9), 10.0)
    return cv2.addWeighted(img, 1.5, gaussian, -0.5, 0, img)
