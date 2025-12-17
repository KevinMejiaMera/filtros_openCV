import cv2
import numpy as np

def aplicar_escala_grises(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def aplicar_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def aplicar_ecualizacion_hist(img):
    if len(img.shape) == 3:
        # Equalize Y channel in YCrCb
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        return cv2.equalizeHist(img)

def aplicar_brillo_contraste(img, alpha=1.2, beta=30):
    # Alpha = Contrast control (1.0-3.0)
    # Beta = Brightness control (0-100)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def aplicar_pseudocolor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
