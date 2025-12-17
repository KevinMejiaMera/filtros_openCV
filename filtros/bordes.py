import cv2
import numpy as np

def aplicar_canny(img):
    return cv2.Canny(img, 100, 200)

def aplicar_sobel(img):
    sobel = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)
    return cv2.convertScaleAbs(sobel)

def aplicar_scharr(img):
    # Scharr es una variante de Sobel con mejor aproximación rotacional
    scharr = cv2.Scharr(img, cv2.CV_64F, 1, 0) # Solo ejemplo en X
    return cv2.convertScaleAbs(scharr)

def aplicar_laplaciano(img):
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return cv2.convertScaleAbs(lap)

def aplicar_log(img):
    # Laplacian of Gaussian (LoG): suavizado gaussiano + laplaciano
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY) if len(blurred.shape) == 3 else blurred
    # Aplicar Laplaciano
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    # Convertir de vuelta para visualización
    return cv2.convertScaleAbs(lap)

def aplicar_prewitt(img):
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(img, -1, kernelx)
    img_prewitty = cv2.filter2D(img, -1, kernely)
    return img_prewittx + img_prewitty

def aplicar_roberts(img):
    roberts_cross_v = np.array([[1, 0], [0, -1]])
    roberts_cross_h = np.array([[0, 1], [-1, 0]])
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = gray.astype('float64') / 255.0
    
    vertical = cv2.filter2D(gray, -1, roberts_cross_v)
    horizontal = cv2.filter2D(gray, -1, roberts_cross_h)
    
    edged_img = np.sqrt(np.square(horizontal) + np.square(vertical))
    edged_img *= 255
    return edged_img.astype(np.uint8)
