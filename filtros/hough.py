import cv2
import numpy as np

def aplicar_hough_lineas_probabilistico(img):
    """
    Detecta segmentos de línea (Probabilistic Hough Transform)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    img_lines = img.copy()
    
    # HoughLinesP devuelve los puntos extremos de los segmentos
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img_lines

def aplicar_hough_lineas_busqueda(img):
    """
    Transformada de Hough Estándar (Líneas Infinitas)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    img_lines = img.copy()
    
    # HoughLines devuelve rho y theta
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # Calculamos dos puntos muy lejanos para simular la linea infinita
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
    return img_lines

def aplicar_hough_circulos(img):
    output = img.copy()
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    gray = cv2.medianBlur(gray, 5)
    
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=10, maxRadius=100)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # centro
            cv2.circle(output, center, 1, (0, 100, 100), 3)
            # contorno
            radius = i[2]
            cv2.circle(output, center, radius, (255, 0, 255), 3)
            
    return output
