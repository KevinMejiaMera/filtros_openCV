import cv2
import numpy as np

# ==========================================
# 1. GEOMETRIC EXTENDED
# ==========================================

def ext_scale_up(img):
    """Escalar 1.5x."""
    return cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

def ext_scale_down(img):
    """Escalar 0.5x."""
    return cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

def ext_shear_x(img):
    """Cizallamiento Horizontal (Shear X)."""
    rows, cols, _ = img.shape
    M = np.float32([[1, 0.5, 0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (int(cols*1.5), rows))

def ext_shear_y(img):
    """Cizallamiento Vertical (Shear Y)."""
    rows, cols, _ = img.shape
    M = np.float32([[1, 0, 0], [0.5, 1, 0]])
    return cv2.warpAffine(img, M, (cols, int(rows*1.5)))

def ext_crop_16_9(img):
    """Recorte Aspecto 16:9 Central."""
    h, w, _ = img.shape
    target_h = int(w * 9 / 16)
    if target_h > h:
        # Width is limiting factor, crop width instead
        target_w = int(h * 16 / 9)
        start_x = (w - target_w) // 2
        return img[:, start_x:start_x+target_w]
    else:
        start_y = (h - target_h) // 2
        return img[start_y:start_y+target_h, :]

# ==========================================
# 2. COLOR CHANNELS & MANIPULATION
# ==========================================

def ext_channel_red(img): return img[:,:,2]
def ext_channel_green(img): return img[:,:,1]
def ext_channel_blue(img): return img[:,:,0]

def ext_show_hue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[:,:,0]

def ext_show_sat(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[:,:,1]

def ext_show_val(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[:,:,2]

def ext_swap_rb(img):
    """Intercambiar canales Rojo y Azul."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR -> RGB effectively swaps RB

def ext_color_map_ocean(img): return cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
def ext_color_map_bone(img): return cv2.applyColorMap(img, cv2.COLORMAP_BONE)
def ext_color_map_spring(img): return cv2.applyColorMap(img, cv2.COLORMAP_SPRING)
def ext_color_map_summer(img): return cv2.applyColorMap(img, cv2.COLORMAP_SUMMER)
def ext_color_map_autumn(img): return cv2.applyColorMap(img, cv2.COLORMAP_AUTUMN)
def ext_color_map_winter(img): return cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
def ext_color_map_rainbow(img): return cv2.applyColorMap(img, cv2.COLORMAP_RAINBOW)

# ==========================================
# 3. NOISE & RESTORATION
# ==========================================

def ext_noise_salt(img, prob=0.02):
    """Ruido Sal (Puntos blancos)."""
    output = np.copy(img)
    thres = 1 - prob
    output[np.random.random(img.shape[:2]) > thres] = 255
    return output

def ext_noise_pepper(img, prob=0.02):
    """Ruido Pimienta (Puntos negros)."""
    output = np.copy(img)
    output[np.random.random(img.shape[:2]) < prob] = 0
    return output

def ext_denoise_fast(img):
    """Eliminación de ruido rápida."""
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# ==========================================
# 4. MORPHOLOGY EXTENDED
# ==========================================

def ext_morph_tophat(img):
    """Top Hat (Resalta objetos claros en fondo oscuro)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

def ext_morph_blackhat(img):
    """Black Hat (Resalta objetos oscuros en fondo claro)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

# ==========================================
# 5. ARTISTIC & DISTORTION EXTENDED
# ==========================================

def ext_glass_effect(img):
    """Efecto Vidrio Esmerilado (Desplazamiento aleatorio)."""
    rows, cols, _ = img.shape
    img_output = np.zeros_like(img)
    for i in range(rows):
        for j in range(cols):
            rand_x = np.random.randint(-5, 6)
            rand_y = np.random.randint(-5, 6)
            try:
                img_output[i,j] = img[i+rand_y, j+rand_x]
            except:
                img_output[i,j] = img[i,j]
    return img_output

def ext_ripple(img):
    """Efecto Ondulación (Ripple)."""
    rows, cols, _ = img.shape
    img_output = np.zeros_like(img)
    for i in range(rows):
        for j in range(cols):
            offset_x = int(10 * np.sin(2 * np.pi * i / 30))
            offset_y = int(10 * np.cos(2 * np.pi * j / 30))
            if i+offset_y < rows and j+offset_x < cols and i+offset_y>=0 and j+offset_x>=0:
                img_output[i,j] = img[i+offset_y, j+offset_x]
            else:
                img_output[i,j] = 0
    return img_output

def ext_emboss_90(img):
    kernel = np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
    return cv2.filter2D(img, -1, kernel)

def ext_vignette_white(img):
    """Viñeta Blanca (Ensueño)."""
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/2)
    kernel_y = cv2.getGaussianKernel(rows, rows/2)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    output = np.copy(img)
    # White background blend
    for i in range(3):
        output[:, :, i] = output[:, :, i] * mask + 255 * (1 - mask)
    return output

# Dictionary map
EXT_FILTERS = {
    'geo_scale_up': ext_scale_up,
    'geo_scale_down': ext_scale_down,
    'geo_shear_x': ext_shear_x,
    'geo_shear_y': ext_shear_y,
    'geo_crop_169': ext_crop_16_9,
    'col_red': ext_channel_red,
    'col_green': ext_channel_green,
    'col_blue': ext_channel_blue,
    'col_hue': ext_show_hue,
    'col_sat': ext_show_sat,
    'col_val': ext_show_val,
    'col_swap_rb': ext_swap_rb,
    'col_map_ocean': ext_color_map_ocean,
    'col_map_bone': ext_color_map_bone,
    'col_map_spring': ext_color_map_spring,
    'col_map_summer': ext_color_map_summer,
    'col_map_autumn': ext_color_map_autumn,
    'col_map_winter': ext_color_map_winter,
    'col_map_rainbow': ext_color_map_rainbow,
    'noise_salt': ext_noise_salt,
    'noise_pepper': ext_noise_pepper,
    'denoise_fast': ext_denoise_fast,
    'morph_tophat': ext_morph_tophat,
    'morph_blackhat': ext_morph_blackhat,
    'art_glass': ext_glass_effect,
    'art_ripple': ext_ripple,
    'art_emboss_90': ext_emboss_90,
    'art_vignette_white': ext_vignette_white
}
