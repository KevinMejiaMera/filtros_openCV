import cv2
import numpy as np

# ==========================================
# 1. AJUSTES DE COLOR Y LUZ (ADJUSTMENTS)
# ==========================================

def gimp_brightness(img, value=30):
    """Aumenta o disminuye el brillo."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value) 
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def gimp_contrast(img, alpha=1.5):
    """Ajusta el contraste (alpha > 1 aumenta)."""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

def gimp_saturation(img, value=50):
    """Aumenta la saturación."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, value)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def gimp_gamma(img, gamma=1.5):
    """Corrección Gamma."""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def gimp_invert(img):
    """Invierte los colores (Negativo)."""
    return cv2.bitwise_not(img)

def gimp_sepia(img):
    """Efecto Sepia."""
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    return cv2.transform(img, kernel)

def gimp_solarize(img, threshold=128):
    """Efecto Solarización: invierte píxeles sobre un umbral."""
    return np.where(img < threshold, img, 255 - img).astype(np.uint8)

def gimp_posterize(img, levels=4):
    """Reducción de colores (Posterización)."""
    factor = 255 / (levels - 1)
    return np.uint8(np.round(img / factor) * factor)

def gimp_temperature(img, val=30):
    """Ajuste de temperatura (Cálido/Frío)."""
    # Simple approach: increase R for warm, B for cool
    b, g, r = cv2.split(img)
    if val > 0: # Warm
        r = cv2.add(r, val)
        b = cv2.subtract(b, val)
    else: # Cool
        b = cv2.add(b, abs(val))
        r = cv2.subtract(r, abs(val))
    return cv2.merge([b, g, r])

def gimp_tint(img, val=30):
    """Ajuste de tinte (Verde/Magenta)."""
    b, g, r = cv2.split(img)
    g = cv2.add(g, val) # Greenish
    return cv2.merge([b, g, r])

# ==========================================
# 2. EFECTOS ARTÍSTICOS (ARTISTIC)
# ==========================================

def gimp_pencil_sketch_bw(img):
    """Boceto a Lápiz (Blanco y Negro)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    inv_blur = cv2.bitwise_not(blur)
    sketch = cv2.divide(gray, inv_blur, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def gimp_pencil_sketch_col(img):
    """Boceto a Lápiz (Color)."""
    # Similar logic mixed with original
    sketch_bw = gimp_pencil_sketch_bw(img)
    return cv2.bitwise_and(img, sketch_bw)

def gimp_cartoon(img):
    """Efecto Caricatura."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def gimp_oil_painting(img):
    """Efecto Pintura al Óleo."""
    return cv2.xphoto.oilPainting(img, 7, 1)

def gimp_watercolor(img):
    """Efecto Acuarela (Stylization)."""
    res = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
    return res

# ==========================================
# 3. DISTORSIONES Y GEOMETRÍA (DISTORTIONS)
# ==========================================

def gimp_vignette(img):
    """Viñeta Oscura."""
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/2)
    kernel_y = cv2.getGaussianKernel(rows, rows/2)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    output = np.copy(img)
    for i in range(3):
        output[:, :, i] = output[:, :, i] * mask
    return output

def gimp_pixelate(img, block_size=10):
    """Pixelado."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def gimp_glitch(img):
    """Glitch cromático simple (desplazamiento de canales)."""
    b, g, r = cv2.split(img)
    rows, cols = img.shape[:2]
    shift = 5
    
    # Shift Blue left
    b_shifted = np.roll(b, -shift, axis=1)
    # Shift Red right
    r_shifted = np.roll(r, shift, axis=1)
    
    return cv2.merge([b_shifted, g, r_shifted])

def gimp_wave(img):
    """Onda Sinusoidal vertical."""
    rows, cols = img.shape[:2]
    img_output = np.zeros(img.shape, dtype=img.dtype)
    for i in range(rows):
        for j in range(cols):
            offset_x = int(25.0 * np.sin(2 * 3.14 * i / 180))
            if j + offset_x < cols:
                img_output[i, j] = img[i, (j + offset_x) % cols]
            else:
                img_output[i, j] = 0
    return img_output

def gimp_mirror(img):
    """Efecto Espejo (Mitad izquierda reflejada)."""
    h, w, _ = img.shape
    half = img[:, :w//2]
    mirror = cv2.flip(half, 1)
    return np.hstack((half, mirror))

def gimp_crop_center(img):
    """Recorte al centro (cuadrado)."""
    h, w, _ = img.shape
    dim = min(h, w)
    center_x, center_y = w//2, h//2
    x = center_x - dim//2
    y = center_y - dim//2
    return img[y:y+dim, x:x+dim]

# ==========================================
# 4. RUIDO Y DETALLES (NOISE/DETAILS)
# ==========================================

def gimp_noise_gaussian(img):
    """Ruido Gaussiano."""
    mean = 0
    var = 100
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, img.shape).astype('uint8')
    return cv2.add(img, gauss)

def gimp_details_enhance(img):
    """Realce de detalles (Detail Enhance)."""
    return cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

# Adding 20 more distinct variations/combinations to reach ~50 total effects concept
def gimp_brightness_high(img): return gimp_brightness(img, 60)
def gimp_brightness_low(img): return gimp_brightness(img, -60)
def gimp_contrast_high(img): return gimp_contrast(img, 1.8)
def gimp_contrast_low(img): return gimp_contrast(img, 0.7)
def gimp_sat_high(img): return gimp_saturation(img, 80)
def gimp_sat_low(img): return gimp_saturation(img, -50)
def gimp_warm(img): return gimp_temperature(img, 40)
def gimp_cool(img): return gimp_temperature(img, -40)
def gimp_gamma_low(img): return gimp_gamma(img, 0.5)
def gimp_gamma_high(img): return gimp_gamma(img, 2.2)
def gimp_posterize_2(img): return gimp_posterize(img, 2)
def gimp_posterize_8(img): return gimp_posterize(img, 8)
def gimp_solarize_low(img): return gimp_solarize(img, 64)
def gimp_tint_magenta(img): 
    b, g, r = cv2.split(img)
    r = cv2.add(r, 40)
    b = cv2.add(b, 40)
    return cv2.merge([b, g, r])
def gimp_channel_blue(img):
    b, g, r = cv2.split(img)
    return cv2.merge([b, np.zeros_like(g), np.zeros_like(r)])
def gimp_channel_red(img):
    b, g, r = cv2.split(img)
    return cv2.merge([np.zeros_like(b), np.zeros_like(g), r])
def gimp_channel_green(img):
    b, g, r = cv2.split(img)
    return cv2.merge([np.zeros_like(b), g, np.zeros_like(r)])
def gimp_emboss_45(img):
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    return cv2.filter2D(img, -1, kernel)
def gimp_sharpen_heavy(img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

# Mapping dictionary for dispatch
GIMP_FILTERS = {
    'g_bright': gimp_brightness,
    'g_bright_plus': gimp_brightness_high,
    'g_bright_minus': gimp_brightness_low,
    'g_contrast': gimp_contrast,
    'g_contrast_plus': gimp_contrast_high,
    'g_contrast_minus': gimp_contrast_low,
    'g_sat': gimp_saturation,
    'g_sat_plus': gimp_sat_high,
    'g_sat_minus': gimp_sat_low,
    'g_gamma': gimp_gamma,
    'g_gamma_plus': gimp_gamma_high,
    'g_gamma_minus': gimp_gamma_low,
    'g_invert': gimp_invert,
    'g_sepia': gimp_sepia,
    'g_solarize': gimp_solarize,
    'g_posterize': gimp_posterize,
    'g_posterize_2': gimp_posterize_2,
    'g_warm': gimp_warm,
    'g_cool': gimp_cool,
    'g_tint_green': gimp_tint,
    'g_tint_magenta': gimp_tint_magenta,
    'g_sketch': gimp_pencil_sketch_bw,
    'g_sketch_col': gimp_pencil_sketch_col,
    'g_cartoon': gimp_cartoon,
    'g_oil': gimp_oil_painting,
    'g_water': gimp_watercolor,
    'g_vignette': gimp_vignette,
    'g_pixel': gimp_pixelate,
    'g_glitch': gimp_glitch,
    'g_wave': gimp_wave,
    'g_mirror': gimp_mirror,
    'g_crop': gimp_crop_center,
    'g_noise': gimp_noise_gaussian,
    'g_detail': gimp_details_enhance,
    'g_blue': gimp_channel_blue,
    'g_red': gimp_channel_red,
    'g_green': gimp_channel_green,
    'g_emboss_45': gimp_emboss_45,
    'g_sharpen_heavy': gimp_sharpen_heavy
}
