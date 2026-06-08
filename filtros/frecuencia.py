import cv2
import numpy as np


def aplicar_dft_magnitude(img):
    """
    Espectro de magnitud de la Transformada Discreta de Fourier (DFT).
    Devuelve imagen en escala de grises con el espectro centrado.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    dft       = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

    # Escala logarítmica + normalización 0-255
    magnitude_log  = 20 * np.log(magnitude + 1)
    magnitude_norm = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(magnitude_norm)


def aplicar_dct_magnitude(img):
    """
    Transformada Discreta del Coseno (DCT) — visualización del espectro.

    Estrategia robusta:
      1. Convierte a escala de grises.
      2. Hace las dimensiones PARES (obligatorio para cv2.dct).
         Se añade un píxel de padding con BORDER_REFLECT si es necesario
         y se recorta al tamaño original antes de devolver el resultado.
      3. Si cv2.dct lanza cualquier error (versión de OpenCV, tamaño extraño,
         etc.) usa el módulo FFT de numpy como fallback, que siempre funciona.
    """
    # ── 1. Escala de grises ──────────────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    h_orig, w_orig = gray.shape[:2]

    # ── 2. Padding a dimensiones pares ───────────────────────────────────────
    pad_h = h_orig % 2   # 0 si ya es par, 1 si es impar
    pad_w = w_orig % 2
    if pad_h or pad_w:
        gray = cv2.copyMakeBorder(
            gray, 0, pad_h, 0, pad_w,
            cv2.BORDER_REFLECT_101
        )

    # ── 3. Intentar DCT de OpenCV ────────────────────────────────────────────
    img_float = np.float32(gray) / 255.0
    try:
        dct        = cv2.dct(img_float)
        dct_log    = 20 * np.log(np.abs(dct) + 1e-6)
        dct_norm   = cv2.normalize(dct_log, None, 0, 255, cv2.NORM_MINMAX)
        result     = np.uint8(dct_norm[:h_orig, :w_orig])

    except (cv2.error, Exception):
        # ── Fallback: FFT 2D de numpy (compatible con cualquier tamaño) ──────
        fft2      = np.fft.fft2(gray.astype(np.float64))
        fft_shift = np.fft.fftshift(fft2)
        magnitude = np.abs(fft_shift)
        mag_log   = 20 * np.log(magnitude + 1)
        mag_norm  = cv2.normalize(mag_log, None, 0, 255, cv2.NORM_MINMAX)
        result    = np.uint8(mag_norm[:h_orig, :w_orig])

    return result
