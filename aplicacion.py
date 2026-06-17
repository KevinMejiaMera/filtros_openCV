from flask import Flask, render_template, request, url_for, jsonify
import cv2
import numpy as np
import os
from datetime import datetime
from urllib.parse import unquote

# ─── Importar SOLO los filtros que se usan en la interfaz ────────────────────
from filtros import (
    # Suavizado y Realce
    aplicar_blur, aplicar_gaussiano, aplicar_mediana, aplicar_bilateral,
    aplicar_sharpen, aplicar_high_pass, aplicar_unsharp_mask,
    # Bordes
    aplicar_canny,
    # Frecuencia
    aplicar_dft_magnitude, aplicar_dct_magnitude,
    # Ruido
    aplicar_noise_salt_pepper, aplicar_g_noise
)

aplicacion = Flask(__name__)
aplicacion.config['CARPETA_SUBIDAS'] = os.path.join(aplicacion.root_path, 'static', 'processed')

if not os.path.exists(aplicacion.config['CARPETA_SUBIDAS']):
    os.makedirs(aplicacion.config['CARPETA_SUBIDAS'])

# ─── DESPACHO DE FILTROS ─────────────────────────────────────────────────────
def aplicar_filtro_logica(img, filtro):
    # Suavizado y Realce
    if   filtro == 'blur':      return aplicar_blur(img)
    elif filtro == 'gaussiano': return aplicar_gaussiano(img)
    elif filtro == 'mediana':   return aplicar_mediana(img)
    elif filtro == 'bilateral': return aplicar_bilateral(img)
    elif filtro == 'sharpen':   return aplicar_sharpen(img)
    elif filtro == 'high_pass': return aplicar_high_pass(img)
    elif filtro == 'unsharp':   return aplicar_unsharp_mask(img)
    # Bordes
    elif filtro == 'canny':     return aplicar_canny(img)
    # Frecuencia
    elif filtro == 'dft':       return aplicar_dft_magnitude(img)
    elif filtro == 'dct':       return aplicar_dct_magnitude(img)
    # Ruido
    elif filtro == 'noise_salt_pepper': return aplicar_noise_salt_pepper(img)
    elif filtro == 'g_noise': return aplicar_g_noise(img)
    else:
        return img  # 'original' u otros: devuelve imagen sin cambios

# ─── DESCRIPCIONES TÉCNICAS (solo los 12 filtros activos) ────────────────────
def obtener_descripcion(filtro):
    descripciones = {
        'blur':              'Suaviza promediando píxeles vecinos en una ventana de 9×9.',
        'gaussiano':         'Suaviza con distribución Gaussiana, reduce ruido de forma natural.',
        'mediana':           'Elimina ruido sal y pimienta conservando bien los bordes.',
        'bilateral':         'Suaviza la textura preservando los contornos (efecto piel suave).',
        'sharpen':           'Aumenta la nitidez enfatizando los bordes con kernel de convolución.',
        'high_pass':         'Paso Alto: realza detalles finos sustrayendo el componente borroso.',
        'unsharp':           'Unsharp Mask: mejora el enfoque mediante sustracción de desenfoque.',
        'canny':             'Detección de bordes multi-etapa de alta precisión (Canny, 1986).',
        'dft':               'Transformada de Fourier Discreta: espectro de magnitud de frecuencias.',
        'dct':               'Transformada Discreta del Coseno: concentra energía en baja frecuencia.',
        'noise_salt_pepper': 'Agrega ruido impulsivo de sal (blanco) y pimienta (negro) al azar.',
        'g_noise':           'Agrega ruido Gaussiano: interferencia aleatoria de distribución normal.',
    }
    return descripciones.get(filtro, 'Filtro aplicado correctamente.')

# ─── RUTAS ───────────────────────────────────────────────────────────────────
@aplicacion.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@aplicacion.route('/subir', methods=['POST'])
def subir():
    archivo = request.files.get('file')
    filtro  = request.form.get('filtro', 'original')
    if not archivo:
        return jsonify({'error': 'No se recibió ningún archivo'}), 400

    nombre_archivo = datetime.now().strftime('%Y%m%d%H%M%S') + '_' + archivo.filename
    ruta_archivo   = os.path.join(aplicacion.config['CARPETA_SUBIDAS'], nombre_archivo)
    archivo.save(ruta_archivo)
    return procesar_imagen_api(ruta_archivo, nombre_archivo, filtro)

@aplicacion.route('/api_procesamiento_local', methods=['POST'])
def api_procesamiento_local():
    nombre_imagen = unquote(request.form.get('image_name', ''))
    filtro        = request.form.get('filtro', 'original')
    origen        = request.form.get('origin', 'processed')

    ruta_original = os.path.join(aplicacion.config['CARPETA_SUBIDAS'], nombre_imagen)

    if not os.path.exists(ruta_original):
        return jsonify({'error': f'Imagen no encontrada: {ruta_original}'}), 404

    # Copiar la imagen original para conservarla y procesar sobre la copia
    nombre_copia = 'local_' + datetime.now().strftime('%Y%m%d%H%M%S') + '_' + nombre_imagen
    ruta_copia   = os.path.join(aplicacion.config['CARPETA_SUBIDAS'], nombre_copia)

    img = cv2.imread(ruta_original)
    if img is None:
        return jsonify({'error': 'No se pudo leer la imagen original'}), 500
    cv2.imwrite(ruta_copia, img)

    return procesar_imagen_api(ruta_copia, nombre_copia, filtro)

# ─── LÓGICA CENTRAL DE PROCESAMIENTO ────────────────────────────────────────
def procesar_imagen_api(ruta_archivo, nombre_archivo, filtro):
    img = cv2.imread(ruta_archivo)

    # Fallback para formatos que OpenCV no lee directamente (PNG-RGBA, WEBP, TIFF…)
    if img is None:
        try:
            from PIL import Image
            img = cv2.cvtColor(
                np.array(Image.open(ruta_archivo).convert('RGB')),
                cv2.COLOR_RGB2BGR
            )
            cv2.imwrite(ruta_archivo, img)
        except Exception as e:
            return jsonify({'error': f'Formato no soportado: {str(e)}'}), 400

    h, w = img.shape[:2]

    try:
        procesada = aplicar_filtro_logica(img, filtro)
    except Exception as e:
        return jsonify({'error': f'Error interno al aplicar el filtro: {str(e)}'}), 500

    # Garantizar que la imagen de salida sea siempre BGR (3 canales)
    if len(procesada.shape) == 2:
        procesada = cv2.cvtColor(procesada, cv2.COLOR_GRAY2BGR)

    ruta_procesada = os.path.join(aplicacion.config['CARPETA_SUBIDAS'], 'proc_' + nombre_archivo)
    cv2.imwrite(ruta_procesada, procesada)

    return jsonify({
        'url_original':  url_for('static', filename='processed/' + nombre_archivo),
        'url_procesada': url_for('static', filename='processed/proc_' + nombre_archivo),
        'descripcion':   obtener_descripcion(filtro),
        'filtro':        filtro,
        'ancho':         w,
        'alto':          h
    })

if __name__ == '__main__':
    aplicacion.run(debug=True)
