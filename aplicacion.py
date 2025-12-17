from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import cv2
import numpy as np
import os
from datetime import datetime
import time

# Importar filtros desde el paquete
from filtros import (
    aplicar_blur, aplicar_gaussiano, aplicar_mediana, aplicar_bilateral,
    aplicar_canny, aplicar_sobel, aplicar_laplaciano, aplicar_prewitt, aplicar_roberts,
    aplicar_scharr, aplicar_log,
    aplicar_hough_lineas_probabilistico, aplicar_hough_lineas_busqueda, aplicar_hough_circulos,
    aplicar_erosion, aplicar_dilatacion, aplicar_apertura, aplicar_cierre, aplicar_gradiente,
    aplicar_umbral_simple, aplicar_umbral_adaptativo, aplicar_otsu,
    aplicar_sharpen, aplicar_high_pass, aplicar_emboss, aplicar_unsharp_mask,
    aplicar_dft_magnitude,
    aplicar_escala_grises, aplicar_hsv, aplicar_ecualizacion_hist, aplicar_brillo_contraste, aplicar_pseudocolor,
    aplicar_rotacion_90, aplicar_rotacion_180, aplicar_volteo_horizontal, aplicar_volteo_vertical,
    aplicar_kmeans,
    GIMP_FILTERS, EXT_FILTERS
)

aplicacion = Flask(__name__)
aplicacion.config['CARPETA_SUBIDAS'] = 'static/processed'

if not os.path.exists(aplicacion.config['CARPETA_SUBIDAS']):
    os.makedirs(aplicacion.config['CARPETA_SUBIDAS'])

filtro_actual = None
camara_activa = False

# ========= FUNCIÓN PRINCIPAL DE DESPACHO =========
def aplicar_filtro_logica(img, filtro):
    # I.1 Suavizado
    if filtro == 'blur': return aplicar_blur(img)
    elif filtro == 'gaussiano': return aplicar_gaussiano(img)
    elif filtro == 'mediana': return aplicar_mediana(img)
    elif filtro == 'bilateral': return aplicar_bilateral(img)
    
    # I.2 Realce (Enhancement)
    elif filtro == 'sharpen': return aplicar_sharpen(img)
    elif filtro == 'high_pass': return aplicar_high_pass(img)
    elif filtro == 'high_boost': return aplicar_sharpen(img) 
    elif filtro == 'unsharp': return aplicar_unsharp_mask(img)
    
    # I.3 Detección de Bordes (Edges)
    elif filtro == 'canny': return aplicar_canny(img)
    elif filtro == 'sobel': return aplicar_sobel(img)
    elif filtro == 'scharr': return aplicar_scharr(img)
    elif filtro == 'laplaciano': return aplicar_laplaciano(img)
    elif filtro == 'log': return aplicar_log(img)
    elif filtro == 'prewitt': return aplicar_prewitt(img)
    elif filtro == 'roberts': return aplicar_roberts(img)
    
    # I.4 Convolución
    elif filtro == 'emboss': return aplicar_emboss(img)

    # I.5 Morfología
    elif filtro == 'erosion': return aplicar_erosion(img)
    elif filtro == 'dilatacion': return aplicar_dilatacion(img)
    elif filtro == 'apertura': return aplicar_apertura(img)
    elif filtro == 'cierre': return aplicar_cierre(img)
    elif filtro == 'gradiente': return aplicar_gradiente(img)
    
    # II. Transformadas (Frequency)
    elif filtro == 'dft': return aplicar_dft_magnitude(img)
    
    # III. Transformaciones de Color
    elif filtro == 'gray': return aplicar_escala_grises(img)
    elif filtro == 'hsv': return aplicar_hsv(img)
    elif filtro == 'equalize': return aplicar_ecualizacion_hist(img)
    elif filtro == 'bright_contrast': return aplicar_brillo_contraste(img)
    elif filtro == 'pseudocolor': return aplicar_pseudocolor(img)

    # IV. Geométricas
    elif filtro == 'rot_90': return aplicar_rotacion_90(img)
    elif filtro == 'rot_180': return aplicar_rotacion_180(img)
    elif filtro == 'flip_h': return aplicar_volteo_horizontal(img)
    elif filtro == 'flip_v': return aplicar_volteo_vertical(img)

    # V. Segmentación
    elif filtro == 'threshold': return aplicar_umbral_simple(img)
    elif filtro == 'adaptive': return aplicar_umbral_adaptativo(img)
    elif filtro == 'otsu': return aplicar_otsu(img)
    elif filtro == 'kmeans': return aplicar_kmeans(img)

    # Hough
    elif filtro == 'hough': return aplicar_hough_lineas_probabilistico(img)
    elif filtro == 'hough_standard': return aplicar_hough_lineas_busqueda(img)
    elif filtro == 'hough_circles': return aplicar_hough_circulos(img)
    
    # GIMP Editor Filters (Legacy/Mapped)
    elif filtro in GIMP_FILTERS:
        return GIMP_FILTERS[filtro](img)
    
    # Extended Filters
    elif filtro in EXT_FILTERS:
        return EXT_FILTERS[filtro](img)
        
    else:
        return img

def obtener_descripcion(filtro):
    if filtro.startswith('g_'):
        return "Efecto de Edición GIMP / Ajuste Visual Avanzado."
    if filtro in EXT_FILTERS:
        return "Filtro Extendido / Efecto Especial."

    descripciones = {
        "blur": "Suaviza promediando píxeles vecinos.",
        "gaussiano": "Suaviza con distribución gaussiana (reduce ruido Gaussiano).",
        "mediana": "Elimina ruido 'sal y pimienta' conservando bordes.",
        "bilateral": "Suaviza preservando bordes (piel suave).",
        "sharpen": "Aumenta la nitidez de la imagen.",
        "high_pass": "Paso Alto: Enfatiza bordes y detalles finos.",
        "unsharp": "Máscara de Enfoque: Realce mediante sustracción de borrosidad.",
        "canny": "Detección de bordes multi-etapa (Canny).",
        "sobel": "Detección de bordes (Derivadas de Sobel).",
        "scharr": "Operador Scharr (Sobel optimizado).",
        "laplaciano": "Detección de bordes omnidireccional (Laplaciano).",
        "log": "Laplaciano del Gaussiano (LoG).",
        "prewitt": "Similar a Sobel con máscaras diferentes.",
        "roberts": "Operador de Roberts (gradiente simple).",
        "emboss": "Crea efecto tridimensional de relieve.",
        "erosion": "Reduce regiones brillantes (erosiona).",
        "dilatacion": "Expande regiones brillantes (dilata).",
        "apertura": "Erosión seguida de dilatación (elimina puntos blancos).",
        "cierre": "Dilatación seguida de erosión (cierra huecos negros).",
        "gradiente": "Detección de bordes morfológicos.",
        "dft": "Espectro de Magnitud de Fourier (Frecuencias).",
        "gray": "Conversión a Escala de Grises (Luminosidad).",
        "hsv": "Espacio de Color Tono-Saturación-Valor.",
        "equalize": "Ecualización de Histograma (Mejora contraste global).",
        "bright_contrast": "Ajuste lineal de Brillo (+30) y Contraste (1.2x).",
        "pseudocolor": "Mapa de color JET aplicado a escala de grises.",
        "rot_90": "Rotación 90° sentido horario.",
        "rot_180": "Rotación 180°.",
        "flip_h": "Volteo Horizontal (Espejo).",
        "flip_v": "Volteo Vertical.",
        "threshold": "Umbral Simple (Binario).",
        "adaptive": "Umbral Adaptativo (Variable localmente).",
        "otsu": "Umbral de Otsu (Automático).",
        "kmeans": "Segmentación por Clustering (K-means, K=8).",
        "hough": "Líneas Probabilísticas.",
        "hough_standard": "Líneas Estándar (Infinitas).",
        "hough_circles": "Detección de Círculos.",
    }
    return descripciones.get(filtro, "Filtro aplicado correctamente.")

# ========= RUTAS =========
@aplicacion.route('/', methods=['GET'])
def index():
    carpeta_img = os.path.join('static', 'img')
    imagenes_locales = []
    if os.path.exists(carpeta_img):
        imagenes_locales = [f for f in os.listdir(carpeta_img) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(carpeta_img, f))]
        
    carpeta_finca = os.path.join('static', 'img', 'finca')
    imagenes_finca = []
    if os.path.exists(carpeta_finca):
        imagenes_finca = [f for f in os.listdir(carpeta_finca) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        try:
            imagenes_finca.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x)
        except:
            pass
            
    return render_template('index.html', imagenes_locales=imagenes_locales, imagenes_finca=imagenes_finca, imagen_original=None)

@aplicacion.route('/subir', methods=['POST'])
def subir():
    archivo = request.files['file']
    filtro = request.form.get('filtro')
    if not archivo:
        return jsonify({'error': 'No file uploaded'})

    nombre_archivo = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + archivo.filename
    ruta_archivo = os.path.join(aplicacion.config['CARPETA_SUBIDAS'], nombre_archivo)
    archivo.save(ruta_archivo)

    return procesar_imagen_api(ruta_archivo, nombre_archivo, filtro)

@aplicacion.route('/api_procesamiento_local', methods=['POST'])
def api_procesamiento_local():
    nombre_imagen = request.form.get('image_name')
    filtro = request.form.get('filtro')
    origen = request.form.get('origin')
    
    if origen == 'finca':
        ruta_original = os.path.join(aplicacion.root_path, 'static', 'img', 'finca', nombre_imagen)
    elif origen == 'processed':
        ruta_original = os.path.join(aplicacion.root_path, 'static', 'processed', nombre_imagen)
    elif origen == 'web_upload':
        ruta_original = os.path.join(aplicacion.config['CARPETA_SUBIDAS'], nombre_imagen)
    else:
        ruta_original = os.path.join(aplicacion.root_path, 'static', 'img', nombre_imagen)
    
    if not os.path.exists(ruta_original):
        return jsonify({'error': f'Image not found: {ruta_original}'}), 404

    nombre_archivo = "local_" + datetime.now().strftime("%Y%m%d%H%M%S") + "_" + nombre_imagen
    ruta_archivo = os.path.join(aplicacion.config['CARPETA_SUBIDAS'], nombre_archivo)
    
    img = cv2.imread(ruta_original)
    if img is None:
        return jsonify({'error': 'Failed to read image'}), 500
        
    cv2.imwrite(ruta_archivo, img)
    
    return procesar_imagen_api(ruta_archivo, nombre_archivo, filtro)

def procesar_imagen_api(ruta_archivo, nombre_archivo, filtro):
    img = cv2.imread(ruta_archivo)
    procesada = aplicar_filtro_logica(img, filtro)

    if len(procesada.shape) == 2:
        procesada_color = cv2.cvtColor(procesada, cv2.COLOR_GRAY2BGR)
    else:
        procesada_color = procesada

    b, g, r = cv2.split(procesada_color)
    
    cv2.imwrite(os.path.join(aplicacion.config['CARPETA_SUBIDAS'], "blue_" + nombre_archivo), cv2.merge([b,np.zeros_like(b),np.zeros_like(b)]))
    cv2.imwrite(os.path.join(aplicacion.config['CARPETA_SUBIDAS'], "green_" + nombre_archivo), cv2.merge([np.zeros_like(g),g,np.zeros_like(g)]))
    cv2.imwrite(os.path.join(aplicacion.config['CARPETA_SUBIDAS'], "red_" + nombre_archivo), cv2.merge([np.zeros_like(r),np.zeros_like(r),r]))

    ruta_procesada = os.path.join(aplicacion.config['CARPETA_SUBIDAS'], "proc_" + nombre_archivo)
    cv2.imwrite(ruta_procesada, procesada_color)
    
    return jsonify({
        'url_original': url_for('static', filename='processed/' + nombre_archivo),
        'url_procesada': url_for('static', filename='processed/proc_' + nombre_archivo),
        'url_azul': url_for('static', filename='processed/blue_' + nombre_archivo),
        'url_verde': url_for('static', filename='processed/green_' + nombre_archivo),
        'url_roja': url_for('static', filename='processed/red_' + nombre_archivo),
        'descripcion': obtener_descripcion(filtro),
        'filtro': filtro,
        'nombre_archivo_resultado': "proc_" + nombre_archivo
    })

# ========= CÁMARA =========
def generar_cuadros():
    camara = cv2.VideoCapture(0)
    time.sleep(0.5)
    while camara_activa:
        success, frame = camara.read()
        if not success:
            break
        if filtro_actual:
            frame = aplicar_filtro_logica(frame, filtro_actual)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camara.release()

@aplicacion.route('/video_en_vivo')
def video_en_vivo():
    return Response(generar_cuadros(), mimetype='multipart/x-mixed-replace; boundary=frame')

@aplicacion.route('/api_control_camara', methods=['POST'])
def api_control_camara():
    global camara_activa, filtro_actual
    accion = request.json.get('action')
    valor = request.json.get('value')
    
    if accion == 'start':
        camara_activa = True
        return jsonify({'status': 'started', 'message': 'Cámara iniciada'})
    elif accion == 'stop':
        camara_activa = False
        return jsonify({'status': 'stopped', 'message': 'Cámara detenida'})
    elif accion == 'filter':
        filtro_actual = valor
        return jsonify({'status': 'updated', 'message': f'Filtro {valor} aplicado'})
    
    return jsonify({'error': 'Invalid action'}), 400

if __name__ == '__main__':
    aplicacion.run(debug=True)
