from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import cv2
import numpy as np
import os
from datetime import datetime
import time

# Importar filtros desde el paquete
from filters import (
    # Existing
    aplicar_blur, aplicar_gaussiano, aplicar_mediana, aplicar_bilateral,
    aplicar_canny, aplicar_sobel, aplicar_laplaciano, aplicar_prewitt, aplicar_roberts,
    aplicar_scharr, aplicar_log,
    aplicar_hough_lineas_probabilistico, aplicar_hough_lineas_busqueda, aplicar_hough_circulos,
    aplicar_erosion, aplicar_dilatacion, aplicar_apertura, aplicar_cierre, aplicar_gradiente,
    aplicar_umbral_simple, aplicar_umbral_adaptativo, aplicar_otsu,
    # NEW
    aplicar_sharpen, aplicar_high_pass, aplicar_emboss, aplicar_unsharp_mask,
    aplicar_dft_magnitude,
    aplicar_escala_grises, aplicar_hsv, aplicar_ecualizacion_hist, aplicar_brillo_contraste, aplicar_pseudocolor,
    aplicar_rotacion_90, aplicar_rotacion_180, aplicar_volteo_horizontal, aplicar_volteo_vertical,
    aplicar_kmeans,
    GIMP_FILTERS, EXT_FILTERS
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/processed'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

filtro_actual = None
camera_active = False

# ========= FUNCIÓN PRINCIPAL DE DESPACHO =========
def aplicar_filtro(img, filtro):
    # I.1 Suavizado
    if filtro == 'blur': return aplicar_blur(img)
    elif filtro == 'gaussiano': return aplicar_gaussiano(img)
    elif filtro == 'mediana': return aplicar_mediana(img)
    elif filtro == 'bilateral': return aplicar_bilateral(img)
    
    # I.2 Realce (Enhancement)
    elif filtro == 'sharpen': return aplicar_sharpen(img)
    elif filtro == 'high_pass': return aplicar_high_pass(img)
    elif filtro == 'high_boost': return aplicar_sharpen(img) # Using sharpen as simple placeholder or combine
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
    
    # GIMP Editor Filters (Legacy)
    elif filtro in GIMP_FILTERS:
        return GIMP_FILTERS[filtro](img)
    
    # Extended Filters
    elif filtro in EXT_FILTERS:
        return EXT_FILTERS[filtro](img)
        
    else:
        return img

def descripcion_filtro(filtro):
    # Check GIMP first for generic description
    if filtro.startswith('g_'):
        return "Efecto de Edición GIMP / Ajuste Visual Avanzado."
    if filtro in EXT_FILTERS:
        return "Filtro Extendido / Efecto Especial."

    descripciones = {
        # ... (Existing)
        # I.1 Suavizado
        "blur": "Suaviza promediando píxeles vecinos.",
        "gaussiano": "Suaviza con distribución gaussiana (reduce ruido Gaussiano).",
        "mediana": "Elimina ruido 'sal y pimienta' conservando bordes.",
        "bilateral": "Suaviza preservando bordes (piel suave).",
        # I.2 Realce
        "sharpen": "Aumenta la nitidez de la imagen.",
        "high_pass": "Paso Alto: Enfatiza bordes y detalles finos.",
        "unsharp": "Máscara de Enfoque: Realce mediante sustracción de borrosidad.",
        # I.3 Bordes
        "canny": "Detección de bordes multi-etapa (Canny).",
        "sobel": "Detección de bordes (Derivadas de Sobel).",
        "scharr": "Operador Scharr (Sobel optimizado).",
        "laplaciano": "Detección de bordes omnidireccional (Laplaciano).",
        "log": "Laplaciano del Gaussiano (LoG).",
        "prewitt": "Similar a Sobel con máscaras diferentes.",
        "roberts": "Operador de Roberts (gradiente simple).",
        # I.4 Convolución
        "emboss": "Crea efecto tridimensional de relieve.",
        # I.5 Morfológicos
        "erosion": "Reduce regiones brillantes (erosiona).",
        "dilatacion": "Expande regiones brillantes (dilata).",
        "apertura": "Erosión seguida de dilatación (elimina puntos blancos).",
        "cierre": "Dilatación seguida de erosión (cierra huecos negros).",
        "gradiente": "Detección de bordes morfológicos.",
        # II. Transformadas
        "dft": "Espectro de Magnitud de Fourier (Frecuencias).",
        # III. Color
        "gray": "Conversión a Escala de Grises (Luminosidad).",
        "hsv": "Espacio de Color Tono-Saturación-Valor.",
        "equalize": "Ecualización de Histograma (Mejora contraste global).",
        "bright_contrast": "Ajuste lineal de Brillo (+30) y Contraste (1.2x).",
        "pseudocolor": "Mapa de color JET aplicado a escala de grises.",
        # IV. Geometricas
        "rot_90": "Rotación 90° sentido horario.",
        "rot_180": "Rotación 180°.",
        "flip_h": "Volteo Horizontal (Espejo).",
        "flip_v": "Volteo Vertical.",
        # V. Segmentacion
        "threshold": "Umbral Simple (Binario).",
        "adaptive": "Umbral Adaptativo (Variable localmente).",
        "otsu": "Umbral de Otsu (Automático).",
        "kmeans": "Segmentación por Clustering (K-means, K=8).",
        # Hough
        "hough": "Líneas Probabilísticas.",
        "hough_standard": "Líneas Estándar (Infinitas).",
        "hough_circles": "Detección de Círculos.",
    }
    return descripciones.get(filtro, "Filtro aplicado correctamente.")

# ========= RUTAS =========
@app.route('/', methods=['GET'])
def index():
    # Root static/img images
    img_folder = os.path.join('static', 'img')
    local_images = []
    if os.path.exists(img_folder):
        local_images = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(img_folder, f))]
        
    # Finca images
    finca_folder = os.path.join('static', 'img', 'finca')
    finca_images = []
    if os.path.exists(finca_folder):
        finca_images = [f for f in os.listdir(finca_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # Sort numerically if possible (imagen1, imagen2...)
        try:
            finca_images.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x)
        except:
            pass
            
    return render_template('index.html', local_images=local_images, finca_images=finca_images, original_image=None)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filtro = request.form.get('filtro')
    if not file:
        return jsonify({'error': 'No file uploaded'})

    filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    return procesar_imagen_api(filepath, filename, filtro)

@app.route('/process_local_api', methods=['POST'])
def process_local_api():
    image_name = request.form.get('image_name')
    filtro = request.form.get('filtro')
    origin = request.form.get('origin') # 'finca' or 'root'
    
    # Explicit searching based on origin
    if origin == 'finca':
        original_path = os.path.join(app.root_path, 'static', 'img', 'finca', image_name)
    elif origin == 'processed':
        original_path = os.path.join(app.root_path, 'static', 'processed', image_name)
    elif origin == 'web_upload':
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    else:
        original_path = os.path.join(app.root_path, 'static', 'img', image_name)
    
    if not os.path.exists(original_path):
        return jsonify({'error': f'Image not found: {original_path}'}), 404

    filename = "local_" + datetime.now().strftime("%Y%m%d%H%M%S") + "_" + image_name
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    img = cv2.imread(original_path)
    if img is None:
        return jsonify({'error': 'Failed to read image'}), 500
        
    cv2.imwrite(filepath, img)
    
    return procesar_imagen_api(filepath, filename, filtro)

def procesar_imagen_api(filepath, filename, filtro):
    img = cv2.imread(filepath)
    processed = aplicar_filtro(img, filtro)

    # Convertir a BGR si el resultado es escala de grises para mostrar canales
    if len(processed.shape) == 2:
        processed_color = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    else:
        processed_color = processed

    # Canales BGR de la Original (Normally processed, but let's show breakdown of result or refined original? User asked for Blue, Green, Red of... typically the ORIGINAL or RESULT? Usually Original to show components, but let's do Processed to show effect on channels? Let's stick to simple logic: breakdown the OUTPUT or INPUT. Code previously broke down INPUT. Let's break down PROCESSED for "Analysis")
    # Actually, previous code broke down 'img' which was loaded from file. Let's break down the PROCESSED image to show filter effect on channels if desired, OR keep original. I will stick to breaking down the RESULT for "Analysis".
    b, g, r = cv2.split(processed_color)
    
    # Save channels
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "blue_" + filename), cv2.merge([b,np.zeros_like(b),np.zeros_like(b)]))
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "green_" + filename), cv2.merge([np.zeros_like(g),g,np.zeros_like(g)]))
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "red_" + filename), cv2.merge([np.zeros_like(r),np.zeros_like(r),r]))

    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], "proc_" + filename)
    cv2.imwrite(processed_path, processed_color)
    
    # Save the original temp copy as well to be consistent with 'original_url' expectation
    # Ideally original_url points to the input file 'filepath'.
    # For 'process_local_api', 'filepath' is already in 'static/processed'.
    # We just need to make sure we serve it correctly.
    
    # NOTE: filename here is "local_timestamp_original.jpg".
    # processed filename is "proc_local_timestamp_original.jpg".

    # Return JSON
    return jsonify({
        'original_url': url_for('static', filename='processed/' + filename),
        'processed_url': url_for('static', filename='processed/proc_' + filename),
        'blue_url': url_for('static', filename='processed/blue_' + filename),
        'green_url': url_for('static', filename='processed/green_' + filename),
        'red_url': url_for('static', filename='processed/red_' + filename),
        'description': descripcion_filtro(filtro),
        'filtro': filtro,
        'result_filename': "proc_" + filename # The filename of the result, relative to 'static/processed'
    })

# ========= CÁMARA =========
def gen_frames():
    camera = cv2.VideoCapture(0)
    # Warmup
    time.sleep(0.5)
    while camera_active:
        success, frame = camera.read()
        if not success:
            break
        if filtro_actual:
            frame = aplicar_filtro(frame, filtro_actual)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api_camera_control', methods=['POST'])
def api_camera_control():
    global camera_active, filtro_actual
    action = request.json.get('action')
    val = request.json.get('value')
    
    if action == 'start':
        camera_active = True
        return jsonify({'status': 'started', 'message': 'Cámara iniciada'})
    elif action == 'stop':
        camera_active = False
        return jsonify({'status': 'stopped', 'message': 'Cámara detenida'})
    elif action == 'filter':
        filtro_actual = val
        return jsonify({'status': 'updated', 'message': f'Filtro {val} aplicado'})
    
    return jsonify({'error': 'Invalid action'}), 400

if __name__ == '__main__':
    app.run(debug=True)
