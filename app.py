from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import numpy as np
import os
from datetime import datetime

# Importar filtros desde el paquete
from filters import (
    aplicar_blur, aplicar_gaussiano, aplicar_mediana, aplicar_bilateral,
    aplicar_canny, aplicar_sobel, aplicar_laplaciano, aplicar_prewitt, aplicar_roberts,
    aplicar_scharr, aplicar_log,
    aplicar_hough_lineas_probabilistico, aplicar_hough_lineas_busqueda, aplicar_hough_circulos,
    aplicar_erosion, aplicar_dilatacion, aplicar_apertura, aplicar_cierre, aplicar_gradiente,
    aplicar_umbral_simple, aplicar_umbral_adaptativo, aplicar_otsu
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/processed'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

filtro_actual = None
camera_active = False  # Estado de la cámara

# ========= FUNCIÓN PRINCIPAL DE DESPACHO =========
def aplicar_filtro(img, filtro):
    # Filtros Espaciales y Suavizado
    if filtro == 'blur': return aplicar_blur(img)
    elif filtro == 'gaussiano': return aplicar_gaussiano(img)
    elif filtro == 'mediana': return aplicar_mediana(img)
    elif filtro == 'bilateral': return aplicar_bilateral(img)
    
    # Detección de Bordes
    elif filtro == 'canny': return aplicar_canny(img)
    elif filtro == 'sobel': return aplicar_sobel(img)
    elif filtro == 'scharr': return aplicar_scharr(img)
    elif filtro == 'laplaciano': return aplicar_laplaciano(img)
    elif filtro == 'log': return aplicar_log(img)
    elif filtro == 'prewitt': return aplicar_prewitt(img)
    elif filtro == 'roberts': return aplicar_roberts(img)
    
    # Hough
    elif filtro == 'hough': return aplicar_hough_lineas_probabilistico(img)
    elif filtro == 'hough_standard': return aplicar_hough_lineas_busqueda(img)
    elif filtro == 'hough_circles': return aplicar_hough_circulos(img)
    
    # Morfología
    elif filtro == 'erosion': return aplicar_erosion(img)
    elif filtro == 'dilatacion': return aplicar_dilatacion(img)
    elif filtro == 'apertura': return aplicar_apertura(img)
    elif filtro == 'cierre': return aplicar_cierre(img)
    elif filtro == 'gradiente': return aplicar_gradiente(img)
    
    # Segmentación / Umbralización
    elif filtro == 'threshold': return aplicar_umbral_simple(img)
    elif filtro == 'adaptive': return aplicar_umbral_adaptativo(img)
    elif filtro == 'otsu': return aplicar_otsu(img)
    
    else:
        return img

def descripcion_filtro(filtro):
    descripciones = {
        "blur": "Suaviza la imagen promediando píxeles cercanos.",
        "gaussiano": "Reduce el ruido con un desenfoque gaussiano.",
        "mediana": "Elimina el ruido sal y pimienta conservando bordes.",
        "bilateral": "Suaviza manteniendo los bordes definidos.",
        
        "canny": "Detecta bordes precisos (Algoritmo de Canny).",
        "sobel": "Resalta cambios de intensidad (Derivadas).",
        "scharr": "Operador Scharr (Sobel optimizado).",
        "laplaciano": "Detecta bordes basado en la segunda derivada.",
        "log": "Laplaciano del Gaussiano (Suavizado + Bordes).",
        "prewitt": "Detecta bordes (similar a Sobel pero más simple).",
        "roberts": "Detecta bordes con operador de Roberts (rápido).",
        
        "hough": "Detecta líneas rectas (Probabilístico).",
        "hough_standard": "Detecta líneas rectas (Estándar, infinitas).",
        "hough_circles": "Detecta formas circulares (Transformada de Hough).",
        
        "erosion": "Reduce áreas brillantes, eliminando ruido.",
        "dilatacion": "Expande áreas brillantes.",
        "apertura": "Erosión seguida de dilatación (elimina ruido).",
        "cierre": "Dilatación seguida de erosión (cierra huecos).",
        "gradiente": "Diferencia entre dilatación y erosión (bordes).",
        
        "threshold": "Umbralización simple (B/N).",
        "adaptive": "Umbralización adaptativa (útil con iluminación variable).",
        "otsu": "Umbralización automática de Otsu."
    }
    return descripciones.get(filtro, "Filtro aplicado correctamente.")

# ========= RUTAS =========
@app.route('/', methods=['GET'])
def index():
    img_folder = os.path.join('static', 'img')
    local_images = []
    if os.path.exists(img_folder):
        local_images = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return render_template('index.html', local_images=local_images)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filtro = request.form.get('filtro')
    if not file:
        return redirect(url_for('index'))

    filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    return procesar_imagen(filepath, filename, filtro)

@app.route('/process_local', methods=['POST'])
def process_local():
    image_name = request.form.get('image_name')
    filtro = request.form.get('filtro')
    original_path = os.path.join('static', 'img', image_name)
    filename = "local_" + datetime.now().strftime("%Y%m%d%H%M%S") + "_" + image_name
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    img = cv2.imread(original_path)
    if img is None:
        return redirect(url_for('index'))
    cv2.imwrite(filepath, img)
    
    return procesar_imagen(filepath, filename, filtro)

def procesar_imagen(filepath, filename, filtro):
    img = cv2.imread(filepath)
    processed = aplicar_filtro(img, filtro)

    # Convertir a BGR si el resultado es escala de grises para mostrar canales
    if len(processed.shape) == 2:
        processed_color = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    else:
        processed_color = processed

    # Canales BGR de la Original
    b, g, r = cv2.split(img)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "blue_" + filename), cv2.merge([b,np.zeros_like(b),np.zeros_like(b)]))
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "green_" + filename), cv2.merge([np.zeros_like(g),g,np.zeros_like(g)]))
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "red_" + filename), cv2.merge([np.zeros_like(r),np.zeros_like(r),r]))

    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], "proc_" + filename)
    cv2.imwrite(processed_path, processed)

    local_imgs = []
    if os.path.exists(os.path.join('static', 'img')):
        local_imgs = [f for f in os.listdir(os.path.join('static', 'img')) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    return render_template('index.html',
                           original_image=url_for('static', filename='processed/' + filename),
                           processed_image=url_for('static', filename='processed/proc_' + filename),
                           blue_image=url_for('static', filename='processed/blue_' + filename),
                           green_image=url_for('static', filename='processed/green_' + filename),
                           red_image=url_for('static', filename='processed/red_' + filename),
                           filtro=filtro,
                           descripcion=descripcion_filtro(filtro),
                           local_images=local_imgs)

# ========= CÁMARA =========
def gen_frames():
    camera = cv2.VideoCapture(0)
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

@app.route('/iniciar_camara', methods=['POST'])
def iniciar_camara():
    global camera_active, filtro_actual
    filtro_actual = request.form.get('filtro')
    camera_active = True
    local_imgs = [f for f in os.listdir(os.path.join('static', 'img')) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(os.path.join('static', 'img')) else []
    return render_template('index.html', camara=True, filtro=filtro_actual, local_images=local_imgs)

@app.route('/detener_camara', methods=['POST'])
def detener_camara():
    global camera_active
    camera_active = False
    return redirect(url_for('index'))

@app.route('/aplicar_filtro_camara', methods=['POST'])
def aplicar_filtro_camara():
    global filtro_actual
    filtro_actual = request.form.get('filtro')
    local_imgs = [f for f in os.listdir(os.path.join('static', 'img')) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(os.path.join('static', 'img')) else []
    return render_template('index.html', camara=True, filtro=filtro_actual, local_images=local_imgs)

if __name__ == '__main__':
    app.run(debug=True)
