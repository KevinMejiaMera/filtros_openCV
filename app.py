from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/processed'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

filtro_actual = None
camera_active = False  # Estado de la cámara

# ========= FUNCIONES DE PROCESAMIENTO =========
def aplicar_filtro(img, filtro):
    if filtro == 'blur':
        return cv2.blur(img, (9, 9))
    elif filtro == 'gaussiano':
        return cv2.GaussianBlur(img, (9, 9), 0)
    elif filtro == 'mediana':
        return cv2.medianBlur(img, 9)
    elif filtro == 'bilateral':
        return cv2.bilateralFilter(img, 9, 75, 75)
    elif filtro == 'canny':
        return cv2.Canny(img, 100, 200)
    elif filtro == 'sobel':
        return cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)
    elif filtro == 'laplaciano':
        return cv2.Laplacian(img, cv2.CV_64F)
    elif filtro == 'erosion':
        kernel = np.ones((5,5), np.uint8)
        return cv2.erode(img, kernel, iterations=1)
    elif filtro == 'dilatacion':
        kernel = np.ones((5,5), np.uint8)
        return cv2.dilate(img, kernel, iterations=1)
    elif filtro == 'apertura':
        kernel = np.ones((5,5), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif filtro == 'cierre':
        kernel = np.ones((5,5), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    elif filtro == 'gradiente':
        kernel = np.ones((5,5), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    else:
        return img

def descripcion_filtro(filtro):
    descripciones = {
        "blur": "Suaviza la imagen promediando píxeles cercanos.",
        "gaussiano": "Reduce el ruido con un desenfoque gaussiano.",
        "mediana": "Elimina el ruido sal y pimienta conservando bordes.",
        "bilateral": "Suaviza manteniendo los bordes definidos.",
        "canny": "Detecta bordes en la imagen.",
        "sobel": "Resalta los cambios de intensidad en la imagen.",
        "laplaciano": "Detecta bordes calculando la segunda derivada.",
        "erosion": "Reduce áreas brillantes, eliminando ruido.",
        "dilatacion": "Expande áreas brillantes en la imagen.",
        "apertura": "Elimina ruido pequeño (erosión + dilatación).",
        "cierre": "Rellena huecos pequeños (dilatación + erosión).",
        "gradiente": "Diferencia entre dilatación y erosión, resalta bordes."
    }
    return descripciones.get(filtro, "Selecciona un filtro para ver su descripción.")

# ========= SUBIR Y PROCESAR IMAGEN =========
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filtro = request.form.get('filtro')
    if not file:
        return redirect(url_for('index'))

    filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    processed = aplicar_filtro(img, filtro)

    # Canales BGR
    b, g, r = cv2.split(img)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "blue_" + filename), cv2.merge([b,np.zeros_like(b),np.zeros_like(b)]))
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "green_" + filename), cv2.merge([np.zeros_like(g),g,np.zeros_like(g)]))
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "red_" + filename), cv2.merge([np.zeros_like(r),np.zeros_like(r),r]))

    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], "proc_" + filename)
    cv2.imwrite(processed_path, processed)

    return render_template('index.html',
                           original_image=url_for('static', filename='processed/' + filename),
                           processed_image=url_for('static', filename='processed/proc_' + filename),
                           blue_image=url_for('static', filename='processed/blue_' + filename),
                           green_image=url_for('static', filename='processed/green_' + filename),
                           red_image=url_for('static', filename='processed/red_' + filename),
                           filtro=filtro,
                           descripcion=descripcion_filtro(filtro))

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
    return render_template('index.html', camara=True, filtro=filtro_actual)

@app.route('/detener_camara', methods=['POST'])
def detener_camara():
    global camera_active
    camera_active = False
    return redirect(url_for('index'))

@app.route('/aplicar_filtro_camara', methods=['POST'])
def aplicar_filtro_camara():
    global filtro_actual
    filtro_actual = request.form.get('filtro')
    return render_template('index.html', camara=True, filtro=filtro_actual)

if __name__ == '__main__':
    app.run(debug=True)
