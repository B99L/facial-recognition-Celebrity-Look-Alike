import io
import os
from flask import Flask, request, send_file, render_template, redirect, url_for, send_from_directory, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import ast
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from deepface import DeepFace
import faiss
import time

app = Flask(__name__)
CORS(app)

# Angabe des Upload-Verzeichnisses
app.config['UPLOAD_FOLDER'] = 'upload_folder'

# Laden der Daten
df = pd.read_csv('my_data.csv')
df1 = pd.read_csv("list_identity_celeba.txt", sep="   ", header=0)
files = df['img_file'].values
df['facial_vec'] = df['facial_vec'].apply(lambda x: ast.literal_eval(x))
features = np.vstack(df['facial_vec'].values)


def matching(my_img):
    try:
        # Speichern des hochgeladenen Bildes im Speicher
        in_memory_file = io.BytesIO()
        my_img.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        img_array = cv2.imdecode(data, color_image_flag)

        # Generierung des Gesichtsfeatures des hochgeladenen Bildes
        my_img_feature = np.array(
            DeepFace.represent(img_array, detector_backend="mtcnn", model_name="Facenet")[0]["embedding"])

        # Erstellung des FAISS-Indexes
        index = faiss.IndexFlatL2(features.shape[1])
        index.add(features)

        # Durchführung der ähnlichsten Suche mit Faiss
        k = 1  # Anzahl der ähnlichsten Gesichter, die Sie suchen möchten
        _, index_list = index.search(np.array([my_img_feature]), k)
        similar_face = files[index_list[0][0]]
        identity_name = df1.loc[df1['image_id'] == similar_face, '  identity_name'].values[0]

        # Laden der Bilder
        similar_image_path = f"archive/img_align_celeba/{similar_face}"
        similar_image = cv2.imread(similar_image_path)

        # Skalieren der Bilder auf dieselbe Größe
        input_image = cv2.resize(img_array, (224, 224))
        similar_image = cv2.resize(similar_image, (224, 224))

        # Speichern der Bilder als PNG-Dateien
        success1 = cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "user_image.png"), input_image)
        success2 = cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "similar_image.png"), similar_image)

        # Überprüfung, ob die Bilder erfolgreich gespeichert wurden
        if not success1 or not success2:
            raise Exception("Failed to save images")

        return "user_image.png", "similar_image.png", identity_name.replace("_", " ")
    except Exception as e:
        return str(e), None, None


@app.route('/')
def home():
    # Wenn Benutzer auf die Homepage zugreift, wird er zur Upload-Seite umgeleitet
    return redirect(url_for('upload_file'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            upload_folder = app.config['UPLOAD_FOLDER']
            print(f"Upload folder: {upload_folder}")

            # Wenn das Upload-Verzeichnis nicht existiert, wird es erstellt
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            # Überprüfen, ob das Verzeichnis lesbar und beschreibbar ist
            if not os.access(upload_folder, os.R_OK | os.W_OK):
                return f"The upload folder {upload_folder} is not readable/writable", 500

            # Zugriff auf die hochgeladene Datei
            file = request.files['file']
            print(f"Received file: {file.filename}")
            file.filename = f"{file.filename}?{int(time.time())}"
            if file:
                # Führt die Gesichtszuordnung durch und gibt die Dateinamen der Benutzer- und ähnlichen Bilder zurück
                user_image_filename, similar_image_filename, identity_name  = matching(file)
                print(f"Matching results: {user_image_filename}, {similar_image_filename}, {identity_name}")
                if similar_image_filename is None:
                    return user_image_filename, 500  # Gibt die Fehlermeldung zurück
                return jsonify({"userImage": user_image_filename, "similarImage": similar_image_filename,"identityName": identity_name.replace("_", " ")})
            else:
                return "No file uploaded", 400

        except Exception as e:
            print(f"Error in upload_file: {e}")
            return jsonify({"error": str(e)}), 500

    else:
        # Bei einem GET-Anforderung wird die index.html Seite gerendert
        return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Gibt die Datei aus dem Upload-Verzeichnis zurück
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)  # Startet den Flask-Server im Debug-Modus