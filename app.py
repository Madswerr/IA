import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS  # Importar Flask-CORS
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Cargar el modelo y los datos
model = load_model('modelo_cnn_hardware.h5')
file_path = r'D:\Ia\PreguntasChat_1.xlsx'
excel_data = pd.ExcelFile(file_path)
software_sheet = excel_data.parse('HARDWARE')

# Preparar los datos
preguntas = software_sheet.iloc[:, 0].astype(str).tolist()
respuestas = software_sheet.iloc[:, 1:]

# Aplanar las respuestas
preguntas_final = []
respuestas_final = []
for i, pregunta in enumerate(preguntas):
    for respuesta in respuestas.iloc[i]:
        if pd.notna(respuesta):
            preguntas_final.append(pregunta)
            respuestas_final.append(respuesta)

# Ajustar el LabelEncoder
le = LabelEncoder()
le.fit(respuestas_final)

# Configuración del Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preguntas_final)
max_len = 50  # Ajusta según tus datos

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    pregunta_test = data['question']
    
    # Predecir la respuesta
    preguntas_test_tokens = tokenizer.texts_to_sequences([pregunta_test])
    preguntas_test_matrix = pad_sequences(preguntas_test_tokens, maxlen=max_len)
    respuesta_secuencia = model.predict(preguntas_test_matrix)
    respuesta_idx = np.argmax(respuesta_secuencia, axis=-1).flatten()
    respuesta = le.inverse_transform(respuesta_idx)

    return jsonify({'respuesta': respuesta.tolist()})

if __name__ == '__main__':
    app.run(debug=True)