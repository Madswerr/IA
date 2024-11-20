import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Función para limpiar texto
def clean_text(text):
    # Eliminar caracteres especiales innecesarios pero mantener interrogaciones y tildes
    text = re.sub(r'[^\w\s¿?áéíóúÁÉÍÓÚñÑ]', ' ', text)  
    text = re.sub(r'\s+', ' ', text)  # Reemplazar múltiples espacios con uno solo
    text = text.strip()  # Eliminar espacios al inicio y final
    text = text.lower()  # Convertir a minúsculas si lo deseas
    return text

# Cargar todas las hojas del Excel
file_path = r'D:\Ia\PreguntasChat_1.xlsx'
sheets = pd.read_excel(file_path, sheet_name=None, dtype=str)  # Leer como cadenas

# Preparar datos
preguntas_final, respuestas_final, categorias_final = [], [], []

for sheet_name, sheet_data in sheets.items():
    preguntas = sheet_data.iloc[:, 0].astype(str).tolist()
    respuestas = sheet_data.iloc[:, 1:].fillna('').astype(str)  # Asegurarse que no haya NaN
    
    # Limpiar preguntas
    preguntas = [clean_text(pregunta) for pregunta in preguntas]
    
    for i, pregunta in enumerate(preguntas):
        for respuesta in respuestas.iloc[i]:
            respuesta = clean_text(respuesta)  # Limpiar respuestas
            if respuesta:  # Verificar si la respuesta no está vacía
                preguntas_final.append(pregunta)
                respuestas_final.append(respuesta)
                categorias_final.append(sheet_name)  # Agregar la categoría

# Ajustar el LabelEncoder
le = LabelEncoder()
le.fit(respuestas_final)
y = le.transform(respuestas_final)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(preguntas_final, y, test_size=0.2, random_state=42)

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1

# Padding
max_len = 50
preguntas_seq = tokenizer.texts_to_sequences(preguntas_final)
preguntas_pad = pad_sequences(preguntas_seq, maxlen=max_len)

# One-hot encoding
num_classes = len(le.classes_)
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# Definir el modelo
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100),
    Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=256, kernel_size=5, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

# Compilar y entrenar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(preguntas_pad[:len(X_train)], y_train_onehot, epochs=50, batch_size=64, validation_data=(preguntas_pad[len(X_train):], y_test_onehot))

# Guardar el modelo
model.save('modelo_cnn.h5')

# Función para predecir la respuesta
def predecir_respuesta(pregunta_test):
    pregunta_test = clean_text(pregunta_test)  # Limpiar la pregunta
    preguntas_test_tokens = tokenizer.texts_to_sequences([pregunta_test])
    preguntas_test_matrix = pad_sequences(preguntas_test_tokens, maxlen=max_len)
    respuesta_secuencia = model.predict(preguntas_test_matrix)
    respuesta_idx = np.argmax(respuesta_secuencia, axis=-1).flatten()
    return le.inverse_transform(respuesta_idx)

# Servir el modelo con Flask
app = Flask(__name__)
CORS(app)

# Cargar el modelo
model = tf.keras.models.load_model('modelo_cnn.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    logging.info(f"Contenido de la solicitud: {data}")  # Log de todo el contenido de la solicitud
    if 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    pregunta_test = data['question']
    pregunta_test = clean_text(pregunta_test)  # Limpiar la pregunta
    logging.info(f"Pregunta recibida: {pregunta_test}")  # Log de la pregunta

    try:
        respuesta = predecir_respuesta(pregunta_test)
        if len(respuesta) == 0:
            return jsonify({'respuesta': 'No se encontró una respuesta adecuada.'}), 404
        return jsonify({'respuesta': respuesta.tolist(), 'pregunta': pregunta_test})
    except Exception as e:
        logging.error(f"Error en la predicción: {e}")
        return jsonify({'error': 'Error procesando la pregunta'}), 500

if __name__ == '__main__':
    app.run(debug=True)
