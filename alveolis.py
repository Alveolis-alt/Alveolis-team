import os
import random
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model

# Cargar el modelo previamente entrenado
model = load_model("modelo_alveolis.h5")

# Definir las clases (enfermedades) que el modelo puede predecir
CLASES_ES = [
    "Pulmón sano", "Procesos inflamatorios", "Derrame pleural", "Neumonía (menor densidad)",
    "EPOC y similares", "Infecciones pulmonares", "Lesiones encapsuladas",
    "Alteraciones mediastínicas", "Alteraciones torácicas atípicas"
]

# Ruta donde se encuentran las radiografías
radiografias_folder = 'radiografias/'  # Cambia esto con tu ruta real

# Función para cargar y preprocesar una imagen
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_prep = preprocess_input(np.expand_dims(img_array, axis=0))
    return img_array, img_prep

# Función principal
def main():
    st.title('Diagnóstico Médico de Radiografías')

    # Obtener lista de todas las imágenes en la carpeta y subcarpetas
    imagenes = []
    for root, dirs, files in os.walk(radiografias_folder):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):  # Asegúrate de que las imágenes sean .jpg o .png
                imagenes.append(os.path.join(root, file))

    # Verificar si hay imágenes disponibles
    if imagenes:
        # Seleccionar una imagen aleatoriamente
        imagen_seleccionada = random.choice(imagenes)

        # Cargar y preprocesar la imagen
        img_array, img_prep = load_and_preprocess_image(imagen_seleccionada)
        
        # Mostrar la imagen seleccionada
        st.image(imagen_seleccionada, caption="Radiografía Seleccionada", use_column_width=True)

        # Realizar la predicción con el modelo
        preds = model.predict(img_prep)
        pred_idx = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]))

        # Mostrar el diagnóstico
        diagnostico = CLASES_ES[pred_idx]
        st.write(f"**Diagnóstico**: {diagnostico}")
        st.write(f"**Confianza**: {conf:.2%}")

    else:
        st.write("No se encontraron imágenes en la carpeta 'radiografias'.")

# Ejecutar la aplicación
if __name__ == '__main__':
    main()
