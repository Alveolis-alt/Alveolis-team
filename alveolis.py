
import streamlit as st
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf

# Cargar el modelo previamente entrenado
model = load_model("modelo_alveolis.h5")

# Definir las clases (enfermedades) que el modelo puede predecir
CLASES_ES = [
    "Pulmón sano", "Procesos inflamatorios", "Derrame pleural", "Neumonía (menor densidad)",
    "EPOC y similares", "Infecciones pulmonares", "Lesiones encapsuladas",
    "Alteraciones mediastínicas", "Alteraciones torácicas atípicas"
]

# Ruta donde se encuentran las radiografías (asegúrate de tener imágenes en este directorio)
radiografias_folder = 'radiografias/'  # Cambia esta ruta si es necesario

# Función para cargar y preprocesar una imagen
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_prep = preprocess_input(np.expand_dims(img_array, axis=0))
    return img_array, img_prep

# Función principal
def main():
    st.title('Diagnóstico Médico de Radiografías')

    # Obtener lista de imágenes en la carpeta de radiografías
    imagenes = [f for f in os.listdir(radiografias_folder) if f.endswith('.jpg') or f.endswith('.png')]

    # Seleccionar una imagen aleatoriamente
    imagen_seleccionada = random.choice(imagenes)

    # Obtener la ruta de la imagen seleccionada
    img_path = os.path.join(radiografias_folder, imagen_seleccionada)
    
    # Cargar y preprocesar la imagen
    img_array, img_prep = load_and_preprocess_image(img_path)
    
    # Mostrar la imagen seleccionada
    st.image(img_path, caption="Radiografía Seleccionada", use_column_width=True)

    # Realizar la predicción con el modelo
    preds = model.predict(img_prep)
    pred_idx = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0]))

    # Mostrar el diagnóstico
    diagnostico = CLASES_ES[pred_idx]
    st.write(f"**Diagnóstico**: {diagnostico}")
    st.write(f"**Confianza**: {conf:.2%}")
    
    # Mostrar un mapa de calor (opcional, solo si quieres ver qué parte de la imagen es relevante)
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer("block7a_project_bn").output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_prep)
        loss = predictions[:, pred_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Mostrar el mapa de calor
    st.subheader("Mapa de Calor")
    plt.imshow(img_array.astype("uint8"))
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis("off")
    st.pyplot(plt)

if __name__ == '__main__':
    main()
