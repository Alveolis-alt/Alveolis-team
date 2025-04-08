import os
import random
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

model = load_model("modelo_alveolis.h5")

CLASES_ES = [
    "Pulmón sano", "Procesos inflamatorios", "Derrame pleural", "Neumonía (menor densidad)",
    "EPOC y similares", "Infecciones pulmonares", "Lesiones encapsuladas",
    "Alteraciones mediastínicas", "Alteraciones torácicas atípicas"
]

def load_and_preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_prep = preprocess_input(np.expand_dims(img_array, axis=0))
    return img_array, img_prep

def mostrar_imagen_con_diagnostico(imagen_path, diagnostico, conf):
    img = cv2.imread(imagen_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(f"Radiografía: {diagnostico} - Confianza: {conf:.2%}")
    plt.axis('off')
    plt.show()

def generar_pdf(imagen_path, diagnostico, conf, sintomas, output_pdf_path):
    c = canvas.Canvas(output_pdf_path, pagesize=letter)
    c.drawString(72, 750, f"Informe Médico")
    c.drawString(72, 735, f"Diagnóstico: {diagnostico}")
    c.drawString(72, 720, f"Confianza: {conf:.2%}")
    c.drawString(72, 705, f"Síntomas del paciente: {sintomas}")
    c.drawString(72, 690, f"Radiografía: {imagen_path}")
    c.drawImage(imagen_path, 72, 400, width=400, height=300)
    c.save()

def main():
    st.title('Sistema de Diagnóstico Médico de Radiografías')

    sintomas = st.text_area("Ingrese los síntomas del paciente", "")
    
    if sintomas:
        uploaded_file = st.file_uploader("Suba la radiografía del paciente", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            img = image.load_img(uploaded_file)
            img_array, img_prep = load_and_preprocess_image(img)
            st.image(uploaded_file, caption="Radiografía Cargada", use_column_width=True)

            preds = model.predict(img_prep)
            pred_idx = int(np.argmax(preds[0]))
            conf = float(np.max(preds[0]))

            diagnostico = CLASES_ES[pred_idx]

            st.write(f"**Diagnóstico**: {diagnostico}")
            st.write(f"**Confianza**: {conf:.2%}")

            mostrar_imagen_con_diagnostico(uploaded_file, diagnostico, conf)

            output_pdf_path = "/mnt/data/diagnostico_radiografia.pdf"
            generar_pdf(uploaded_file, diagnostico, conf, sintomas, output_pdf_path)

            st.download_button("Descargar informe médico en PDF", data=open(output_pdf_path, "rb").read(), file_name="diagnostico_radiografia.pdf")
        else:
            st.write("Por favor, cargue una radiografía.")
    else:
        st.write("Por favor, ingrese los síntomas del paciente.")

if __name__ == '__main__':
    main()
