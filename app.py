import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

st.title("😷 Face Mask Detection App")

model = load_model("mask_model.h5")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)

    face = cv2.resize(img, (224, 224))
    face = face / 255.0
    face = np.reshape(face, (1, 224, 224, 3))

    pred = model.predict(face)

    label = "Mask 😷" if pred[0][0] > 0.5 else "No Mask ❌"

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.success(f"Prediction: {label}")
