import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.title("😷 Face Mask Detection")

model = load_model("mask_model.h5")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = face / 255.0
        face = np.reshape(face, (1, 224, 224, 3))

        pred = model.predict(face)
        label = "Mask" if pred[0][1] > pred[0][0] else "No Mask"

        color = (0,255,0) if label=="Mask" else (0,0,255)

        cv2.putText(frame, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

    FRAME_WINDOW.image(frame)

camera.release()