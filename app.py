import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("😷 Real-Time Face Mask Detection")

model = load_model("mask_model.h5")

class MaskDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        face = cv2.resize(img, (224, 224))
        face = face / 255.0
        face = np.reshape(face, (1, 224, 224, 3))

        pred = model.predict(face)

        if pred[0][1] > pred[0][0]:
            label = "Mask"
            color = (0, 255, 0)
        else:
            label = "No Mask"
            color = (0, 0, 255)

        cv2.putText(img, label, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return img

webrtc_streamer(key="mask", video_transformer_factory=MaskDetector)
