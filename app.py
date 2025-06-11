import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown
import os
from PIL import Image

# Tải model từ Google Drive nếu chưa có
model_url = 'https://drive.google.com/uc?id=1ELXWxH1IJ525FM1F4niKr6R1x0pR2pAn'
model_path = 'fer_emotion_model.h5'

if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        gdown.download(model_url, model_path, quiet=False)

# Load model
model = load_model(model_path)

# Nhãn tương ứng với output
class_names = ['negative', 'neutral', 'positive']

# Giao diện
st.title("😊 Emotion Detection from Image")

uploaded_files = st.file_uploader("Upload one or more images", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption="Uploaded Image", width=200)
        img = Image.open(uploaded_file).convert('L').resize((48, 48))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, 48, 48, 1)

        # Dự đoán
        prediction = model.predict(img_array)
        predicted_label = class_names[np.argmax(prediction)]
        st.success(f"🧠 Predicted Emotion: **{predicted_label}**")
