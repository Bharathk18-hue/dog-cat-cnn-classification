"""Simple Streamlit app: upload an image, get Dog or Cat. Run: streamlit run app.py"""
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

IMG_SIZE = 128
MODEL_PATH = Path(__file__).resolve().parent / "dog_cat_cnn.keras"

st.title("Dog vs Cat Classifier")
st.write("Upload a photo and the CNN will predict whether it shows a dog or a cat.")


@st.cache_resource
def get_model():
    return tf.keras.models.load_model(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Model file 'dog_cat_cnn.keras' not found. Train the model first (see README.md).")
    st.stop()

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("This file could not be read as an image.")
        st.stop()
    st.image(image, caption="Uploaded image", width=300)
    arr = np.array(image.resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8) / 255.0
    p_dog = float(get_model().predict(arr[np.newaxis], verbose=0)[0][0])
    label = "Dog" if p_dog > 0.5 else "Cat"
    confidence = p_dog if p_dog > 0.5 else 1 - p_dog
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence * 100:.1f}%")
