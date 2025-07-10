import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

st.set_page_config(page_title="Bean Disease Classifier", layout="centered")

# 1. Load model from Google Drive
@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?id=YOUR_FILE_ID&export=download"
    output_model = "model.h5"
    if not os.path.exists(output_model):
        response = requests.get(url)
        with open(output_model, 'wb') as f:
            f.write(response.content)
    return tf.keras.models.load_model(output_model)

model = load_model()

# 2. Class labels (update if your model has different ones)
class_names = ['angular_leaf_spot', 'bean_rust', 'healthy']

# 3. Header
st.title("🌱 Bean Disease Classifier")
st.write("Upload a leaf image and the model will predict the disease.")

# 4. File uploader
uploaded_file = st.file_uploader("Choose a bean leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))  # Adjust size to match your model's input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Show result
    st.success(f"**Prediction:** {predicted_class}")
