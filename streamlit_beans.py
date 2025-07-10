import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from PIL import Image

@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?id=1pWJ1iVdzxqYmgIz3m8pRJcRsfwRRcnB5"
    output_model = "beans_model.h5"
    gdown.download(url, output_model, quiet=False)
    return tf.keras.models.load_model(output_model)


model = load_model()
class_names = ['angular_leaf_spot', 'bean_rust', 'healthy']

# Streamlit UI
st.title("Bean Disease Image Classifier")
st.write("Upload an image of a bean leaf to detect diseases.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    image = image.resize((224, 224)) 
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = 100 * np.max(predictions)

    st.subheader(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: {confidence:.2f}%")
