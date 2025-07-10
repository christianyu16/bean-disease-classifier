import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

@st.cache_resource
def load_model():
    file_id = '1pWJ1iVdzxqYmgIz3m8pRJcRsfwRRcnB5'  
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    response = requests.get(url)
    model = tf.keras.models.load_model(BytesIO(response.content))
    return model

model = load_model()

# Define class labels
class_names = ['angular_leaf_spot', 'bean_rust', 'healthy']

st.title("Bean Disease Classifier")
st.write("Upload an image of a bean leaf to classify it.")

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = image.resize((224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0) / 255.0

    # Predict
    predictions = model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = round(100 * np.max(predictions), 2)

    # Display results
    st.markdown(f"### Prediction: `{predicted_class}`")
    st.markdown(f"### Confidence: `{confidence}%`")
