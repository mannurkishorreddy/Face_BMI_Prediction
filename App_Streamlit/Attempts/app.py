import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('bmi_model.h5')

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to match model input shape
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def main():
    st.title("BMI Prediction App")
    st.write("Upload an image and get the predicted BMI")

    # Upload file and get BMI prediction
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        st.write("Predicted BMI:", prediction[0][0])

if __name__ == '__main__':
    main()