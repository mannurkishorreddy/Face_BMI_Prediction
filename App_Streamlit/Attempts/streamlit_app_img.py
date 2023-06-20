import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('bmi_model.h5')

def predict_bmi(image):
    # Preprocess the image
    img = image.resize((224, 224))  # Resize the image to match the input size of your model
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the pixel values

    # Expand dimensions to create a batch of size 1
    img_array = np.expand_dims(img_array, axis=0)

    # Make the prediction
    prediction = model.predict(img_array)
    bmi = prediction[0][0]  # Assuming your model returns a single scalar value

    return bmi


def main():
    st.title("BMI Prediction App")

    # File uploader for the image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_image is not None:
        # Check if the uploaded file is an image
        if uploaded_image.type.startswith('image/'):
            # Display the uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Predict BMI
            bmi = predict_bmi(image)

            st.write("Predicted BMI:", bmi)
        else:
            st.write("Please upload a valid image file.")

if __name__ == "__main__":
    main()
