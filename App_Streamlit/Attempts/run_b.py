import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('20230517_Mid.h5')

# Define a function to preprocess the input image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Create a function to perform face detection and predict BMI
def predict_bmi(image):
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    processed_image = preprocess_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if len(faces) > 0:
        processed_face = preprocess_image(image[y:y + h, x:x + w])
        bmi_prediction = model.predict(processed_face)
        bmi_text = "Predicted BMI: {:.2f}".format(bmi_prediction[0][0]).zfill(5)
        cv2.putText(image, bmi_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return image

# Create a Streamlit app
def main():
    st.title("BMI Prediction")
    option = st.sidebar.selectbox("Choose an option", ("Webcam Input", "Upload Image"))

    if option == "Webcam Input":
        st.write("Webcam input is not available in this interface.")
    elif option == "Upload Image":
        image_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            image = np.array(bytearray(image_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            processed_image = predict_bmi(image)
            st.image(processed_image, channels="BGR")
        else:
            st.sidebar.write("Please upload an image.")

if __name__ == "__main__":
    main()
