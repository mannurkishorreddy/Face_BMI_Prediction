import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import time

# Load the BMI prediction model
model = tf.keras.models.load_model('bmi_model_mod.h5')

# Define the image preprocessing function
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define the bounding box drawing function
def draw_bounding_box(image, xmin, ymin, xmax, ymax):
    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    return image

# Define the Streamlit app
def main():
    st.title("BMI Prediction App")

    # Choose between file upload or webcam input
    option = st.radio("Select Input Method:", ("Upload File", "Webcam"))

    if option == "Upload File":
        uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Read the image file
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Convert PIL image to OpenCV format
            img_array = np.array(image)
            img_array_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Detect faces in the image
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img_array_rgb, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                st.write("No faces detected in the image.")
            else:
                for (x, y, w, h) in faces:
                    # Preprocess the face image
                    face_image = img_array_rgb[y:y+h, x:x+w]
                    processed_image = preprocess_image(face_image)

                    # Predict the BMI
                    prediction = model.predict(processed_image)[0]
                    bmi = prediction[0]

                    # Draw bounding box on the face
                    img_array_rgb = draw_bounding_box(img_array_rgb, x, y, x+w, y+h)

                    # Display the predicted BMI value on the face
                    cv2.putText(img_array_rgb, f"BMI: {bmi:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Convert the image back to PIL format for display
                result_image = Image.fromarray(cv2.cvtColor(img_array_rgb, cv2.COLOR_BGR2RGB))
                st.image(result_image, caption='Bounding Boxes and Predicted BMI', use_column_width=True)

    elif option == "Webcam":
        # Open the webcam
        cap = cv2.VideoCapture(0)

        # Allow time for the camera to warm up
        st.write("Please wait for 3 seconds before the webcam starts...")  # Inform the user to wait
        time.sleep(3)

