import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

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

            # Preprocess the image
            img_array = np.array(image)
            processed_image = preprocess_image(img_array)

            # Predict the BMI
            prediction = model.predict(processed_image)[0]
            bmi = prediction[0]

            # Draw bounding box
            bbox_image = draw_bounding_box(img_array.copy(), 0, 0, img_array.shape[1], img_array.shape[0])
            st.image(bbox_image, caption='Bounding Box', use_column_width=True)

            # Display the predicted BMI value
            st.write("Predicted BMI:", bmi)

    elif option == "Webcam":
        st.write("Coming soon...")

if __name__ == '__main__':
    main()
