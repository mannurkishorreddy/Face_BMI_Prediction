import cv2
import tensorflow as tf
import numpy as np
#import pandas as pd

# Load the pre-trained BMI prediction model
model = tf.keras.models.load_model('bmi_model.h5')

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face_image = gray[y:y+h, x:x+w]
        face_image = cv2.resize(face_image, (224, 224))

        # Convert grayscale to RGB
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)

        # Preprocess the face image for BMI prediction
        face_image_preprocessed = np.expand_dims(face_image_rgb, axis=0)
        face_image_preprocessed = face_image_preprocessed / 255.0  # Normalize the image

        # Predict BMI using the pre-trained model
        bmi_prediction = model.predict(face_image_preprocessed)

        # Display the predicted BMI on the frame
        bmi_prediction = bmi_prediction.flatten()  # Flatten the prediction array
        bmi_prediction_text = f'BMI: {bmi_prediction:.2f}'
        cv2.putText(frame, bmi_prediction_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Face BMI Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
