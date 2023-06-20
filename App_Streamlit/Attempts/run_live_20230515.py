import cv2
import numpy as np
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

# Create a function to capture live video from the camera, preprocess the frames, and predict BMI
def predict_bmi_live():
    cap = cv2.VideoCapture(0)  # Open the camera
    
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()  # Read a frame from the camera
        
        # Preprocess the frame
        processed_image = preprocess_image(frame)
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Perform face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Iterate over detected faces
        for (x, y, w, h) in faces:
            # Draw a bounding box around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Extract the face region from the frame
            face_image = frame[y:y + h, x:x + w]
            
            # Preprocess the face image
            processed_face = preprocess_image(face_image)
            
            # Predict BMI
            bmi_prediction = model.predict(processed_face)
            
            # Display the predicted BMI on the frame
            bmi_text = "Predicted BMI: {:.2f}".format(bmi_prediction[0][0]).zfill(5)
            cv2.putText(frame, bmi_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('BMI Prediction', frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all windows

def predict_bmi():
    choice = input("Choose an option:\n1. Webcam Input\n2. Upload Image\n")

    if choice == "1":
        predict_bmi_live()
    elif choice == "2":
        image_path = input("Enter the path of the image: ")
        image = cv2.imread(image_path)
        if image is not None:
            processed_image = preprocess_image(image)
            bmi_prediction = model.predict(processed_image)
            bmi_text = "Predicted BMI: {:.2f}".format(bmi_prediction[0][0]).zfill(5)
            cv2.putText(image, bmi_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('BMI Prediction', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Invalid image path.")
    else:
        print("Invalid choice.")

# Call the predict_bmi() function to choose between webcam input and uploading an image
predict_bmi()