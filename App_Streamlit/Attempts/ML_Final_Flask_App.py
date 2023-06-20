from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

def preprocess_image(image_file):
    img = Image.open(image_file)
    img = img.resize((224, 224))  # Adjust the size based on your model's input shape
    img = np.array(img) / 255.0  # Normalize pixel values
    return img

@app.route('/predict_bmi', methods=['POST'])
def predict_bmi():
    # Retrieve the image file from the request
    image_file = request.files['image']
    
    # Preprocess the image
    processed_image = preprocess_image(image_file)
    
    # Load the saved model
    model = tf.keras.models.load_model('bmi_model.h5')
    
    # Make predictions
    predictions = model.predict(np.expand_dims(processed_image, axis=0))
    
    # Convert predictions to BMI value (assuming regression output)
    bmi_prediction = predictions[0][0]  # Adjust this based on your model's output shape
    
    # Return the prediction as a JSON response
    response = {'bmi': bmi_prediction}
    return jsonify(response)

if __name__ == '__main__':
    app.run()
