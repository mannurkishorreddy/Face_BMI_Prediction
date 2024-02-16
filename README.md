# Face BMI Prediction using Deep Neural Networks

## Overview
This project implements a deep learning model for predicting Body Mass Index (BMI) using image data. The script begins by importing necessary libraries and setting up the Google Cloud Storage for accessing the image dataset.

## Data Handling
- **Dataset Retrieval**: Images are retrieved from the "ml-files-img" bucket in Google Cloud Storage and saved locally in a directory named "proj".
- **Data Preprocessing**: The data is preprocessed for the model, with specific focus on images ending with `.bmp`, `.jpg`, and `.png` for one part, and `.csv` for another.

## Model Building
- **Deep Learning Framework**: TensorFlow and Keras are used for building the model.
- **Base Model**: VGG16, a pre-trained model, serves as the base, with additional custom layers added.
- **Custom Layers**: Include GlobalAveragePooling2D, Dense layers with ReLU activation, Dropout for regularization, and a final Dense layer for BMI prediction.

## Training and Evaluation
- **Training**: The model is trained with early stopping to prevent overfitting, and image data augmentation techniques are employed.
- **Evaluation Metrics**: Metrics such as RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), AUC (Area Under the Curve), and R2 score are used to evaluate the model's performance.

## Results and Export
- Predictions are compared with actual BMI values, and the results are saved as a CSV file.

## Additional Configurations
- GPU memory growth is enabled for efficient training.
- Mixed precision training is used to enhance performance on compatible hardware.
