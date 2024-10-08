

# Real-Time Facial Emotion Detection System

This project implements a real-time facial emotion detection system using a pre-trained convolutional neural network (CNN) model. The model detects facial expressions such as angry, disgust, fear, happy, neutral, sad, and surprise from live webcam feed. It is built using Python, OpenCV for real-time video processing, and Keras for loading and predicting with the pre-trained model.

## Dataset

The model is trained using the [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset), which contains 48x48 pixel grayscale images of faces labeled with one of seven emotions.

## Features

- **Real-time emotion detection**: Captures facial expressions via webcam and identifies emotions such as angry, happy, and sad.
- **Facial recognition**: Utilizes OpenCV’s Haar Cascade for face detection.
- **Emotion classification**: A CNN model predicts one of the seven possible emotions.
  
## Model Architecture

The CNN model was trained on the face expression recognition dataset, with several Conv2D layers, MaxPooling, and Dropout layers to prevent overfitting. The architecture includes:

- **Input Layer**: 48x48 grayscale images.
- **Conv2D + MaxPooling Layers**: For feature extraction.
- **Dropout Layers**: To reduce overfitting.
- **Dense Layers**: For final emotion classification.


## Usage

- Once the webcam is turned on, the system will start detecting faces and predicting emotions in real-time.
- The predicted emotion label will be displayed on the live video feed.
- Press `q` to exit the webcam feed.

## Dependencies

- Python 3.x
- TensorFlow/Keras
- OpenCV
- Numpy

## Future Improvements

- Improve model accuracy by retraining on larger datasets.
- Add support for multiple faces in the same frame.
- Explore transfer learning with more advanced pre-trained models.

