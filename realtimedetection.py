import cv2
from keras.models import model_from_json
import numpy as np
from keras.models import Sequential, model_from_json

# Load the model from the JSON file
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# Load the model's weights
model.load_weights("facialemotionmodel.h5")

# Load the Haar Cascade file for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Reshape to the model's input shape
    return feature / 255.0  # Normalize the pixel values

# Start the webcam feed
webcam = cv2.VideoCapture(0)

# Labels for the emotion prediction
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, im = webcam.read()
    if not ret:
        break  # If webcam capture fails, exit the loop
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces in the grayscale image
    
    # Loop through detected faces
    for (p, q, r, s) in faces:
        image = gray[q:q+s, p:p+r]  # Crop the face from the image
        cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)  # Draw a rectangle around the face
        image = cv2.resize(image, (48, 48))  # Resize the face to 48x48 (model input size)
        
        img = extract_features(image)  # Preprocess the face image
        pred = model.predict(img)  # Make the emotion prediction
        prediction_label = labels[pred.argmax()]  # Get the label for the predicted emotion
        
        # Overlay the predicted emotion on the video feed
        cv2.putText(im, '%s' % prediction_label, (p, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
    
    # Display the frame with the detected face and predicted emotion
    cv2.imshow("Output", im)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop when 'q' is pressed

# Release the webcam and close windows
webcam.release()
cv2.destroyAllWindows()
