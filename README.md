#  Emotion, Age & Gender Detection using Deep Learning

This project performs real-time facial **emotion**, **age**, and **gender** detection using Convolutional Neural Networks (CNNs) with a webcam feed. It leverages the **CK+ dataset** for emotion recognition and custom CNNs trained for predicting age and gender.

---

##  Project Structure
emotion-age-gender-detection/

│

├── emotion_finder.py # Main python file for real-time detection script

├── emotion_model.keras # Trained emotion detection model

├── gender_model.keras # Trained gender classification model

├── age_model.keras # Trained age regression model

├── haarcascade_frontalface_default.xml # Haar Cascade for face detection

├── requirements.txt # Python dependencies

├── emotion_finder.ipynb # Main jupyter notebook for real-time detection script

└── README.md # Project documentation


---

##  Model Details

###  Emotion Detection
- Trained on: CK+48 (resized grayscale 48x48)
- Output: 7 emotion classes (`angry`, `happy`, `sad`, `fear`, `disgust`, `surprised`, `contempt`)
- Activation: Softmax

###  Gender Detection
- Trained on: UTKFace subset
- Output: Binary classification (`Male`, `Female`)
- Activation: Sigmoid

###  Age Prediction
- Trained on: UTKFace
- Output: Single regression output (0–100)
- Activation: Linear

---

##  Running the Real-Time Detector

###  Install Requirements

python version 3.10.9 (Recommended) or lower required

pip install -r requirements.txt

### Run the script
python emotion_detector.py

## To quit the webcam
Press q to quit the webcam window.

# Output Sample
When you run the webcam script, each detected face is annotated with:

Emotion label (top of the face)

Gender (below the face on the left)

Predicted age (below the face on the right)

# Datasets used
1)https://www.kaggle.com/datasets/shawon10/ckplus [for expression recognition]

2)https://www.kaggle.com/datasets/moritzm00/utkface-cropped [for age and gender recognition]

