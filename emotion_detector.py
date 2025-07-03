from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import tensorflow as tf
from keras.metrics import MeanSquaredError

face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
emotion_model = load_model(r'emotion_detection_model_100epochs.keras')
age_model = load_model(r'age_model_50epochs.keras', custom_objects={'mse': MeanSquaredError()})
gender_model = load_model(r'gender_model_50epochs.keras')

class_labels = ['angry', 'contempt', 'disgust', 'fear', 'happy','sadness', 'surprised']
gender_labels = ['Male','Female']


cap = cv2.VideoCapture(0)
while True:
  ret,frame = cap.read()
  labels=[]
  gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  faces = face_classifier.detectMultiScale(
    gray,
    1.1,
    4,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
    )

  for(x,y,w,h) in faces:
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)

    roi_gray = gray[y:y+h,x:x+w]
    roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


    roi = roi_gray.astype('float')/255.0
    roi = np.expand_dims(roi,axis=-1)
    roi = np.expand_dims(roi, axis=0)
    # roi = img_to_array(roi)
    # roi = np.expand_dims(roi,axis=0)


    preds = emotion_model.predict(roi,verbose=0)[0]
    label = class_labels[np.argmax(preds)]
    label_position = (x,y)
    cv2.putText(frame,label,label_position,cv2.FACE_RECOGNIZER_SF_FR_COSINE,1,(0,255,255),2)

    roi_color = frame[y:y+h,x:x+w]
    roi_color = cv2.resize(roi_color,(100,100),interpolation=cv2.INTER_AREA)
    roi_color = roi_color.astype('float32') / 255.0
    roi_color_exp = np.expand_dims(roi_color, axis=0)
    
    gender_predict = gender_model.predict(roi_color_exp, verbose = 0)
    # gender_predict = (gender_predict>=0.5).astype(int)[:,0]
    gender_label = gender_labels[int(gender_predict[0][0]>=0.5)]
    gender_label_position = (x,y+h+50)
    cv2.putText(frame,gender_label,gender_label_position,cv2.FACE_RECOGNIZER_SF_FR_COSINE,1,(0,255,255),2)

    age_predict = age_model.predict(roi_color_exp,verbose=0)
    age = str(round(age_predict[0,0]))
    age_label_position = (x+h,y+h)
    cv2.putText(frame,"Age ="+str(age),age_label_position,cv2.FACE_RECOGNIZER_SF_FR_COSINE,1,(0,255,255),2)

  cv2.imshow('emotion Detector',frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()



