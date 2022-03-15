import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2 as cv
from keras.models import load_model

face = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

capture = cv.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)

font = cv.FONT_HERSHEY_COMPLEX

model = load_model('keras_model.h5')

def get_className(classNo):
    if classNo == 0:
        return "Julian"
    else:
        return "Not Found"

while True:
    success, imgOriginal = capture.read()
    faces = face.detectMultiScale(imgOriginal, 1.3, 5)
    for x,y,w,h in faces:
        crop_img = imgOriginal[y:y+h, x:x+h]
        img = cv.resize(crop_img, (224,224))
        img = img.reshape(1, 224, 224, 3)
        pred = model.predict(img)
        classIndex = np.argmax(pred, axis=-1)
        prob = np.amax(pred)
        if classIndex == 0:
            cv.rectangle(imgOriginal, (x,y), (x+w, y+h), (0,255,0), 2)
            cv.rectangle(imgOriginal, (x,y-40),(x+w, y), (0,255,0),-2)
            cv.putText(imgOriginal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv.LINE_AA)
        else:
            cv.rectangle(imgOriginal, (x,y), (x+w, y+h), (0,255,0), 2)
            cv.rectangle(imgOriginal, (x,y-40),(x+w, y), (0,255,0),-2)
            cv.putText(imgOriginal, str("Not Found"),(x,y-10), font, 0.75, (255,255,255),1, cv.LINE_AA)
        cv.putText(imgOriginal,str(round(prob*100, 2))+"%" ,(180, 75), font, 0.75, (255,0,0),2, cv.LINE_AA)
    cv.imshow("Result", imgOriginal)
    k = cv.waitKey(1)
    if k == ord('q'):
        break

capture.release()
cv.destroyAllWindows()