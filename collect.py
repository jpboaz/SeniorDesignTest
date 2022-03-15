import cv2 as cv
import os

video = cv.VideoCapture(0)

face = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

count = 0

nameID = str(input("Enter your name: ")).lower()

path = 'images/' + nameID

exists = os.path.exists(path)

if exists:
    print("Name taken")
    nameID = str(input("Enter name again: "))
else:
    os.makedirs(path)

while True:
    ret, frame = video.read()
    faces = face.detectMultiScale(frame, 1.3, 5)
    for x,y,w,h in faces:
        count = count + 1
        name = './images/' + nameID + '/' + str(count) + '.jpg'
        cv.imwrite(name, frame[y:y+h, x:x+w])
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
    cv.imshow("WindowFrame", frame)
    cv.waitKey(1)
    if count > 500:
        break

video.release()
cv.destroyAllWindows()
