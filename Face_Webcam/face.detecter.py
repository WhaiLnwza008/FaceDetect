import numpy as np
import cv2

video = cv2.VideoCapture(0)

cascPath = "haarcascade_frontalface_default.xml"

while True:

    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cascPath)
    faces = faceCascade.detectMultiScale(gray)

    for face in faces:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame,"FACE", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX,0.8, (0,255,0),2)

    cv2.imshow("Faces found", frame)
    if cv2.waitKey(1) & 0xFF == ord ("q"):
        break
        

    