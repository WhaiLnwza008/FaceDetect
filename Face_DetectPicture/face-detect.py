import cv2 as cv

img = cv.imread('face.jpg')
face_all = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_all.detectMultiScale(gray)

for (x, y, w, h) in faces:
    cv.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2)

cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()