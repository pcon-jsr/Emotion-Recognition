import cv2
import er_model
from PIL import Image
import numpy as np
import time

model = er_model.model_class()

cap = cv2.VideoCapture(0)

emotion = { 0 : "Anger" , 1 : "Fear" , 2 : "Happy" , 3 : "Sad" , 4 : "Surprise" , 5 : "Neutral"}

face_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier()
face_cascade.load(face_cascade_name)


while(1):
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = cv2.copyMakeBorder(frame,0,0,180,0,cv2.BORDER_CONSTANT,value=[0,0,0])
    gray = cv2.cvtColor(frame, cv2.cv.CV_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,  1.3, 5, 0|cv2.cv.CV_HAAR_SCALE_IMAGE, (150, 150))
    time.sleep(0.35)

    for (x, y, w, h) in faces:
        cropped_image = gray[y:y+h , x:x+w]
        img = Image.fromarray(np.uint8(cropped_image))
        img = img.resize((48, 48), Image.ANTIALIAS)
        img_array = np.asarray(img)
        img_array = np.reshape(img_array, (48, 48, 1))
        img_array = img_array[np.newaxis, :]
        img_array = img_array/225.
        prediction =  model.predict(img_array)
        emotions_present = prediction[0].argsort()[-3:][::-1]
        y=20
        font = cv2.FONT_HERSHEY_COMPLEX
        for idx in emotions_present:
            cv2.putText(frame, str(emotion[idx]) + " : " + str("{0:.2f}".format(prediction[0][idx]*100)), (20,y), font, 0.5, (225, 225, 225), 1)
            y=y+20

    cv2.imshow('Emotion Recognition',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
    	break

cap.release()
cv2.destroyAllWindows()
