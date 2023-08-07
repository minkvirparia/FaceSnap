import cv2
import numpy as np
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

#id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Mink'] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)

# Define min window size to be recognized as a face
minW = 0.1*cam.get(5)
minH = 0.1*cam.get(6)
while True:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (int(minW), int(minH)))

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        # Print the id value for debugging
        print("Predicted ID:", id)

        if (confidence < 100):
            if id >= 0 and id < len(names):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
    
    cv2.imshow('camera',img) 
    k = cv2.waitKey(50) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break


# Do a bit of cleanup
print("\nExiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()