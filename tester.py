import cv2
import os
import numpy as np
import FaceRecognition as fr


test_img=cv2.imread('test_images/sosa.jpeg')
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)
#comment the below 3 lines while testing
#faces,faceID=fr.labels_for_training_data('trainingImages')
#face_recognizer=fr.train_classifier(faces,faceID)
#face_recognizer.write('trainingData.json')

#comment the below 2 lines while training the data
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.json')

name={0:"Soumi",1:"Saswata",2:"Sandipan",3:"Sourav",4:"",5:"Supriyo"}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>47):
        continue
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(800,600))
cv2.imshow("face dtecetion tutorial",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows
