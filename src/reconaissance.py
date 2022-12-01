import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model
import os

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyedetect = cv2.CascadeClassifier("haarcascade_eye.xml")


cap_ecr=cv2.VideoCapture(0)
cap_ecr.set(3, 640)
cap_ecr.set(4, 480)

font=cv2.FONT_HERSHEY_COMPLEX

model = load_model('keras_model.h5')

while True:
	sucess, imgOrignal=cap_ecr.read()
	faces = facedetect.detectMultiScale(imgOrignal,1.3,5)
	eyes = eyedetect.detectMultiScale(imgOrignal,1.3,5) 
	for x,y,w,h in faces:
		crop_img1=imgOrignal[y:y+h,x:x+h]
		img=cv2.resize(crop_img1, (224,224))
		img=img.reshape(1, 224, 224, 3)
		prediction=model.predict(img)
		probabilite=np.amax(prediction) 
		cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,0,255),2)
		cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,0,255),-2)
		cv2.putText(imgOrignal, str("visage"),(x,y-10), font, 0.75, (0,0,0),1, cv2.LINE_AA)
  
		for x,y,w,h in eyes:
			crop_img2=imgOrignal[y:y+h,x:x+w]
			img=cv2.resize(crop_img2, (224,224))
			img=img.reshape(1, 224, 224, 3)
			prediction=model.predict(img)
			probabilite=np.amax(prediction) 
			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(255,0,0),2)
			cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (255,0,0),-1)
			cv2.putText(imgOrignal, str("yeux"),(x,y-10), font, 0.55, (255,255,255),1, cv2.LINE_AA)
   
			cv2.putText(imgOrignal,str(round(probabilite*100, 2))+"%" ,(50, 30), font, 0.75, (255,0,0),2, cv2.LINE_AA)

  		
		cv2.putText(imgOrignal,str(round(probabilite*100, 2))+"%" ,(50, 75), font, 0.75, (0,0,255),2, cv2.LINE_AA)
  
	cv2.imshow("Dectection des visages et yeux ",imgOrignal)
	k=cv2.waitKey(1)
	if k==ord('q'):
		break


cap_ecr.release()
cv2.destroyAllWindows()















