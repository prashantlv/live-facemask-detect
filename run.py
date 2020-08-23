import cv2 
import numpy
from tensorflow.keras.models import load_model

model = load_model('./models/model-8.py')

label_dict = {0:'without mask', 1:'masked'}
color_dict = (0:(0,0,255),1:(0,255,0))
webcam = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    (val, im) = webcam.read()
    im = cv2.flip(im,1,1)    # create mirror image 

    img = cv2.resize(im, (im.shape[1]//4 , im.shape[0]//4))
    faces = classifier.detectMultiScale(img)

    