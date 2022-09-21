from cvzone.SerialModule import SerialObject
import cv2
from hands_detect_moudle import handDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

arduino =SerialObject("COM6")

handDetector = handDetector()
classifier =Classifier('keras_model.h5','labels.txt')

cap =cv2.VideoCapture(0)

offset =20
imagesize =300
labels=['0','1','2']


while True:
    success,img =cap.read()
    #imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)*
    img,myHands =handDetector.findHands(img)
    imgout =img.copy()

    try:
        if myHands:
            myHand = myHands[0]
            x, y, w, h = myHand['bbox']

            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgWhite = np.ones((imagesize, imagesize, 3), np.uint8) * 255

            imgCropshape = imgCrop.shape

            aspectratio = h / w
            if aspectratio > 1:
                k = imagesize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imagesize))
                wGap = math.ceil((imagesize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
                prediction, idx = classifier.getPrediction(imgWhite)
            else:
                k = imagesize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imagesize, hCal))
                hGap = math.ceil((imagesize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize
                prediction, idx = classifier.getPrediction(imgWhite)

            print(prediction, idx)
            cv2.rectangle(imgout, (x - 20, y - 70), (x + 70, y - 20), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgout, labels[idx], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('imgCrop', imgCrop)
            cv2.imshow("imageWhite", imgWhite)

            if labels[idx]=='0':
                arduino.sendData([1,0,0])
            elif labels[idx] =='1':
                arduino.sendData([0, 1, 0])
            else:
                arduino.sendData([0,0,1])


    except:
        pass

    cv2.imshow('img',img)
    cv2.waitKey(1)