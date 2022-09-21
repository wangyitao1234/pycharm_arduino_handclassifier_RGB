import cv2
import mediapipe as mp
import time
import numpy as np
import math
import time

class handDetector():
    def __init__(self,mode =False,maxHands =2,complexity =1,detectionCon =0.5,trackCon=0.5):
        self.mode =mode
        self.maxHands =maxHands
        self.complexity =complexity
        self.detectionCon =detectionCon
        self.trackCon =trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw =True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []

        if self.results.multi_hand_landmarks:

            h,w,c =img.shape#高，宽，位深

            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):

                ## lmList
                myHand = {}
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)
                myHand["lmlist"] =mylmList

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH

                myHand["bbox"] = bbox

                #point,line
                cz0 = handLms.landmark[0].z
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

                #handType
                if handType.classification[0].label == "Right":
                    myHand["type"] = "Left"
                else:
                    myHand["type"] = "Right"

                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                              (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                              (255, 0, 255), 2)
                cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 2)
                allHands.append(myHand)

        return img, allHands


def main():
    cap = cv2.VideoCapture(0)

    offset =20

    pTime = 0
    cTime = 0
    color = (0, 255, 0)
    imagesize =300

    counter =0

    FilePath ='./DATA/2'

    detector =handDetector()
    while True:
        success, img = cap.read()#type(img)=ndarray
        #img =cv2.imread("C:/Users/15634/hand.jpg")
        #img = cv2.imread("C:/Users/15634/two_hand.jpg")

        img,myHands=detector.findHands(img)

        try:
            if myHands:
                myHand =myHands[0]
                x,y,w,h =myHand['bbox']

                imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
                imgWhite = np.ones((imagesize, imagesize, 3), np.uint8) * 255

                imgCropshape = imgCrop.shape

                aspectratio = h/ w
                if aspectratio > 1:
                    k=imagesize/h
                    wCal = math.ceil(k*w)
                    imgResize = cv2.resize(imgCrop, (wCal,imagesize))
                    wGap =math.ceil((imagesize-wCal)/2)
                    imgWhite[:,wGap:wGap+wCal] =imgResize
                else:
                    k=imagesize/w
                    hCal = math.ceil(k*h)
                    imgResize = cv2.resize(imgCrop, (imagesize,hCal))
                    hGap =math.ceil((imagesize-hCal)/2)
                    imgWhite[hGap:hGap+hCal,:] =imgResize


                cv2.imshow('imgCrop', imgCrop)
                cv2.imshow("imageWhite", imgWhite)
        except:
                pass

    # Lmlist = detector.findPosition(img)
    # if len(Lmlist) != 0:
    #     print(Lmlist[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, 'fps:' + str(int(fps)), (10, 70), cv2.FONT_ITALIC, 1, color, 3)
        img =cv2.resize(img,(0,0),None,0.5,0.5)
        cv2.imshow('image', img)
        cv2.waitKey(1)
        key = cv2.waitKey(1)
        if key ==ord('s'):
            counter+=1
            cv2.imwrite(f'{FilePath}/images_{time.time()}.jpg',imgWhite)
            print(counter)





if __name__ == '__main__':
    main()