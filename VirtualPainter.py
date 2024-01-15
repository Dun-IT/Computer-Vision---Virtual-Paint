import os

import cv2
import numpy as np

import HandTrackingModule as htm

##############################################
folderPath = 'Image'
myList = os.listdir(folderPath)
overlayList = []
# BGR color
drawColor = (255, 0, 255)
# Brush
brushThickness = 15
# Eraser
eraserThickness = 100
##############################################

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
header = overlayList[0]

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
# 3 channel (RGB),uint8 0->255
imgCanvas = np.zeros((480, 640, 3), np.uint8)

while True:
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find handlandmark
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. checking which finger are up
        fingers = detector.fingerUp()
        print(fingers)

        # 4. If selection mode - Two finger are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if y1 < 50:
                if 35 < x1 < 165:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 205 < x1 < 335:
                    header = overlayList[1]
                    drawColor = (0, 255, 0)
                elif 375 < x1 < 505:
                    header = overlayList[2]
                    drawColor = (255, 0, 0)
                elif 535 < x1 < 580:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. If Drawing mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Kich thuoc anh tu 640x50 px
    # Cai dat header image
    img[0:50, 0:640] = header
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("Invert", imgInv)
    cv2.waitKey(1)
