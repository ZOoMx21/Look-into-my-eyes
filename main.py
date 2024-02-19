import cv2
import cvzone
from FaceMeshModule import FaceMeshDetector
import time
from SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0) #detect the camera to use it
detector = FaceMeshDetector(maxFaces=1) # face mesh detector for 1 face only

idList = [22, 23, 24, 26, 110, 130, 157, 158, 159, 160, 161, 243]
ratioList = []
blinkCount = 0
frameCount = 0
startTime = time.time()
previousTime = time.time()
ded = False

segmentor = SelfiSegmentation()
imgList = os.listdir("imgs")
index = 0
imgs =[]
sPressed = True

for imgPath in imgList:
    bg = cv2.imread(f"imgs/{imgPath}")
    imgs.append(bg)

while True:
    success, img = cap.read()  # read frames from camera
    img = cv2.flip(img,1)
    imgBg = segmentor.removeBG(img, imgs[index], cutThreshold=0.8)

    img, faces = detector.findFaceMesh(img, draw=False)  # detect faces in camera frames without drawing face points
    if faces:

        currentTime = time.time()
        if currentTime - startTime >= 60:
            if blinkCount > 25:
                ded = True
            else:
                ded = False
            blinkCount = 0
            startTime = currentTime

        if currentTime - previousTime >= 1:
            elapsedTime = currentTime - startTime
            minutes = int(elapsedTime // 60)
            seconds = int(elapsedTime % 60)
            previousTime = currentTime

        face = faces[0] # store the first face detected
        up = face[159]
        down = face[23]
        left = face[130]
        right = face[243]
        pointLeft = face[145]  # 145 is the point number of left eye in face mesh detector
        pointRight = face[374]  # 374 is the point number of right eye in face mesh detector
        vertical, _ = detector.findDistance(up, down)
        horizontal, _ = detector.findDistance(right, left)

        ratio = int((vertical / horizontal) * 100)
        ratioList.append(ratio)

        w, _ = detector.findDistance(pointLeft, pointRight)  # calculate the pixels number between eyes
        W = 6.3  # the most common distance between eyes in cm
        # d = 50 # default object distance in cm
        # f = (w*d)/W # calculate focal length in mm
        # print(f)
        f = 700
        d = int((W * f) / w)

        if len(ratioList) > 2:
            ratioList.pop(0)
        ratioAvg = sum(ratioList)/len(ratioList)

        if ratioAvg < 25 and frameCount == 0:
            blinkCount += 1
            frameCount = 1
        if frameCount !=0:
            frameCount +=1
            if frameCount > 10:
                
                frameCount = 0

    if sPressed == True:
        if d<50:
            cvzone.putTextRect(img, f'Mesafe: {d}cm', (face[10][0] - 130, face[10][1] - 50), scale=2, colorR=(0, 0, 255))
            cvzone.putTextRect(img, f'Lutfen ekrandan geriye cekinin!!', pos=(10,30), scale=2, colorR=(0, 0, 255))
        else:
            cvzone.putTextRect(img, f'Mesafe: {d}cm', (face[10][0] - 130, face[10][1] - 50), scale=2, colorR=(0, 200, 0))
            cvzone.putTextRect(img, f'Saglikli mesafedesiniz', pos=(10, 30), scale=2, colorR=(0, 200, 0))
        if ded:
            cvzone.putTextRect(img, f'Gozleriniz kuru oldu, ara verin.', pos=(10, 70), scale=2, colorR=(0, 0, 255))
        else:
            cvzone.putTextRect(img, f'Gozlerinizin durumu iyi.', pos=(10, 70), scale=2, colorR=(0, 200, 0))
        cvzone.putTextRect(img, f'Kirpma: {blinkCount}, Zaman: {minutes:02d}:{seconds:02d}', (10, 470))
        img = cv2.resize(img, (640, 480))
        cv2.imshow('Img', img)
    elif sPressed == False:
        if d<50:
            cvzone.putTextRect(imgBg, f'Mesafe: {d}cm', (face[10][0] - 130, face[10][1] - 50), scale=2, colorR=(0, 0, 255))
            cvzone.putTextRect(imgBg, f'Lutfen ekrandan geriye cekinin!!', pos=(10,30), scale=2, colorR=(0, 0, 255))
        else:
            cvzone.putTextRect(imgBg, f'Mesafe: {d}cm', (face[10][0] - 130, face[10][1] - 50), scale=2, colorR=(0, 200, 0))
            cvzone.putTextRect(imgBg, f'Saglikli mesafedesiniz', pos=(10, 30), scale=2, colorR=(0, 200, 0))
        if ded:
            cvzone.putTextRect(imgBg, f'Gozleriniz kuru oldu, ara verin.', pos=(10, 70), scale=2, colorR=(0, 0, 255))
        else:
            cvzone.putTextRect(imgBg, f'Gozlerinizin durumu iyi.', pos=(10, 70), scale=2, colorR=(0, 200, 0))
        cvzone.putTextRect(imgBg, f'Kirpma: {blinkCount}, Zaman: {minutes:02d}:{seconds:02d}', (10, 470))
        img = cv2.resize(imgBg, (640, 480))
        cv2.imshow('Img', imgBg)

    key = cv2.waitKey(1)
    if key == ord('a'):
        sPressed = False
        if index > 0:
            index -= 1
    elif key == ord('d'):
        sPressed = False
        if index < len(imgs) - 1:
            index += 1
    elif key == ord('s'):
        sPressed = True
    elif key == ord('q'):
        break
