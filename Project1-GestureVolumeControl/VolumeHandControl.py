import cv2 as cv
import time
import numpy as np
import mediapipe as mp
import HandTrackingModule as htm
import math 
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wcam ,hcam = 640,480

cap = cv.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)
pTime = 0



devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(-20.0, None)
# print(volRange)
minVol = volRange[0]
maxVol = volRange[1]

detector = htm.handDetector(detectionCon=0.75)
vol  =0 
volbar = 400
volper = 0 
while True:
    success,frame = cap.read()
    if not success:
        break
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame,draw  =False)
    
    if len(lmList):
        # print(lmList[2])
        x1,y1 = lmList[4][1],lmList[4][2]
        x2,y2 = lmList[8][1],lmList[8][2]
        cx,cy = (x1+x2)//2, (y1+y2)//2
        cv.circle(frame,(x1,y1),10,(255,50,255),cv.FILLED)
        cv.circle(frame,(x2,y2),10,(255,50,255),cv.FILLED)
        cv.line(frame,(x1,y1),(x2,y2),(255,50,255),1)
        cv.circle(frame,(cx,cy),5,(255,50,255),cv.FILLED)

        length = math.hypot(x2-x1,y2-y1)
        print(length)

        vol = np.interp(length,[30,255],[minVol,maxVol])
        volbar = np.interp(length,[30,255],[400,150])
        volper = np.interp(length,[30,255],[0,100])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)


        if length < 30:
            cv.circle(frame,(cx,cy),5,(0,50,255),cv.FILLED)

    cv.rectangle(frame,(50,150),(85,400),(0,255,50),2)
    cv.rectangle(frame,(50,int(volbar)),(85,400),(0,255,50),cv.FILLED)
    cv.putText(frame,f'{int(volper)}%',(40,450),cv.FONT_HERSHEY_TRIPLEX,1,(255,0,0),2)

    cTime =time.time()
    fps = 1/(cTime-pTime)
    pTime =cTime
    cv.putText(frame,f'FPS:{int(fps)}',(10,50),cv.FONT_HERSHEY_TRIPLEX,1,(255,0,0),2)

    

    cv.imshow("Webcam Feed", frame) 

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()