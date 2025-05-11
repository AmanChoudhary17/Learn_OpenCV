import cv2 as cv
import mediapipe as mp
import time
import numpy as np


cap = cv.VideoCapture(0)
pTime = 0
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 3)

drawSpec = mpDraw.DrawingSpec(color = (255,255,0),thickness = 1,circle_radius =0)



while True:
    success ,frame  =cap.read()

    if not success:
        break

    imgRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB) 
    results = faceMesh.process(imgRGB)
    face3d = []
    face2d = []
    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame,facelms,mpFaceMesh.FACEMESH_FACE_OVAL,drawSpec,drawSpec)
            for id,lm in enumerate(facelms.landmark):
                ih,iw,ic = frame.shape
                if id ==33 or id ==263 or id ==1 or id ==61 or id ==291 or id ==199 : 
                    if id == 1:
                        nose2d = (lm.x*iw, lm.y*ih)
                        nose3d = (lm.x*iw, lm.y*ih,lm.z*3000)
                
                x,y = int(lm.x*iw),int(lm.y*ih)
                face2d.append([x,y])
                face3d.append([x,y,lm.z])
                face2d = mp.array(face2d,dtype = np.float64)
                face2d = mp.array(face2d,dtype = np.float64)

                focal_length = 1*i
                print(id,x,y)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(frame,str(int(fps)),(10,70),cv.FONT_HERSHEY_SIMPLEX,3,(255,0,255),3)

    cv.imshow('Cam Feed',frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()