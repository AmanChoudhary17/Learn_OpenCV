import cv2 as cv
import time
import mediapipe as mp

cap  =cv.VideoCapture("videos/video1.mp4")
pTime  = 0
mpDraw   =  mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 3)
drawSpec = mpDraw.DrawingSpec(color = (255,255,0),thickness = 1,circle_radius =2)

while True: 
    success, frame = cap.read()
    if not success:
        break;
    frame = cv.resize(frame, (840, 460))
    imgRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, facelms, mpFaceMesh.FACEMESH_TESSELATION,drawSpec,drawSpec)
            for id,lm in enumerate(facelms.landmark):
                # print(lm)
                ih,iw,ic = frame.shape
                x,y = int(lm.x*iw),int(lm.y*ih)
                print(id,x,y)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(frame,str(int(fps)),(10,70),cv.FONT_HERSHEY_SIMPLEX,3,(255,0,255),3)
    
    

    cv.imshow('Face Mesh',frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv.destroyAllWindows()