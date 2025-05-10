import cv2 as cv
import time
import mediapipe as mp

cap  = cv.VideoCapture("videos/video1.mp4")
pTime = 0
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    success, frame = cap.read()
    imgRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB) 
    results = faceDetection.process(imgRGB)
    # print(results)

    if results.detections:
        for id,detection in enumerate(results.detections):
            # mpDraw.draw_detection(frame,detection)
            # print(id, detection)
            # print(detection.score)
            # print()
            bboxC = detection.location_data.relative_bounding_box
            ih,iw,ic =frame.shape
            bbox  = int(bboxC.xmin*iw) , int(bboxC.ymin*ih),\
                int(bboxC.width*iw) , int(bboxC.height*ih)
            cv.rectangle(frame,bbox,(255,0,255),2)
            cv.putText(frame,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv.FONT_HERSHEY_SIMPLEX,3,(255,0,255),3)
    if not success:
        break
    
    frame = cv.resize(frame, (840, 460))
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(frame,str(int(fps)),(10,70),cv.FONT_HERSHEY_SIMPLEX,3,(255,0,255),3)
    cv.imshow("Webcam Feed", frame)  # Show webcam window
    # cv.waitKey(1)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
   
cap.release()
cv.destroyAllWindows()