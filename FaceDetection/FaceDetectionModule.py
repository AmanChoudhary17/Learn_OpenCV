import cv2 as cv
import time
import mediapipe as mp

class FaceDetector():
    def __init__(self,minDetectionCon = 0.5):
        
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self,frame,draw = True):
        imgRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB) 
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxs = []
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                # mpDraw.draw_detection(frame,detection)
                # print(id, detection)
                # print(detection.score)
                # print()
                bboxC = detection.location_data.relative_bounding_box
                ih,iw,ic =frame.shape
                bbox  = int(bboxC.xmin*iw) , int(bboxC.ymin*ih),\
                    int(bboxC.width*iw) , int(bboxC.height*ih)
                bboxs.append([id,bbox,detection.score])
                # cv.rectangle(frame,bbox,(255,0,255),2)
                if draw: 
                    self.fancyDraw(frame,bbox)
                    cv.putText(frame,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv.FONT_HERSHEY_SIMPLEX,3,(255,0,255),3)

        return frame, bboxs
    
    def fancyDraw(self,frame,bbox,l = 30,t = 10,rt =2):
        x,y,w,h = bbox
        x1,y1 = x+w,y+h
        cv.rectangle(frame,bbox,(255,0,255),rt)
        
        # top left
        cv.line(frame,(x,y),(x+l,y),(2555,0,255),t)
        cv.line(frame,(x,y),(x,y+l),(2555,0,255),t)
        # top right
        cv.line(frame,(x1,y),(x1-l,y),(2555,0,255),t)
        cv.line(frame,(x1,y),(x1,y+l),(2555,0,255),t)
        # bottom right
        cv.line(frame,(x1,y1),(x1-l,y1),(2555,0,255),t)
        cv.line(frame,(x1,y1),(x1,y1-l),(2555,0,255),t)
        # bottom left
        cv.line(frame,(x,y1),(x+l,y1),(2555,0,255),t)
        cv.line(frame,(x,y1),(x,y1-l),(2555,0,255),t)
        


        
        
        return frame

def main():
    cap  = cv.VideoCapture("videos/video1.mp4")
    pTime = 0

    detector  = FaceDetector()
    while True:
        success, frame = cap.read()
        frame,bboxs   = detector.findFaces(frame)
        print(bboxs)
        frame = cv.resize(frame, (940, 660))
        if not success:
            break
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


if __name__ == '__main__':
    main()


