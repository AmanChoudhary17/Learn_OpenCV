import cv2 as cv
import time
import mediapipe as mp

class FaceMeshDetector():
    def __init__(self,staticMode =False,maxFaces =3,minDetectionCon = 0.5,minTrackCon = 0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        
        self.mpDraw   =  mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=bool(staticMode),
            max_num_faces=int(maxFaces),
            min_detection_confidence=float(minDetectionCon),
            min_tracking_confidence=float(minTrackCon))
        self.drawSpec = self.mpDraw.DrawingSpec(color = (255,255,0),thickness = 1,circle_radius =0)



    def findFaceMesh(self,frame,draw =True):
        frame = cv.resize(frame, (840, 460))
        self.imgRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        
        faces = []
        if self.results.multi_face_landmarks:
            for facelms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, facelms, self.mpFaceMesh.FACEMESH_FACE_OVAL, self.drawSpec, self.drawSpec)
                    # self.mpDraw.draw_landmarks(frame, facelms, self.mpFaceMesh.FACEMESH_TESSELATION,self.drawSpec,self.drawSpec)
                
                face = []
                for id,lm in enumerate(facelms.landmark):
                    # print(lm)
                    ih,iw,ic = frame.shape
                    x,y = int(lm.x*iw),int(lm.y*ih)
                    # cv.putText(frame,str(id),(x,y),cv.FONT_HERSHEY_PLAIN,0.7,(50,255,50),0.5)
                    # print(id,x,y)
                    face.append([x,y])
                faces.append(face)
        return frame,faces


def main():
    cap  =cv.VideoCapture('videos/video1.mp4')
    pTime  = 0
    detector = FaceMeshDetector()
    while True: 
        success, frame = cap.read()
        if not success:
            break
        frame,faces = detector.findFaceMesh(frame)
        if len(faces) != 0:
            print(len(faces))
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv.putText(frame,str(int(fps)),(10,70),cv.FONT_HERSHEY_SIMPLEX,3,(255,0,255),3)
        
        

        cv.imshow('Face Mesh',frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break 

    cap.release()
    cv.destroyAllWindows()

if __name__  == "__main__":
    main()