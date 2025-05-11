import cv2 as cv
import mediapipe as mp
import time 

class handDetector():
    def __init__(self,mode= False,maxHands =2,detectionCon =0.5,trackCon =0.5):
        self.mode =mode
        self.maxHands =maxHands
        self.detectionCon =detectionCon
        self.trackCon  = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpdraw = mp.solutions.drawing_utils

    def findHands(self,frame,draw = True):
        imgRGB =cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                
                if draw:
                    # for id, lm in enumerate(handLms.landmark):
                    #     if id == 4 or id == 8:
                    #         h, w, c = frame.shape
                    #         cx, cy = int(lm.x * w), int(lm.y * h)
                    #         cv.circle(frame, (cx, cy), 5, (255, 0, 255), cv.FILLED)
                    self.mpdraw.draw_landmarks(frame,handLms,self.mpHands.HAND_CONNECTIONS)
        return frame
    
    def findPosition(self,frame,handNo = 0,draw =True):
        lmList = []
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(hand.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw :
                        cv.circle(frame, (cx, cy), 15, (255, 0, 255), cv.FILLED)
        return lmList

# print(results.multi_hand_landmarks)









def main():
    cTime = 0 
    pTime = 0
    cap =  cv.VideoCapture(0)
    detector = handDetector()
    while True:
        success, frame = cap.read()
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame,draw = False)
        # if len(lmList) !=0:
        #     print(lmList[4])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv.putText(frame,str(int(fps)),(10,70),cv.FONT_HERSHEY_SIMPLEX,3,(255,0,255),3)
        
        if not success:
            break

        cv.imshow("Webcam Feed", frame)  # Show webcam window

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()


