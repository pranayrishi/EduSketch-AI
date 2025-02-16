import cv2
import mediapipe as mp
import time
import math

class HandDetector:
 def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
     self.mode = mode
     self.maxHands = maxHands
     self.detectionCon = detectionCon
     self.trackCon = trackCon

     self.mpHands = mp.solutions.hands
     self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                     min_detection_confidence=float(self.detectionCon), min_tracking_confidence=float(self.trackCon))
     self.mpDraw = mp.solutions.drawing_utils
     self.tipIds = [4, 8, 12, 16, 20]

 def findHands(self, img, draw=True):
     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     self.results = self.hands.process(imgRGB)

     if self.results.multi_hand_landmarks:
         for handLms in self.results.multi_hand_landmarks:
             if draw:
                 self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

     return img

 def findPosition(self, img, handNo=0, draw=True):
     xList = []
     yList = []
     self.lmList = []
     if self.results.multi_hand_landmarks:
         myHand = self.results.multi_hand_landmarks[handNo]
         for id, lm in enumerate(myHand.landmark):
             h, w, c = img.shape
             cx, cy = int(lm.x * w), int(lm.y * h)
             xList.append(cx)
             yList.append(cy)
             self.lmList.append([id, cx, cy])
             if draw:
                 cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
         bbox = min(xList), min(yList), max(xList), max(yList)
         if draw:
             cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
         return self.lmList, bbox
     return [], []  # Add this return statement to avoid returning None

 def fingersUp(self):
     fingers = []
     if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
         fingers.append(1)
     else:
         fingers.append(0)

     for id in range(1, 5):
         if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
             fingers.append(1)
         else:
             fingers.append(0)
     return fingers

 def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
     x1, y1 = self.lmList[p1][1:]
     x2, y2 = self.lmList[p2][1:]
     cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
     if draw:
         cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
         cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
         cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
         cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
     length = math.hypot(x2 - x1, y2 - y1)
     return length, img, [x1, y1, x2, y2, cx, cy]

def main():
 pTime = 0
 cap = cv2.VideoCapture(0)
 detector = HandDetector()

 while cap.isOpened():
     success, img = cap.read()
     if not success:
         print("Failed to capture image")
         break

     img = detector.findHands(img)
     lmList, bbox = detector.findPosition(img)
     if lmList:
         x1,y1 = lmList[8][1:]
         x2,y2 = lmList[12][1:]

     cTime = time.time()
     fps = 1 / (cTime - pTime)
     pTime = cTime

     cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
     cv2.imshow("Image", img)

     if cv2.waitKey(1) & 0xFF == ord('q'):
         break

 cap.release()
 cv2.destroyAllWindows()

if __name__ == "__main__":
 main()
