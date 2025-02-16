import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import os
import HandTrackingModule as htm

#######################
brushThickness = 25
eraserThickness = 100
########################

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetector(detectionCon=0.65, maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

try:
    while True:
        # 1. Import image
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame")
            continue  # Skip to the next iteration if frame is empty

        img = cv2.flip(img, 1)

        # Eye Tracking
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = img.shape
        if landmark_points:
            landmarks = landmark_points[0].landmark
            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(img, (x, y), 3, (0, 255, 0))
                if id == 1:
                    screen_x = screen_w * landmark.x
                    screen_y = screen_h * landmark.y
                    pyautogui.moveTo(screen_x, screen_y)
            left = [landmarks[145], landmarks[159]]
            for landmark in left:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(img, (x, y), 3, (0, 255, 255))
            if (left[0].y - left[1].y) < 0.004:
                pyautogui.click()
                pyautogui.sleep(1)

        # 2. Find Hand Landmarks
        img = detector.findHands(img)
        lmList, _ = detector.findPosition(img, draw=False)  # Added _ to ignore bbox

        if len(lmList) >= 9:
            # tip of index and middle fingers
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            # 3. Check which fingers are up
            fingers = detector.fingersUp()

            # 4. If Selection Mode - Two finger are up
            if fingers[1] and fingers[2]:
                print("Selection Mode")
                if y1 < 125:
                    if 250 < x1 < 450:
                        header = overlayList[0]
                        drawColor = (255, 0, 255)
                    elif 550 < x1 < 750:
                        header = overlayList[1]
                        drawColor = (255, 0, 0)
                    elif 800 < x1 < 950:
                        header = overlayList[2]
                        drawColor = (0, 255, 0)
                    elif 1050 < x1 < 1200:
                        header = overlayList[3]
                        drawColor = (0, 0, 0)
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

            # 5. If Drawing Mode - Index finger is up
            if fingers[1] and not fingers[2]:  # Simplified condition
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                print("Drawing Mode")
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                if drawColor == (0, 0, 0):  # Eraser logic
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                else:
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

                xp, yp = x1, y1

            # Clear Canvas when all fingers are up
            if all(x >= 1 for x in fingers):
                imgCanvas = np.zeros((720, 1280, 3), np.uint8)
                print("Canvas Cleared")  # Add a print statement for debugging

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        img = cv2.addWeighted(img, 0.7, imgCanvas, 0.3, 0)  # Correct Blending

        # Setting the header image
        img[0:125, 0:1280] = header

        cv2.imshow("Image", img)
        cv2.imshow("Canvas", imgCanvas)  # Keep the canvas window for debugging
        # cv2.imshow("Inv", imgInv) # You can comment out this line to remove the inv window
        cv2.waitKey(1)

except KeyboardInterrupt:
    print("Program terminated.")
finally:
    cap.release()
    cv2.destroyAllWindows()
