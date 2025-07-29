import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mphand = mp.solutions.hands
hand = mphand.Hands()
mpdraw = mp.solutions.drawing_utils

pTime = 0

while True:
    _, frame = cap.read()

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hand.process(rgb)
    li = []

    if result.multi_hand_landmarks:
        for handlm in result.multi_hand_landmarks:
            # for id, lm in enumerate(handlm.landmark):
            #     h,w,c = frame.shape
            #     cx, cy = int(lm.x * w), int(lm.y * h)

            #     if id == 8:
            #         cv2.circle(frame, (cx,cy), 10, (255,0,255), -1)
                
            mpdraw.draw_landmarks(frame, handlm, mphand.HAND_CONNECTIONS)

    cv2.putText(frame, str(int(fps)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255,0,0), 2)

    cv2.imshow('webcam', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()