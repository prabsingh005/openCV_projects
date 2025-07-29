import cv2
import mediapipe as mp
import time
import numpy as np
import math
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mphand = mp.solutions.hands
hand = mphand.Hands(min_detection_confidence = 0.7)
mpdraw = mp.solutions.drawing_utils

pTime = 0

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_range = volume.GetVolumeRange()
minVol = vol_range[0]
maxVol = vol_range[1]

vol = 0
volBar = 600
volText = 0
brightness = 0
brightnessBar = 600

while True:
    _, frame = cap.read()

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (30,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hand.process(rgb)

    if result.multi_hand_landmarks:
        for handlm in result.multi_hand_landmarks:
            landmark_id = {}

            for id, lm in enumerate(handlm.landmark):

                h,w,c = frame.shape
                cx,cy = int(lm.x*w), int(lm.y*h)

                landmark_id[id] = (cx,cy)

            if 4 in landmark_id and 8 in landmark_id:
                x1,y1 = landmark_id[4]
                x2,y2 = landmark_id[8]
                
                length = int(math.hypot(x2-x1, y2-y1))

                x3,y3 = landmark_id[12]

                if landmark_id[0] < (w//2,h//2):
                    cv2.putText(frame, 'Right hand', (x3-50,y3-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                    vol = np.interp(length, [40,260], [minVol, maxVol])
                    volBar = np.interp(length, [40,260], [600,100])
                    volText = np.interp(length, [40, 260], [0, 100])
                    volume.SetMasterVolumeLevel(vol, None)

                if landmark_id[0] > (w//2,h//2):
                    cv2.putText(frame, 'Left hand', (x3-50,y3-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                    brightness = int(np.interp(length, [40, 260], [0, 100]))
                    brightnessBar = int(np.interp(length, [40, 260], [600, 100]))
                    sbc.set_brightness(brightness)

                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                
                cv2.line(frame, (x1,y1), (x2,y2), (255,0,255), 2)
                cv2.circle(frame, (mid_x, mid_y), 13, (255,0,255), -1)
                cv2.circle(frame, landmark_id[4], 15, (255,0,255), -1)
                cv2.circle(frame, landmark_id[8], 15, (255,0,255), -1)

                if length < 40:
                    cv2.circle(frame, (mid_x, mid_y), 13, (0,255,0), -1)
            
            draw = mpdraw.draw_landmarks(frame, handlm, mphand.HAND_CONNECTIONS)


    cv2.rectangle(frame, (50,100), (100,600), (255,0,0), 2)
    cv2.rectangle(frame, (50,int(volBar)), (100,600), (255,0,0), -1)
    cv2.putText(frame, f'Volume: {int(volText)}', (50,650), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

    cv2.rectangle(frame, (1180,100), (1230,600), (255,0,0), 2)
    cv2.rectangle(frame, (1180,int(brightnessBar)), (1230,600), (255,0,0), -1)
    cv2.putText(frame, f'Brightness: {brightness}%', (1000, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('webcam', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
