import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
# test
mpface = mp.solutions.face_detection
face = mpface.FaceDetection()

pTime = 0

while True:
    _, frame = cap.read()

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (30,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face.process(rgb)

    if result.detections:
        for id, facelm in enumerate(result.detections):
            bounding_box = facelm.location_data.relative_bounding_box
            
            h,w,c = frame.shape
            cx,cy = int(bounding_box.xmin*w), int(bounding_box.ymin*h)
            cw,ch = int(bounding_box.width*w), int(bounding_box.height*h)
            
            score = int(facelm.score[0] * 100)
            cv2.putText(frame, f'{score}%', (cx,cy-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            
            cv2.rectangle(frame, (cx,cy), ((cx+cw), (cy+ch)), (0,255,0), 2)

    cv2.imshow('webcam', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
