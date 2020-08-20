import cv2
import numpy as np
from random import randrange
face_cascade = cv2.CascadeClassifier('face.xml')
smile_cascade = cv2.CascadeClassifier('smile.xml')
webcam = cv2.VideoCapture(0)
while True:
    (read, frame) = webcam.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_tracker = face_cascade.detectMultiScale(gray_img)
    # smile_tracker = smile_cascade.detectMultiScale(
    #     gray_img, 1.7, 20)
    if not read:
        break
    for (x, y, w, h) in face_tracker:
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (randrange(256), randrange(256), randrange(256)), 10)
        the_face = frame[y:y+w, x:x+h]
        gray_face = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        smile_tracker = smile_cascade.detectMultiScale(gray_face, 1.7, 20)

        if len(smile_tracker) > 0:
            font = cv2.FONT_HERSHEY_PLAIN
            org = (x, y+h+40)
            fontScale = 3
            color = (255, 255, 255)
            thickness = 3
            cap = cv2.putText(frame, 'Smiling', org, font,
                              fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Smile Detection", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

webcam.release()

cv2.destroyAllWindows()

print("Code Completed")
