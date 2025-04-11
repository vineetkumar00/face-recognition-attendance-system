import cv2
import numpy as np
import os
import datetime
from utils.face_recognition import recognize_face
from utils.face_detection import detect_face

cap = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_model/model.yml")
names = ["", "User1", "User2", "User3"]  # Map label IDs to user names

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_face(gray)

    for (x, y, w, h) in faces:
        id_, confidence = recognize_face(recognizer, gray[y:y + h, x:x + w])
        if confidence < 50:
            name = names[id_]
            color = (0, 255, 0)
            date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("attendance/attendance.csv", "a") as f:
                f.write(f"{name},{date_time}\n")
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
