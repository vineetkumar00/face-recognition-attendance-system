import cv2

def detect_face(gray_image):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5)
    return faces
