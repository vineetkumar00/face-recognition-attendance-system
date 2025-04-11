def recognize_face(recognizer, face_image):
    id_, confidence = recognizer.predict(face_image)
    return id_, confidence
