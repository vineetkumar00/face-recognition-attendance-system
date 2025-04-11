import os
import cv2
import numpy as np

def train_model(data_path='face_data'):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    ids = []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith("jpg") or file.endswith("png"):
                path = os.path.join(root, file)
                label = int(os.path.basename(root))
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                faces.append(img)
                ids.append(label)

    recognizer.train(faces, np.array(ids))
    recognizer.save("trained_model/model.yml")

if __name__ == "__main__":
    train_model()
