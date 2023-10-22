import os
import pickle
import cv2
import numpy as np

current_id = 0
# {'zhaolei': 5, ...}
label_ids = {}
x_train = []
y_labels = []

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "dataset")

recognizer = cv2.face.LBPHFaceRecognizer_create()

classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

for root, dirs, files in os.walk(image_dir):
    for file in files:
        path = os.path.join(root, file)
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_array = np.array(gray, "uint8")
        label = os.path.basename(root)

        if not label in label_ids:
            label_ids[label] = current_id
            current_id += 1

        id_ = label_ids[label]

        faces = classifier.detectMultiScale(image_array, scaleFactor = 1.5, minNeighbors = 5)

        for (x, y, w, h) in faces:
            roi = image_array[y: y+h, x: x+w]
            x_train.append(roi)
            y_labels.append(id_)

with open("label.pickle", "wb") as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("mytrainer.xml")