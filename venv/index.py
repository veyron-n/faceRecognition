import cv2
import numpy as np
import pickle

# 创建摄像头对象 0表示默认摄像头，如果有多个摄像头，可以尝试使用1,2,3...
cap = cv2.VideoCapture(0)

face_caseade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./mytrainer.xml")

labels = {}

with open("label.pickle", "rb") as f:
    origin_labels = pickle.load(f) # { 'zhaolei': 5}
    labels = {v: k for k, v in origin_labels.items()}
    print(labels)

# 循环读取摄像头图像
while True:
    # 逐帧捕获
    ret, frame = cap.read()
    
    # 如果帧读取失败，退出循环
    if not ret:
        break
    
    # 显示图像
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_caseade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5)
        for (x, y, w, h) in faces:
            gray_roi = gray[y: y+h, x: x+w]
            id_, conf = recognizer.predict(gray_roi)
            if conf >= 50:
                print(labels[id_])

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            cv2.putText(frame, str(labels[id_]), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

            cv2.imshow("frame", frame)
        
    # 按下 q 键退出循环
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放摄像头
cap.release()

# 关闭窗口
cv2.destroyAllWindows()
