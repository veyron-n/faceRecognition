import cv2
import numpy as np

# 创建摄像头对象 0表示默认摄像头，如果有多个摄像头，可以尝试使用1,2,3...
cap = cv2.VideoCapture(0)
face_caseade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

count = 0

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()
    

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
        faces = face_caseade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            count += 1
            #cv2.imwrite("dataset" + str(count) + ".jpg", gray[y: y+h, x: x+w])
            

        cv2.imshow("frame", frame)
        cv2.imshow("gray", gray)
        
    # 按下 q 键退出循环
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    elif count == 50:
        break

# 释放摄像头
cap.release()

# 关闭窗口
cv2.destroyAllWindows()
