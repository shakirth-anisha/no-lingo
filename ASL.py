import cv2
import numpy as np
import util as ut
import svm_train as st
import re

model = st.trainSVM(17)

cam = int(input("Enter Camera number: "))
cap = cv2.VideoCapture(cam)
font = cv2.FONT_HERSHEY_SIMPLEX

text = " "
temp = 0
previouslabel = None
previousText = " "
label = None

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1) 

    # Selection area
    cv2.rectangle(img, (700, 100), (1400, 600), (0, 255, 0), 3)  

    img1 = img[100:600, 700:1400]
    img_ycrcb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
    blur = cv2.GaussianBlur(img_ycrcb, (11, 11), 0)

    # Skin detection
    mask = cv2.inRange(blur, np.array((0, 138, 67)), np.array((255, 173, 133)))

    # Find contours
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = ut.getMaxContour(contours, 4000)

    if cnt is not None:
        gesture, label = ut.getGestureImg(cnt, img1, mask, model)
        if label is not None:
            if temp == 0:
                previouslabel = label
            if previouslabel == label:
                temp += 1
            else:
                temp = 0
            
            if temp == 40:
                if label == 'P':
                    label = " "
                text += label
                if label == 'Q':
                    words = re.split(" +", text)
                    words.pop()
                    text = " ".join(words)
                print(text)

        cv2.putText(img, f"Current: {label}", (50, 100), font, 3, (0, 255, 255), 5)
        cv2.rectangle(img, (0, img.shape[0] - 70), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
        cv2.putText(img, f"Previous: {text}", (50, img.shape[0] - 20), font, 2, (255, 255, 255), 4)

    cv2.imshow('Frame', img)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(10) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
