import cv2 
import numpy as np
import svm_train as st

def getMaxContour(contours, minArea=5000):
    maxC = None
    maxArea = minArea
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > maxArea:
            maxArea = area
            maxC = cnt
    return maxC

previous_text = ""

def getGestureImg(cnt, img, mask, model):
    global previous_text
    
    x, y, w, h = cv2.boundingRect(cnt)
    padding = 20  
    x, y = max(0, x - padding), max(0, y - padding)
    w, h = min(img.shape[1] - x, w + 2 * padding), min(img.shape[0] - y, h + 2 * padding)
    
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)  

    imgT = img[y:y + h, x:x + w]
    imgT = cv2.bitwise_and(imgT, imgT, mask=mask[y:y + h, x:x + w])
    imgT = cv2.resize(imgT, (200, 200))
    imgTG = cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY)

    resp = st.predict(model, imgTG)

    char_label = "?"
    if resp is not None and len(resp) > 0:
        char_code = int(resp[0]) + 64
        if 65 <= char_code <= 90:
            char_label = chr(char_code)

    if char_label != "?":
        previous_text += char_label + " "

    return img, char_label
