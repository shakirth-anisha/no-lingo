import cv2
import numpy as np

# Get camera input
cam = int(input("Enter Camera Index: "))
cap = cv2.VideoCapture(cam)

i = 16  # Starting class index
j = 201  # Starting image index
name = ""

# Function to get the largest contour above a certain area threshold
def getMaxContour(contours, minArea=4000):
    maxC = None
    maxArea = minArea
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > maxArea:
            maxArea = area
            maxC = cnt
    return maxC

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break  # Exit loop if camera fails
    
    img = cv2.flip(img, 1)  # Ensures correct orientation

    # Define new bounding box
    x_center, y_center = 1100, 300  
    box_size = 800  

    x1, y1 = max(0, x_center - box_size), max(0, y_center - box_size)
    x2, y2 = min(img.shape[1], x_center + box_size), min(img.shape[0], y_center + box_size)

    # UI Enhancements
    overlay = img.copy()
    alpha = 0.5  # Transparency level for overlay
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (50, 50, 50), -1)  # Dark overlay
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    img1 = img[y1:y2, x1:x2]  # Cropped region of interest
    
    # Convert to YCrCb and apply Gaussian blur
    img_ycrcb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
    blur = cv2.GaussianBlur(img_ycrcb, (11, 11), 0)

    # Skin color range
    skin_ycrcb_min = np.array((0, 138, 67))
    skin_ycrcb_max = np.array((255, 173, 133))
    
    # Create a mask
    mask = cv2.inRange(blur, skin_ycrcb_min, skin_ycrcb_max)

    # Find contours
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = getMaxContour(contours)

    if cnt is not None:
        x, y, w, h = cv2.boundingRect(cnt)
        imgT = img1[y:y + h, x:x + w]
        imgT = cv2.bitwise_and(imgT, imgT, mask=mask[y:y + h, x:x + w])
        imgT = cv2.resize(imgT, (200, 200))

        cv2.imshow('Trainer', imgT)

    # UI Text
    cv2.putText(img, "Hand Capture", (x1 + 20, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow('Frame', img)
    cv2.imshow('Thresh', mask)

    k = cv2.waitKey(10) & 0xFF
    if k == 27:  # Exit on ESC key
        break
    if k == 13:  # Save image on Enter key
        name = f"TrainData/{chr(i+64)}_{j}.jpg"  
        cv2.imwrite(name, imgT)
        print(f"Saved: {name}")
        
        if j < 400:
            j += 1
        else:
            print("Press 'n' to start a new class")
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('n'):
                    j = 201
                    i += 1
                    break

cap.release()
cv2.destroyAllWindows()
