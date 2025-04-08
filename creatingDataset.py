import cv2
import numpy as np

# Get camera input
cam = int(input("Enter Camera Index: "))
cap = cv2.VideoCapture(cam)

i = 16  # Starting class index
j = 201  # Starting image index
name = ""

def nothing(x):
    pass

# Function to get the largest contour above a certain area threshold
def getMaxContour(contours, minArea=200):
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
    
    # Define new bounding box (400% increase)
    x_center, y_center = 1100, 300  # Approximate center of the previous box
    box_size = 1600 // 2  # Half of the new width and height

    x1, y1 = max(0, x_center - box_size), max(0, y_center - box_size)
    x2, y2 = min(img.shape[1], x_center + box_size), min(img.shape[0], y_center + box_size)

    # Draw new bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    img1 = img[y1:y2, x1:x2]  # Cropped region of interest
    
    # Convert to YCrCb color space and apply Gaussian blur
    img_ycrcb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
    blur = cv2.GaussianBlur(img_ycrcb, (11, 11), 0)

    # Skin color range in YCrCb
    skin_ycrcb_min = np.array((0, 138, 67))
    skin_ycrcb_max = np.array((255, 173, 133))
    
    # Create a mask for skin detection
    mask = cv2.inRange(blur, skin_ycrcb_min, skin_ycrcb_max)

    # Find contours
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = getMaxContour(contours, 4000)

    if cnt is not None:
        x, y, w, h = cv2.boundingRect(cnt)
        imgT = img1[y:y + h, x:x + w]
        imgT = cv2.bitwise_and(imgT, imgT, mask=mask[y:y + h, x:x + w])
        imgT = cv2.resize(imgT, (200, 200))
        cv2.imshow('Trainer', imgT)

    cv2.imshow('Frame', img)
    cv2.imshow('Thresh', mask)

    k = cv2.waitKey(10) & 0xFF
    if k == 27:  # Exit on ESC key
        break
    if k == 13:  # Save image on Enter key
        name = f"TrainData/{chr(i+64)}_{j}.jpg"  # Fix filename format
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
