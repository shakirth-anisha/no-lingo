import cv2
import numpy as np
from numpy.linalg import norm
import os

# HOG Feature
def preprocess_hog(images):
    samples = []
    for img in images:
        img = cv2.equalizeHist(img)  
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)

        bin_n = 16
        bin_cells = np.int32(bin_n * ang / (2 * np.pi))
        mag_cells = mag

        hists = [np.bincount(b.ravel(), weights=m.ravel(), minlength=bin_n) 
                 for b, m in zip(np.split(bin_cells, 4), np.split(mag_cells, 4))]
        
        hist = np.hstack(hists)

        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)

# Train
def trainSVM(num_classes):
    imgs, labels = [], []
    
    for i in range(65, 65 + num_classes):
        for j in range(1, 201):  
            img_path = f'TrainData/{chr(i)}_{j}.jpg'
            if os.path.exists(img_path):
                img = cv2.imread(img_path, 0)
                if img is not None:
                    imgs.append(img)
                    labels.append(i - 64)

        print(f'Loaded images for class {chr(i)}')

    if not imgs:
        print("Error: No training images found.")
        return None

    samples = preprocess_hog(imgs)

    print('Training SVM...')
    model = cv2.ml.SVM_create()
    model.setKernel(cv2.ml.SVM_RBF)
    model.setType(cv2.ml.SVM_C_SVC)
    model.setC(2.67)
    model.setGamma(5.383)
    model.train(samples, cv2.ml.ROW_SAMPLE, np.array(labels, dtype=np.int32))

    return model

# Predict
def predict(model, img):
    samples = preprocess_hog([img])
    return model.predict(samples)[1].ravel()
