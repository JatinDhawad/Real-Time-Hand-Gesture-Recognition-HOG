import os
import cv2
import joblib
import numpy as np
from sklearn.svm import SVC
from hog_features import extract_hog

dataset_path = "dataset"

X = []
y = []

for label in os.listdir(dataset_path):
    for img_name in os.listdir(f"{dataset_path}/{label}"):
        img = cv2.imread(f"{dataset_path}/{label}/{img_name}")
        features = extract_hog(img)
        X.append(features)
        y.append(label)

X = np.array(X)

model = SVC(kernel="linear", probability=True)
model.fit(X, y)

joblib.dump(model, "hog_model.pkl")
print("Model trained and saved.")