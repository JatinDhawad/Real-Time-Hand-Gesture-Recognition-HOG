import cv2
import numpy as np

hog = cv2.HOGDescriptor(
    _winSize=(128, 128),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)

def extract_hog(image):
    image = cv2.resize(image, (128, 128))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog.compute(gray)
    return features.flatten()