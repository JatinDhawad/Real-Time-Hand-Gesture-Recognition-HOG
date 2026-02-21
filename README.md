# 🖐 Real-Time Hand Gesture Recognition using HOG

## 📌 Overview

This project implements real-time hand gesture recognition using classical computer vision and machine learning techniques.

The system extracts shape-based features using Histogram of Oriented Gradients (HOG) and classifies gestures using a Support Vector Machine (SVM). It performs real-time detection using a webcam.

This project demonstrates feature extraction, supervised learning, and real-time computer vision implementation.

---

## 🧠 Technologies Used

- Python
- OpenCV
- Histogram of Oriented Gradients (HOG)
- Support Vector Machine (SVM)
- Scikit-learn
- NumPy

---

## ⚙️ Working Pipeline

### 1️⃣ Data Collection
Gesture images are stored in class-wise folders:

dataset/
palm/
fist/
ok/
victory/
Each folder contains multiple labeled images of the gesture.

---

### 2️⃣ Feature Extraction (HOG)

- Images are resized to 128×128
- Converted to grayscale
- HOG extracts gradient orientation histograms
- Produces a fixed-length feature vector

HOG Parameters:
- Window size: 128×128
- Block size: 16×16
- Block stride: 8×8
- Cell size: 8×8
- Orientation bins: 9

---

### 3️⃣ Model Training (SVM)

- Extracted HOG features are used as input
- Linear SVM classifier is trained
- Probability mode enabled for confidence estimation
- Model saved as `hog_model.pkl`

---

### 4️⃣ Real-Time Prediction

- Webcam captures live frames
- Region of Interest (ROI) extracted
- HOG features computed from ROI
- SVM predicts gesture class
- Confidence score displayed on screen

Press **'q'** to exit.

---

## 🚀 How to Run

### Step 1 – Install Dependencies

```bash
pip install opencv-python numpy scikit-learn joblib
Step 2 – Train the Model
python train_model.py


This generates:

hog_model.pkl

Step 3 – Run Real-Time Detection
python realtime_predict.py
```

## 📊 Performance Notes

Works best under good lighting

Plain background improves accuracy

Recommended: 150–300 images per gesture

Consistent hand size improves detection stability

## 🎯 Advantages of HOG for Gesture Recognition

Fast computation

Suitable for silhouette-based objects

Works well with classical ML

Lightweight compared to deep learning models

## ⚠ Limitations

Sensitive to lighting changes

Background clutter may reduce accuracy

Requires sufficient dataset variation

## 🔮 Future Improvements

Skin color segmentation for background removal

Majority voting across frames

Deep learning (CNN-based) gesture recognition

Mobile or GUI deployment

Confusion matrix & evaluation metrics

## 👨‍💻 Author

Jatin Dhawad
B.Tech Computer Engineering
Computer Vision Project
