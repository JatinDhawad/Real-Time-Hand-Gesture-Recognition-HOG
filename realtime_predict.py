import cv2
import joblib
import numpy as np
from hog_features import extract_hog

model = joblib.load("hog_model.pkl")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    x1, y1 = 300, 100
    x2, y2 = 600, 400
    roi = frame[y1:y2, x1:x2]

    features = extract_hog(roi)

    prob = model.predict_proba([features])[0]
    pred = model.predict([features])[0]
    confidence = round(np.max(prob), 2)

    cv2.putText(frame, f"{pred} ({confidence})",
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.rectangle(frame, (x1,y1), (x2,y2),
                  (0,255,0), 2)

    cv2.imshow("HOG Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()