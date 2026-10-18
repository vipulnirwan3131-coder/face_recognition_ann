"""
recognize.py
-------------
Loads the trained ANN model + preprocessing objects and performs
face recognition on:
  (a) a single uploaded image, or
  (b) live webcam frames.

Unknown-person handling:
  The MLP outputs a probability (softmax) per class via predict_proba().
  If the highest probability is below CONFIDENCE_THRESHOLD, the face is
  labeled "Unknown" instead of being force-assigned to the closest class.
  This matters because softmax always sums to 1 across KNOWN classes only
  - it has no built-in notion of "none of the above" - so an explicit
  threshold is required to reject unfamiliar faces.

NOTE: scaler.transform(...) / pca.transform(...) below are TRANSFORM-ONLY
calls (never .fit(...)) — the scaler and PCA were already fit on the
training split inside train_model.py, so using them here on live frames
is correct and does not leak any information.
"""

import os
import cv2
import joblib
import numpy as np

from utils.face_utils import detect_faces, crop_face, preprocess_face, MODELS_DIR

# Raised from 0.55 -> 0.65. With a small training set the ANN can still be
# confidently *wrong*, so a slightly higher bar for accepting a match as
# "known" meaningfully reduces false positives (strangers / other people
# being labeled as a registered person). If you retrain with a much larger,
# balanced dataset (80-100+ images/person) you can lower this again.
CONFIDENCE_THRESHOLD = 0.65  # tune based on validation results

MODEL_PATH = os.path.join(MODELS_DIR, "face_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
PCA_PATH = os.path.join(MODELS_DIR, "pca.pkl")


class FaceRecognizer:
    def __init__(self):
        for path, name in [(MODEL_PATH, "model"), (ENCODER_PATH, "label encoder"),
                            (SCALER_PATH, "scaler"), (PCA_PATH, "PCA")]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Trained {name} not found at '{path}'. "
                    f"Please train the model first (Train Model tab)."
                )
        self.model = joblib.load(MODEL_PATH)
        self.label_encoder = joblib.load(ENCODER_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.pca = joblib.load(PCA_PATH)

    def predict_face(self, face_gray_img):
        """
        Predict identity for a single cropped grayscale face image.
        Returns (name, confidence) where name is "Unknown" if below threshold.
        """
        feature_vector = preprocess_face(face_gray_img).reshape(1, -1)
        feature_scaled = self.scaler.transform(feature_vector)
        feature_pca = self.pca.transform(feature_scaled)

        probabilities = self.model.predict_proba(feature_pca)[0]
        best_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[best_idx])

        if confidence >= CONFIDENCE_THRESHOLD:
            name = self.label_encoder.inverse_transform([best_idx])[0]
        else:
            name = "Unknown"

        return name, confidence

    def recognize_frame(self, frame):
        """
        Run full pipeline on a BGR frame: detect -> crop -> preprocess ->
        predict for every face found. Returns annotated frame + list of
        (name, confidence, box) results.
        """
        faces, gray = detect_faces(frame)
        results = []

        for (x, y, w, h) in faces:
            face_crop = crop_face(gray, (x, y, w, h))
            name, confidence = self.predict_face(face_crop)
            results.append((name, confidence, (x, y, w, h)))

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{name} ({confidence*100:.1f}%)"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame, results

    def recognize_image_file(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError("Invalid image file - could not be read.")
        return self.recognize_frame(frame)


if __name__ == "__main__":
    recognizer = FaceRecognizer()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera unavailable.")

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, results = recognizer.recognize_frame(frame)
        cv2.imshow("Face Recognition - press q to quit", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()