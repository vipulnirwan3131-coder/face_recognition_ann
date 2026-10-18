"""
face_utils.py
--------------
Shared utility functions used across the project.
"""

import os
import json
import cv2
import numpy as np

IMG_SIZE = (100, 100)
DATASET_DIR = "dataset"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    return faces, gray


def crop_face(gray_image, box):
    x, y, w, h = box
    return gray_image[y:y + h, x:x + w]


def preprocess_face(face_img):
    face_resized = cv2.resize(face_img, IMG_SIZE)
    face_normalized = face_resized.astype("float32") / 255.0
    return face_normalized.flatten()


def ensure_dirs():
    for d in [DATASET_DIR, MODELS_DIR, OUTPUTS_DIR]:
        os.makedirs(d, exist_ok=True)


def list_registered_people():
    ensure_dirs()
    return sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])


def count_images(person_name):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.exists(person_dir):
        return 0
    return len([f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".png"))])


def total_images():
    return sum(count_images(p) for p in list_registered_people())


def load_last_metrics():
    path = os.path.join(MODELS_DIR, "last_metrics.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
