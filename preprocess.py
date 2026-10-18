"""
preprocess.py
--------------
Builds the (X, y) dataset used for training:
- Walks through dataset/<person_name>/*.jpg
- Preprocesses each face image into a flattened, normalized feature vector
- Returns X (features) and y (string labels)

This module is imported by train_model.py. It can also be run directly
to sanity-check dataset size per person.
"""

import os
import cv2
import numpy as np
from utils.face_utils import preprocess_face, DATASET_DIR


def build_dataset():
    """
    Returns:
        X : np.ndarray of shape (num_samples, IMG_SIZE[0]*IMG_SIZE[1])
        y : list of string labels (person names), same length as X
    """
    X, y = [], []

    if not os.path.exists(DATASET_DIR) or len(os.listdir(DATASET_DIR)) == 0:
        raise RuntimeError(
            "Dataset is empty. Please register at least 2 people using "
            "'Register Person' before training."
        )

    people = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    if len(people) < 2:
        raise RuntimeError(
            "At least 2 registered people are required to train a "
            "classifier. Please register more people."
        )

    for person in people:
        person_dir = os.path.join(DATASET_DIR, person)
        image_files = [f for f in os.listdir(person_dir)
                       if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        if len(image_files) < 5:
            print(f"Warning: '{person}' has only {len(image_files)} images. "
                  f"Recognition accuracy for this person may be poor.")

        for img_name in image_files:
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # skip unreadable/corrupt images
            feature_vector = preprocess_face(img)
            X.append(feature_vector)
            y.append(person)

    if len(X) == 0:
        raise RuntimeError("No valid images could be read from the dataset.")

    return np.array(X), y


if __name__ == "__main__":
    X, y = build_dataset()
    print(f"Total samples: {len(X)}")
    print(f"Feature vector length: {X.shape[1]}")
    from collections import Counter
    print("Samples per person:", Counter(y))
