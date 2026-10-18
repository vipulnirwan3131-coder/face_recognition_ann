"""
collect_faces.py
------------------
Standalone script to register a new person by capturing face images
from the webcam and saving them into dataset/<person_name>/.

Run directly:
    python collect_faces.py

Or call collect_images_from_webcam() from the Streamlit app.
"""

import os
import cv2
import time
from utils.face_utils import detect_faces, crop_face, DATASET_DIR, ensure_dirs


def collect_images_from_webcam(person_name, num_images=60, cam_index=0,
                                progress_callback=None, frame_callback=None):
    """
    Capture `num_images` face crops for `person_name` from the webcam.

    progress_callback(count, total): optional, called after every saved image
        (used by the Streamlit UI to update a progress bar).
    frame_callback(frame): optional, called with the raw annotated frame
        (used by Streamlit to display a live preview).

    Returns the number of images actually saved.
    """
    ensure_dirs()
    person_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    existing = len([f for f in os.listdir(person_dir) if f.endswith(".jpg")])

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(
            "Camera unavailable. Check that a webcam is connected and not "
            "in use by another application."
        )

    saved = 0
    try:
        while saved < num_images:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame from camera.")

            faces, gray = detect_faces(frame)

            if len(faces) == 0:
                cv2.putText(frame, "No face detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif len(faces) > 1:
                cv2.putText(frame, "Multiple faces detected - show only one",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                (x, y, w, h) = faces[0]
                face_crop = crop_face(gray, (x, y, w, h))
                img_index = existing + saved + 1
                file_path = os.path.join(person_dir, f"image_{img_index:03d}.jpg")
                cv2.imwrite(file_path, face_crop)
                saved += 1

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Captured {saved}/{num_images}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if progress_callback:
                    progress_callback(saved, num_images)

                time.sleep(0.15)  # small delay so consecutive frames differ a bit

            if frame_callback:
                frame_callback(frame)
            else:
                cv2.imshow("Register Face - press q to stop early", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        if not frame_callback:
            cv2.destroyAllWindows()

    return saved


if __name__ == "__main__":
    name = input("Enter person's name/ID: ").strip()
    if not name:
        print("Name cannot be empty.")
    else:
        n = collect_images_from_webcam(name, num_images=60)
        print(f"Saved {n} images for '{name}' in dataset/{name}/")
