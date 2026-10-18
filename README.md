# ANN Face Recognition System

A complete Face Recognition System built with **Python, OpenCV, and an Artificial
Neural Network (ANN/MLP)** via scikit-learn, with a **Streamlit** web interface.
Built for a **B.Tech 3rd-year AI/ML project**.

---

## 1. Project Structure

```
face_recognition_ann/
│
├── app.py                 # Streamlit web interface (main entry point)
├── collect_faces.py       # Webcam dataset collection
├── preprocess.py          # Builds (X, y) feature/label arrays from dataset
├── train_model.py         # Trains the ANN (MLPClassifier) + evaluation
├── recognize.py           # Loads model, performs recognition
├── evaluate.py            # Standalone re-evaluation script
├── requirements.txt
├── README.md
│
├── dataset/                # person_name/image_001.jpg ... (created at runtime)
├── models/                 # face_model.pkl, label_encoder.pkl, scaler.pkl, pca.pkl
├── utils/
│   └── face_utils.py       # shared detection/preprocessing helpers
└── outputs/
    ├── confusion_matrix.png
    └── training_graph.png
```

---

## 2. Prerequisites

- Windows 10/11
- [Python 3.10 or 3.11](https://www.python.org/downloads/) installed and added to PATH
- [VS Code](https://code.visualstudio.com/) with the Python extension
- A working webcam

---

## 3. Setup Instructions (VS Code + Windows)

### Step 1 — Open the project folder
Open VS Code → `File > Open Folder` → select the `face_recognition_ann` folder.

### Step 2 — Open a terminal in VS Code
`Terminal > New Terminal` (this opens PowerShell by default on Windows).

### Step 3 — Create a virtual environment
```powershell
python -m venv venv
```

### Step 4 — Activate the virtual environment
```powershell
venv\Scripts\activate
```
If PowerShell blocks the script, run this once (as your normal user):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
You should see `(venv)` at the start of your terminal prompt once activated.
In VS Code, also select this interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → choose the one inside `venv`.

### Step 5 — Install dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Running the Project

### Step 1 — Register people (build the dataset)
Either use the Streamlit UI (recommended, see below) or run directly:
```powershell
python collect_faces.py
```
Enter a name when prompted; the webcam window opens and automatically
captures face images into `dataset/<name>/`. Repeat for at least **2 people**.

### Step 2 — Train the ANN model
```powershell
python train_model.py
```
This preprocesses the dataset, trains the MLP (ANN) classifier, prints
accuracy/precision/recall/F1, and saves:
- `models/face_model.pkl`
- `models/label_encoder.pkl`
- `models/scaler.pkl`
- `models/pca.pkl`
- `outputs/confusion_matrix.png`
- `outputs/training_graph.png`

### Step 3 — Run real-time webcam recognition (optional, script mode)
```powershell
python recognize.py
```
Press `q` to quit the webcam window.

### Step 4 — Launch the full Streamlit application (recommended)
```powershell
streamlit run app.py
```
This opens a browser window with tabs for **Home**, **Register Person**,
**Train Model**, **Face Recognition**, and **Model Evaluation** — you can
do everything (register → train → recognize → evaluate) from the UI without
touching the terminal again.

---

## 5. Typical Workflow

1. `streamlit run app.py`
2. Go to **Register Person** → capture faces for each person (repeat for ≥2 people)
3. Go to **Train Model** → click **Start Training** → review accuracy/metrics
4. Go to **Face Recognition** → upload an image or take a webcam snapshot →
   see the predicted name + confidence, or "Unknown" if confidence is below
   threshold
5. Go to **Model Evaluation** → view confusion matrix & training curve any time

---

## 6. Error Handling Notes

| Situation | Behavior |
|---|---|
| Camera unavailable | Clear `RuntimeError` raised, shown in Streamlit as an error message |
| No face detected during capture/recognition | Skipped frame with an on-screen/UI warning |
| Multiple faces during registration | User is asked to show only one face |
| Empty dataset / < 2 people | Training blocked with an explanatory message |
| Model not trained yet | Recognition/Evaluation tabs show a warning instead of crashing |
| Invalid/corrupt image file | Caught and reported, does not crash the app |
| Low-confidence prediction | Reported as **Unknown Person** rather than a wrong guess |

---

## 7. Security & Privacy Notice

- Facial images are sensitive personal (biometric) data.
- Only collect faces of people who have given explicit consent.
- Keep the `dataset/` and `models/` folders private — do not upload them to
  public repositories.
- This project is intended for **educational/demonstration purposes only**
  and is **not** suitable as-is for production security or surveillance use.
- Limitations: accuracy depends heavily on lighting, camera quality, pose
  variation, and the number of training images; it can be fooled by photos
  of a registered person (no liveness/anti-spoofing check is implemented).

---

## 8. Retraining

Every time you register a new person or add more images to an existing
person, re-run training (`Train Model` tab or `python train_model.py`) so
the ANN and PCA are refit on the updated dataset.
