"""
train_model.py
----------------
Trains the ANN (Multi-Layer Perceptron) face recognition classifier.

Pipeline:
    dataset images -> preprocess.build_dataset() -> train/test split
    -> scaling (fit on TRAIN only) -> PCA (fit on TRAIN only)
    -> MLPClassifier (ANN) -> evaluate on held-out TEST -> save

IMPORTANT FIX (vs earlier version):
    Previously the StandardScaler and PCA were fit on the *entire* dataset
    BEFORE the train/test split. That leaks information about the test
    images into the scaler/PCA, which (a) makes the reported test accuracy
    look better than the model actually is, and (b) encourages the model to
    overfit to the exact faces it was shown, so it then mis-recognizes
    anyone new (or even the same people in slightly different lighting) in
    live webcam use. Scaler and PCA must only ever be *fit* on the training
    split, and only *transform* (never fit) on the test split / on live
    frames. That's what this version does.

We use scikit-learn's MLPClassifier, which is a true Artificial Neural
Network (fully-connected feedforward network trained with backpropagation
and stochastic gradient-based optimization), just with a simpler API than
TensorFlow/Keras - appropriate for a B.Tech 3rd-year project while still
genuinely demonstrating ANN concepts (weights, bias, activation functions,
backprop, epochs, learning rate, loss).

Run:
    python train_model.py
"""

import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # so it works without a display (needed for Streamlit/server use)
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from preprocess import build_dataset
from utils.face_utils import MODELS_DIR, OUTPUTS_DIR, ensure_dirs

# ---------------------------------------------------------------------------
# ANN / PCA hyperparameters
# ---------------------------------------------------------------------------
PCA_COMPONENTS = 100          # upper bound; actual value is capped by train-set size
HIDDEN_LAYERS = (256, 128)    # two hidden Dense/ReLU layers
ACTIVATION = "relu"
MAX_EPOCHS = 400
LEARNING_RATE_INIT = 0.001
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Minimum images recommended per person for the model to generalize at all.
MIN_IMAGES_PER_PERSON = 40


def train(progress_callback=None):
    """
    Full training pipeline. Returns a dict of results/metrics so a caller
    (e.g. the Streamlit app) can display them without re-reading files.
    """
    ensure_dirs()

    if progress_callback:
        progress_callback("Loading and preprocessing dataset...", 0.1)
    X, y_raw = build_dataset()

    # ---- Sanity-check dataset balance / size BEFORE training ----
    # A severely small or imbalanced dataset is the #1 cause of a model
    # that "recognizes everyone as the same person" in live use, even if
    # the training pipeline itself is correct.
    counts = Counter(y_raw)
    warnings = []
    for person, n in counts.items():
        if n < MIN_IMAGES_PER_PERSON:
            warnings.append(f"'{person}' has only {n} images (recommended: {MIN_IMAGES_PER_PERSON}+)")
    if len(counts) >= 2:
        max_n, min_n = max(counts.values()), min(counts.values())
        if max_n > 2 * min_n:
            warnings.append(
                f"Dataset is imbalanced: largest class has {max_n} images, "
                f"smallest has {min_n}. The model will be biased toward the larger class."
            )

    # ---- Encode string labels ("rahul", "priya", ...) to integers ----
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    if progress_callback:
        progress_callback("Splitting train/test sets...", 0.25)

    # Split FIRST, on the raw feature vectors — before any fitting happens.
    # stratify keeps class proportions balanced across train/test.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    if progress_callback:
        progress_callback("Scaling features (fit on training data only)...", 0.35)

    # ---- Feature scaling (zero mean, unit variance) ----
    # Fit ONLY on the training split, then apply the same transform to the
    # test split (and, at inference time, to live webcam frames). Fitting
    # on test/live data would leak information and make the model look
    # better than it really is.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    if progress_callback:
        progress_callback("Applying PCA (fit on training data only)...", 0.45)

    # ---- PCA: dimensionality reduction ----
    # Raw feature vectors are IMG_SIZE[0]*IMG_SIZE[1] dimensions. PCA finds
    # the directions of maximum variance in the TRAINING data and projects
    # onto the top-K, drastically cutting computation and reducing the risk
    # of overfitting, while retaining most of the useful facial variation.
    # n_components is capped by the number of TRAINING samples (PCA cannot
    # produce more components than samples-1).
    n_components = min(PCA_COMPONENTS, X_train_scaled.shape[0] - 1, X_train_scaled.shape[1])
    pca = PCA(n_components=n_components, whiten=True, random_state=RANDOM_STATE)
    X_train = pca.fit_transform(X_train_scaled)
    X_test = pca.transform(X_test_scaled)

    if progress_callback:
        progress_callback("Training ANN (MLPClassifier)...", 0.6)

    # ---- The ANN itself ----
    # hidden_layer_sizes=(256,128): Input -> Dense(256, ReLU) -> Dense(128, ReLU)
    #   -> Output(Softmax, auto for multi-class classification)
    # solver='adam': gradient-descent-based optimizer (adaptive learning rate)
    # alpha: L2 regularization to reduce overfitting
    # early_stopping: stops training when validation score stops improving
    #   (acts like a lightweight Dropout/overfitting guard)
    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation=ACTIVATION,
        solver="adam",
        alpha=1e-3,  # slightly stronger L2 regularization for small datasets
        batch_size="auto",
        learning_rate_init=LEARNING_RATE_INIT,
        max_iter=MAX_EPOCHS,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    model.fit(X_train, y_train)

    if progress_callback:
        progress_callback("Evaluating model on held-out test set...", 0.85)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "report": classification_report(
            y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
        ),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_classes": len(label_encoder.classes_),
        "classes": list(label_encoder.classes_),
        "n_iterations_run": model.n_iter_,
        "dataset_warnings": warnings,
        "images_per_class": dict(counts),
    }

    # ---- Confusion matrix plot ----
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # ---- Loss curve plot (proxy for training progress across epochs) ----
    plt.figure(figsize=(7, 5))
    plt.plot(model.loss_curve_, label="Training Loss")
    if hasattr(model, "validation_scores_") and model.validation_scores_:
        plt.plot(model.validation_scores_, label="Validation Accuracy")
    plt.xlabel("Epoch (Iteration)")
    plt.ylabel("Loss / Accuracy")
    plt.title("ANN Training Curve")
    plt.legend()
    plt.tight_layout()
    curve_path = os.path.join(OUTPUTS_DIR, "training_graph.png")
    plt.savefig(curve_path)
    plt.close()

    # ---- Save model + preprocessing objects ----
    # scaler and pca were fit on the TRAINING split only, so they are the
    # correct objects to reuse at inference time (recognize.py already does
    # scaler.transform(...) / pca.transform(...), never .fit(...), so no
    # change is needed there).
    joblib.dump(model, os.path.join(MODELS_DIR, "face_model.pkl"))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(pca, os.path.join(MODELS_DIR, "pca.pkl"))

    if progress_callback:
        progress_callback("Training complete.", 1.0)

    metrics["confusion_matrix_path"] = cm_path
    metrics["training_graph_path"] = curve_path
    return metrics


if __name__ == "__main__":
    results = train(progress_callback=lambda msg, pct: print(f"[{pct*100:5.1f}%] {msg}"))
    print("\n--- Evaluation ---")
    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")
    print(f"F1-score : {results['f1']:.4f}")
    if results["dataset_warnings"]:
        print("\n⚠️  Dataset warnings:")
        for w in results["dataset_warnings"]:
            print(f"  - {w}")
    print("\nClassification report:\n", results["report"])
    print(f"Confusion matrix saved to: {results['confusion_matrix_path']}")
    print(f"Training graph saved to: {results['training_graph_path']}")