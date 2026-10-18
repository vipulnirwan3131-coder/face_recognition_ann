"""
evaluate.py
------------
Loads the already-trained model and re-evaluates it on a fresh
train/test split of the current dataset, printing metrics and
regenerating the confusion matrix / training graph in outputs/.

Useful when you just want evaluation numbers without retraining
from the Streamlit app, e.g. for the project report.

Run:
    python evaluate.py
"""

from train_model import train

if __name__ == "__main__":
    print("Re-running training + evaluation pipeline...")
    results = train(progress_callback=lambda msg, pct: print(f"[{pct*100:5.1f}%] {msg}"))

    print("\n================ MODEL EVALUATION ================")
    print(f"Classes           : {results['classes']}")
    print(f"Training samples  : {results['n_train']}")
    print(f"Testing samples   : {results['n_test']}")
    print(f"Accuracy          : {results['accuracy']:.4f}")
    print(f"Precision         : {results['precision']:.4f}")
    print(f"Recall            : {results['recall']:.4f}")
    print(f"F1-score          : {results['f1']:.4f}")
    print("\nDetailed classification report:")
    print(results["report"])
    print(f"\nConfusion matrix  -> {results['confusion_matrix_path']}")
    print(f"Training graph    -> {results['training_graph_path']}")
    print("====================================================")
