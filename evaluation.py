import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_classifier(y_true, y_pred, labels=None) -> dict:
    """Return accuracy, precision, recall, and F1 for a classifier."""
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "F1 Score": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def print_metrics(metrics: dict) -> None:
    for k, v in metrics.items():
        print(f"  {k}: {v}")


def print_report(y_true, y_pred, target_names=None) -> None:
    print(classification_report(y_true, y_pred, target_names=target_names))
