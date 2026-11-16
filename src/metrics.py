import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def evaluate(y_true, prob, threshold=0.5) -> dict:
    """Metrics for the fake class (positive = 1)."""
    y_true = np.asarray(y_true)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob) if both else np.nan,
        "pr_auc": average_precision_score(y_true, prob) if both else np.nan,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
    }


def best_f1_threshold(y_true, prob) -> float:
    """Threshold maximising F1 on the given (training/CV) predictions."""
    grid = np.linspace(0.05, 0.95, 91)
    return float(max(grid, key=lambda t: f1_score(y_true, prob >= t, zero_division=0)))
