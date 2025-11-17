"""
Evaluation helpers aligned with the business cases (Casosdeuso.md).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data.dataset_loader import ORIGINAL_LABELS, decode_labels

DEFAULT_LABEL_ORDER = ORIGINAL_LABELS  # Negativo, Neutral/Mixto, Positivo


def _to_predictions(y_proba: np.ndarray) -> np.ndarray:
    if y_proba.ndim == 1:
        return y_proba.astype(int)
    return np.argmax(y_proba, axis=1)


def f1_macro(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def recall_negative(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(recall_score(y_true, y_pred, labels=[0], average="macro"))


def precision_positive(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(precision_score(y_true, y_pred, labels=[1], average="macro"))


def normalized_confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> List[List[float]]:
    matrix = confusion_matrix(y_true, y_pred, labels=DEFAULT_LABEL_ORDER, normalize="true")
    return matrix.tolist()


def evaluate_predictions(
    y_true: Sequence[int],
    y_proba: np.ndarray,
    metric_names: Iterable[str],
) -> Dict[str, float]:
    """
    Compute requested metrics from probability outputs.
    """

    y_pred = _to_predictions(y_proba)
    y_true = decode_labels(y_true)
    y_pred = decode_labels(y_pred)
    metrics: Dict[str, float] = {}

    for name in metric_names:
        if name == "f1_macro":
            metrics[name] = f1_macro(y_true, y_pred)
        elif name == "recall_neg":
            metrics[name] = recall_negative(y_true, y_pred)
        elif name == "precision_pos":
            metrics[name] = precision_positive(y_true, y_pred)
        elif name == "confusion_matrix":
            metrics[name] = normalized_confusion(y_true, y_pred)
    return metrics


__all__ = [
    "evaluate_predictions",
    "f1_macro",
    "precision_positive",
    "recall_negative",
    "normalized_confusion",
]

