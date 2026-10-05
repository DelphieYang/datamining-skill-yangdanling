"""Evaluation and model comparison.

Reports accuracy first, followed by class-appropriate metrics (balanced
accuracy, precision/recall/F1, ROC-AUC) and a confusion matrix.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Return a metrics dict for a fitted pipeline."""
    y_pred = model.predict(X_test)
    binary = len(np.unique(y_test)) == 2

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }
    if binary:
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        except Exception:
            metrics["roc_auc"] = None

    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    return metrics


def compare(model_results: dict, metrics_by_model: dict,
            output_path: str | Path) -> pd.DataFrame:
    """Write a comparison CSV and return the table."""
    rows = []
    for name, res in model_results.items():
        m = metrics_by_model.get(name, {})
        rows.append({
            "model": name,
            "best_params": str(res["best_params"]),
            "cv_mean": round(res["best_cv_score"], 4),
            "accuracy": round(m.get("accuracy", np.nan), 4),
            "balanced_accuracy": round(m.get("balanced_accuracy", np.nan), 4),
            "f1_macro": round(m.get("f1_macro", np.nan), 4),
            "roc_auc": round(m["roc_auc"], 4) if m.get("roc_auc") is not None else np.nan,
            "train_time_s": res["train_time_s"],
        })
    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
