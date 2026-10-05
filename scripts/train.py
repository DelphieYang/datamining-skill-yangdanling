"""Model training: a fixed default pair plus hyperparameter tuning.

Trains Logistic Regression and Random Forest and tunes them with GridSearchCV.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def _default_models() -> dict[str, tuple[Any, dict]]:
    """Return the fixed (estimator, param_grid) pairs."""
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42, class_weight="balanced"),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10]},
        ),
    }


def train_models(X_train: np.ndarray, y_train: np.ndarray,
                 preprocessor) -> dict[str, dict]:
    """Train each model inside a pipeline and tune it with GridSearchCV."""
    results = {}
    for name, (estimator, param_grid) in _default_models().items():
        pipe = Pipeline([("prep", preprocessor), ("clf", estimator)])
        search = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)

        start = time.time()
        search.fit(X_train, y_train)
        elapsed = time.time() - start

        results[name] = {
            "best_params": search.best_params_,
            "best_cv_score": float(search.best_score_),
            "train_time_s": round(elapsed, 2),
            "model": search.best_estimator_,
        }
    return results
