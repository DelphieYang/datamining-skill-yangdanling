"""Preprocessing: build a leakage-safe pipeline.

NOTE — several defaults here are deliberately simple and open to challenge
(see the Reflection in the assignment):
  * missing numeric values are filled with the column MEAN (not median),
  * outliers are winsorized automatically via the IQR rule,
  * StandardScaler is applied to ALL numeric features regardless of model,
  * one-hot encoding is used for every categorical column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Winsorizer(BaseEstimator, TransformerMixin):
    """Cap outliers to the IQR bounds (factor 1.5)."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_ = None
        self.upper_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        X = np.clip(X, self.lower_, self.upper_)
        return X


def build_preprocessing(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer for the given feature frame."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipe = Pipeline([
        # Flaw #1 + #2: mean imputation then automatic IQR winsorizing.
        ("impute", SimpleImputer(strategy="mean")),
        ("winsorize", Winsorizer()),
        # Flaw #5: scale everything, even for tree models.
        ("scale", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")
