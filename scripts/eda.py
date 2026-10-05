"""Exploratory data analysis: profile a dataframe and write a summary."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _categorical_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def summarize(df: pd.DataFrame, target: str, output_path: str | Path) -> dict:
    """Produce a summary dict and write it to a markdown file."""
    n_rows, n_cols = df.shape
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = _categorical_cols(df)
    datetime = df.select_dtypes(include=["datetime"]).columns.tolist()

    missing = df.isna().sum()
    missing = missing[missing > 0]
    duplicates = int(df.duplicated().sum())

    target_counts = df[target].value_counts(dropna=False)
    imbalance_ratio = float(target_counts.max() / target_counts.min()) if len(target_counts) > 1 else np.nan

    summary = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "numeric_cols": numeric,
        "categorical_cols": categorical,
        "datetime_cols": datetime,
        "missing": {k: int(v) for k, v in missing.items()},
        "duplicates": duplicates,
        "target_distribution": {str(k): int(v) for k, v in target_counts.items()},
        "imbalance_ratio": imbalance_ratio,
    }

    # Markdown report
    lines = [f"# EDA Summary\n", f"- Rows: {n_rows}, Columns: {n_cols}",
             f"- Numeric: {len(numeric)}, Categorical: {len(categorical)}, Datetime: {len(datetime)}",
             f"- Duplicate rows: {duplicates}",
             f"- Imbalance ratio (max/min class): {imbalance_ratio:.2f}",
             "\n## Missing values\n"]
    if len(missing):
        for col, cnt in missing.items():
            lines.append(f"- {col}: {cnt} ({cnt / n_rows:.1%})")
    else:
        lines.append("- None")

    lines.append("\n## Target distribution\n")
    for k, v in target_counts.items():
        lines.append(f"- {k}: {v}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines))

    return summary
