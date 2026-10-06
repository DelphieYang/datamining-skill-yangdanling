"""Orchestrator for the tabular-classifier skill.

Usage:
    python scripts/run_skill.py \
        --dataset_path path/to/train.csv \
        --test_path path/to/test.csv \
        --target_column label \
        --output_dir outputs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from . import eda, evaluate, preprocessing, report, train, utils
except ImportError:  # run as `python scripts/run_skill.py`
    import eda, evaluate, preprocessing, report, train, utils


def infer_target(df: pd.DataFrame) -> str:
    """Heuristic target-column inference (open to critique).

    Prefers well-known names, then falls back to the LAST categorical column
    with low cardinality. This can misfire on low-cardinality features or
    identifier-like columns.
    """
    known = ["label", "target", "class", "y", "outcome"]
    for name in known:
        if name in df.columns:
            return name
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical:
        return categorical[-1]
    raise ValueError("Could not infer target column; please pass --target_column.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Tabular classifier skill")
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--test_path", default=None)
    p.add_argument("--target_column", default=None)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--report_title", default="Tabular Classification Report")
    p.add_argument("--author", default="")
    p.add_argument("--llm_model_name", default="")
    p.add_argument("--llm_interface", default="")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    utils.set_seed(args.seed)
    utils.init_run(args.output_dir, args.dataset_path,
                   args.llm_model_name, args.llm_interface, args.seed)

    df = utils.load_dataset(args.dataset_path)
    target = args.target_column or infer_target(df)

    # Drop rows with a missing target label (clean, not imputed).
    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    if args.test_path:
        df_test = utils.load_dataset(args.test_path).dropna(subset=[target])
        X_test, y_test = df_test.drop(columns=[target]), df_test[target]
        X_train, y_train = X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=args.seed)

    out = Path(args.output_dir)

    # EDA
    eda_summary = eda.summarize(df, target, out / "intermediate" / "eda_summary.md")

    # Preprocessing
    preprocessor = preprocessing.build_preprocessing(X_train)

    # Training
    model_results = train.train_models(X_train, y_train, preprocessor)

    # Evaluation
    metrics_by_model = {
        name: evaluate.evaluate(res["model"], X_test, y_test)
        for name, res in model_results.items()
    }
    utils.save_json({"models": metrics_by_model}, out / "metrics.json")
    evaluate.compare(model_results, metrics_by_model,
                     out / "intermediate" / "model_comparison.csv")
    evaluate.plot_confusion_matrices(
        metrics_by_model, out / "intermediate" / "confusion_matrix.png")

    # Report
    report.generate_report(
        {"models": metrics_by_model},
        {"report_title": args.report_title,
         "author": args.author,
         "llm_model_name": args.llm_model_name,
         "llm_interface": args.llm_interface,
         "figure": "![Confusion matrices](intermediate/confusion_matrix.png)"},
        out / "report.md",
    )

    # Convert to PDF, degrading gracefully if no converter is installed.
    try:
        report.to_pdf(out / "report.md", out / "report.pdf")
    except Exception as exc:
        print(f"Warning: could not generate report.pdf ({exc}); report.md is available.")

    print(f"Done. Outputs written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
