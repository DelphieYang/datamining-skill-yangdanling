# Tabular Classifier Skill

A reusable **SKILL** for an end-to-end tabular classification workflow with automated report generation. Built for **IN6227 Data Mining — Assignment 1, Variant 2**.

The skill takes a **dataset path as input** and, on its own, explores the data, selects preprocessing and models, trains and compares at least two classifiers, and produces a **≤2-page PDF report** plus a separate **Reflection** draft.

## What it does

1. Loads any tabular dataset (CSV / Excel / Parquet).
2. Infers the target column and confirms the task is classification.
3. Runs exploratory data analysis (missing values, outliers, class imbalance, leakage).
4. Applies leakage-safe preprocessing (imputation, encoding, scaling).
5. Trains and tunes two classifiers (Logistic Regression + Random Forest).
6. Evaluates with class-appropriate metrics and compares the models.
7. Generates `report.md` → `report.pdf` and a `reflection.md` draft.

## Requirements

```bash
pip install -r requirements.txt
```

PDF conversion uses `pandoc` + `xelatex` (optional; install with `brew install pandoc`).

## Usage

```bash
python scripts/run_skill.py \
  --dataset_path data/train.csv \
  --target_column label \
  --output_dir outputs \
  --llm_model_name claude-haiku-4-5 \
  --llm_interface "Claude Code 2.1.24" \
  --github_link https://github.com/<you>/<repo>
```

If `--target_column` is omitted, the skill infers it and the human should confirm.

## Outputs

```
outputs/
├── report.md              # main report (≤2 pages)
├── report.pdf
├── reflection.md          # draft for the human to edit
├── metrics.json
├── run_log.json
└── intermediate/
    ├── eda_summary.md
    ├── preprocessing_plan.md
    ├── model_comparison.csv
    └── checkpoints/
```

## Human-in-the-loop

The skill writes intermediate files at each stage. Review them and intervene before
the final PDF is produced — this is deliberate, not an error.

## Notes on design decisions

Several defaults are intentionally simple and open to challenge — e.g. mean imputation,
automatic IQR winsorizing, a fixed model pair, and reporting accuracy first. These are
documented in the code and are the intended subjects of the assignment's **Reflection**
section.
