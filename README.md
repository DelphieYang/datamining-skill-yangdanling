# Tabular Classifier Skill

A reusable **SKILL** for an end-to-end tabular classification workflow with automated report generation.

The skill takes a **dataset path as input** and, on its own, explores the data, selects preprocessing and models, trains and compares at least two classifiers, and produces a **≤2-page PDF report**.

## What it does

1. Loads any tabular dataset (CSV / Excel / Parquet).
2. Infers the target column and confirms the task is classification.
3. Runs exploratory data analysis (missing values, outliers, class imbalance, leakage).
4. Applies leakage-safe preprocessing (imputation, encoding, scaling).
5. Trains and tunes two classifiers (Logistic Regression + Random Forest).
6. Evaluates with class-appropriate metrics and compares the models.
7. Generates `report.md` → `report.pdf`.

## Requirements

```bash
pip install -r requirements.txt
```

PDF conversion uses `pandoc` + `xelatex` (optional; the script falls back to a
built-in LaTeX path if `pandoc` is missing).

## Usage

```bash
python scripts/run_skill.py \
  --dataset_path path/to/train.csv \
  --test_path path/to/test.csv \
  --target_column label \
  --output_dir outputs \
  --report_title "My Classification Report" \
  --author "Your Name" \
  --llm_model_name "some-model-4.5" \
  --llm_interface "API / ChatUI / Agent Harness"
```

All arguments except `--dataset_path` are optional. If `--target_column` is omitted,
the skill infers it and the human should confirm.

## Outputs

```
outputs/
├── report.md              # main report (≤2 pages)
├── report.pdf
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

Several defaults are intentionally simple — mean imputation, automatic IQR
winsorizing, a fixed model pair, and accuracy reported first. They favor clarity
over complexity; adjust them in `scripts/preprocessing.py` and `scripts/train.py`.
