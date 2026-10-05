---
name: tabular-classifier-skill
description: Reusable end-to-end skill for any tabular classification dataset. Load the data, clean and explore it, select features, train and compare at least two classifiers with leakage-safe preprocessing, evaluate with class-appropriate metrics, and generate a concise PDF report.
version: 0.1.0
inputs:
  - name: dataset_path
    required: true
    description: Path to a tabular dataset file (CSV / Excel / Parquet), or a train/test pair.
  - name: test_path
    required: false
    description: Optional path to a held-out test set (CSV / Excel / Parquet).
  - name: target_column
    required: false
    description: Name of the target column. If omitted, the skill infers it and asks for confirmation.
  - name: output_dir
    required: false
    default: ./outputs
  - name: report_title
    required: false
    default: Tabular Classification Report
  - name: author
    required: false
    description: Optional author name shown in the report header.
  - name: llm_model_name
    required: false
    description: Optional name/version of the model used to write the report prose, for transparency.
  - name: llm_interface
    required: false
    description: Optional interface used to write the report, for transparency.
outputs:
  - report.md
  - report.pdf
  - metrics.json
  - run_log.json
  - intermediate/eda_summary.md
  - intermediate/preprocessing_plan.md
  - intermediate/model_comparison.csv
  - intermediate/confusion_matrix.png
---

# Tabular Classifier Skill

## Purpose

This skill automates an end-to-end tabular classification workflow and produces a concise, self-contained report. It is designed to work on **any tabular classification dataset**: it inspects the data, chooses preprocessing and models, justifies those choices, trains and compares at least two classifiers, and emits a report of **no more than two pages**.

The main report covers:

1. Data exploration and cleaning
2. Feature selection / engineering
3. Model training
4. Evaluation and comparison
5. Findings and discussion

---

## Non-Goals

- Do not hard-code assumptions about a specific dataset.
- Do not optimize only for accuracy.
- Do not use test data for preprocessing, feature selection, or hyperparameter tuning.
- Do not fabricate metrics, model names, versions, or references.
- Do not skip human review checkpoints.

---

## Workflow

### 0. Initialize Run

- Create `output_dir` and set random seeds for reproducibility.
- Record environment: Python version, key library versions, dataset path + file hash, and any report metadata.
- Write `run_log.json`.

### 1. Load Dataset

- Detect format (`.csv`, `.xlsx`, `.parquet`) and load with the matching reader.
- Report shape, columns, dtypes, and a head sample.
- If a separate test file is given, use it as the held-out test set; otherwise split one from the loaded data.

### 2. Infer Task and Target Column

- If `target_column` is given, use it.
- Otherwise infer it by name (e.g. `label`, `target`, `class`, `y`) and, failing that, by **low cardinality + position + dtype** (categorical columns with few unique values, especially the last column, are assumed to be the target).
- Determine binary vs multiclass. If the task looks like regression, stop and warn.

### 3. Exploratory Data Analysis

Write `intermediate/eda_summary.md` covering: shape, numeric/categorical/datetime split, missing values, duplicates, class distribution and imbalance ratio, summary statistics, cardinality, and correlation with the target.

### 4. Cleaning and Preprocessing

- **Missing numeric values**: fill with the **mean** of the column.
- **Missing categorical values**: fill with the most frequent value.
- **Outliers**: detect with the IQR rule and **cap (winsorize) them automatically**.
- **Encoding**: one-hot for low-cardinality categoricals.
- **Scaling**: apply `StandardScaler` to all numeric features regardless of model choice.
- **Class imbalance**: apply `class_weight='balanced'` where supported.
- All preprocessing is fit on training folds only, to avoid leakage.

Write `intermediate/preprocessing_plan.md`.

### 5. Feature Selection / Engineering

- Drop ID-like and timestamp columns to avoid leakage.
- Use variance threshold to drop near-constant columns.
- Create date-part features from datetime columns if present; otherwise state that no engineering was needed and why.

### 6. Model Selection

Train and compare **two** classifiers by default: **Logistic Regression** and **Random Forest**, chosen as a fixed default pair representing a linear and a non-linear model. Write the model plan to `intermediate/model_plan.md`.

### 7. Training and Hyperparameter Tuning

- Use stratified K-fold cross-validation.
- Tune with `GridSearchCV` over a small grid (e.g. `C` for logistic regression; `n_estimators`, `max_depth` for random forest).
- Record best parameters, CV scores, and training time.

### 8. Evaluation

Evaluate on the held-out test set and report: **accuracy first**, then balanced accuracy, precision/recall/F1 (macro and weighted), ROC-AUC / PR-AUC for binary, and a confusion matrix. Save everything to `metrics.json`. Generate a confusion-matrix plot for each model (plus an ROC curve for binary classification) and save it to `intermediate/confusion_matrix.png`.

### 9. Model Comparison

Write `intermediate/model_comparison.csv` and discuss which model won, why, and the interpretability/speed/performance trade-offs.

### 10. Generate Report

Fill `report.md` from `metrics.json` and the intermediate outputs, embedding the confusion-matrix figure in the results, then convert to `report.pdf`. Body must fit within two pages; keep at most 1–2 small figures.

### 11. Final Validation

Confirm `report.pdf` exists, is ≤ 2 pages, metrics match `metrics.json`, and at least two models were compared.

---

## Human-in-the-Loop Checkpoints

The skill writes intermediate files at each stage so a human can review and edit them:

1. `intermediate/eda_summary.md` — dataset profile and issues found.
2. `intermediate/preprocessing_plan.md` — what was cleaned and why.
3. `intermediate/model_plan.md` — models and metrics chosen.
4. `intermediate/model_comparison.csv` — comparison results.

The human is expected to read these and can intervene before the final PDF is produced.

---

## Report Template

```markdown
# {report_title}

Author: {author}
Model used: {llm_model_name}
Interface: {llm_interface}

## INTRODUCTION
...

## METHODS OR PROCEDURES
**Data exploration and cleaning.** ...
**Feature selection and engineering.** ...
**Model training.** ...

## RESULTS
...

## DISCUSSION
...

## CONCLUSION
...

## REFERENCES
[1] ...
```

---

## Decision Policies

- If the target column cannot be inferred confidently, stop and ask the human.
- If the task is regression, stop (this skill targets classification tasks).
- If classes are imbalanced, also report balanced accuracy, macro F1, PR-AUC, and the confusion matrix.
- If the dataset is small, prefer simple models and cross-validation.
- If a model errors during training, log the failure and continue with the other.
- If the report exceeds two pages, trim the discussion and move detail to the intermediate files.

---

## Failure Handling

- Missing file → stop with a clear error.
- Unsupported format → try alternate pandas readers, else ask the human.
- No target column → ask the human.
- PDF conversion fails → keep `report.md` and report the conversion command.

---

## Quality Checklist

- [ ] Dataset path accepted as input.
- [ ] Target column inferred or confirmed; task confirmed as classification.
- [ ] Missing values, outliers, duplicates, and imbalance examined.
- [ ] Preprocessing justified and leakage-safe.
- [ ] Feature selection/engineering explained or explicitly skipped.
- [ ] At least two models trained and compared.
- [ ] Hyperparameter tuning described.
- [ ] Metrics appropriate for class balance.
- [ ] Report body fits two pages.
- [ ] All numbers traceable to `metrics.json`.
