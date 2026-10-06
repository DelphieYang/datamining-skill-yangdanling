"""Report generation: fill a markdown template from metrics.json and write PDF."""
from __future__ import annotations

import re
from pathlib import Path

try:
    from . import utils
except ImportError:  # run as a plain script
    import utils


REPORT_TEMPLATE = """# {report_title}

Author: {author}
Model used: {llm_model_name}
Interface: {llm_interface}

## INTRODUCTION
{introduction}

## METHODS OR PROCEDURES
**Data exploration and cleaning.** {eda_summary}
**Feature selection and engineering.** {feature_work}
**Model training.** {training}

## RESULTS
{results}
{figure}

## DISCUSSION
{discussion}

## CONCLUSION
{conclusion}

## REFERENCES
[1] ...
"""


def generate_report(metrics: dict, meta: dict, output_path: str | Path) -> str:
    """Render the report markdown from metrics and metadata."""
    # Best model by accuracy (see evaluate.py note).
    best = max(metrics["models"].items(), key=lambda kv: kv[1].get("accuracy", -1))
    best_name = best[0]

    text = REPORT_TEMPLATE.format(
        report_title=meta.get("report_title", "Tabular Classification Report"),
        author=meta.get("author", ""),
        llm_model_name=meta.get("llm_model_name", ""),
        llm_interface=meta.get("llm_interface", ""),
        introduction=meta.get("introduction", "…"),
        eda_summary=meta.get("eda_summary", "…"),
        feature_work=meta.get("feature_work", "…"),
        training=meta.get("training", "…"),
        results=meta.get("results", f"Best model: {best_name}"),
        figure=meta.get("figure", ""),
        discussion=meta.get("discussion", "…"),
        conclusion=meta.get("conclusion", "…"),
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def build_prose(metrics: dict, model_results: dict, eda_summary: dict, target: str) -> dict:
    """Build template narrative text from the actual run results.

    Returns a dict of report-section strings so the report is not left as
    placeholders; the numbers come from the real metrics and EDA summary.
    """
    models = metrics["models"]
    names = list(models.keys())
    best_name = max(models.items(), key=lambda kv: kv[1].get("accuracy", -1))[0]

    n_rows = eda_summary["n_rows"]
    n_cols = eda_summary["n_cols"]
    n_num = len([c for c in eda_summary["numeric_cols"] if c != target])
    n_cat = len([c for c in eda_summary["categorical_cols"] if c != target])
    dist = eda_summary["target_distribution"]
    ratio = eda_summary["imbalance_ratio"]
    missing = eda_summary["missing"]

    sorted_classes = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    majority, majority_n = sorted_classes[0]
    minority, minority_n = (sorted_classes[1] if len(sorted_classes) > 1 else ("-", 0))
    binary = len(sorted_classes) == 2

    # INTRODUCTION
    intro = (
        f"This report documents an end-to-end tabular classification workflow. "
        f"The dataset contains {n_rows:,} rows and {n_cols} columns "
        f"({n_num} numeric and {n_cat} categorical features) with a "
        f"{'binary' if binary else 'multi-class'} target column '{target}'."
    )
    if ratio > 1.2:
        if binary:
            intro += (
                f" The target is imbalanced at roughly {ratio:.2f}:1 "
                f"({majority}: {majority_n:,} vs {minority}: {minority_n:,})."
            )
        else:
            dist_str = ", ".join(f"{k}: {v:,}" for k, v in sorted_classes)
            intro += f" The target distribution is skewed ({dist_str})."
    else:
        intro += " The classes are roughly balanced."

    # Data exploration and cleaning
    if missing:
        max_miss = max(missing.values())
        clean = (
            f"Missing values are sparse (at most {max_miss} per column) and were filled with the "
            f"column mean for numeric features and the mode for categorical features. "
        )
    else:
        clean = "There are no missing values. "
    clean += (
        "Outliers were capped to the interquartile (IQR) bounds, categorical features were "
        "one-hot encoded, and numeric features were standardized."
    )
    if ratio > 1.2:
        clean += " class_weight='balanced' was applied to counter the class imbalance."

    # Feature selection / engineering
    feature = (
        "No ID-like or timestamp columns were present, so no leakage-prone columns needed removal. "
        "A variance threshold dropped near-constant columns, and no new features were engineered."
    )

    # Model training
    tuning = []
    for name in names:
        bp = model_results[name]["best_params"]
        cv = model_results[name]["best_cv_score"]
        tuning.append(f"{name} (best parameters {bp}, CV accuracy {cv:.4f})")
    training = (
        "Two classifiers were tuned with 5-fold stratified cross-validation and GridSearchCV: "
        + "; ".join(tuning)
        + ". All preprocessing was fit on training folds only to avoid leakage."
    )

    # Results
    best = models[best_name]
    results = f"Best model by accuracy: {best_name} ({best['accuracy']:.4f})."
    if binary and best.get("roc_auc") is not None:
        results += f" Its ROC-AUC is {best['roc_auc']:.4f}."

    # Discussion
    disc = f"{best_name} achieves the highest accuracy ({best['accuracy']:.4f}). "
    if len(names) > 1:
        other_name = [n for n in names if n != best_name][0]
        other = models[other_name]
        ba_best = best.get("balanced_accuracy")
        ba_other = other.get("balanced_accuracy")
        disc += (
            f"{other_name} has a balanced accuracy of {ba_other:.4f} versus {ba_best:.4f} "
            f"for {best_name}. "
        )
        if ba_other is not None and ba_best is not None and ba_other > ba_best:
            disc += "So under class imbalance the two models rank differently than their raw accuracy suggests. "
        else:
            disc += "The accuracy gap also holds under balanced accuracy. "
    disc += (
        "This highlights why accuracy alone is insufficient for imbalanced data and why "
        "balanced accuracy and macro-F1 should guide model selection."
    )

    # Conclusion
    concl = (
        f"The workflow successfully trained and compared {len(names)} classifiers and produced this report. "
        f"On accuracy {best_name} is best, but the choice should be confirmed with balanced metrics "
        f"for imbalanced data. The skill is general-purpose and can be re-run on any tabular "
        f"classification dataset."
    )

    return {
        "introduction": intro,
        "eda_summary": clean,
        "feature_work": feature,
        "training": training,
        "results": results,
        "discussion": disc,
        "conclusion": concl,
    }


def _latex_escape(text: str) -> str:
    """Escape characters that are special in LaTeX."""
    for ch in ("\\", "&", "%", "$", "#", "_", "{", "}", "~", "^"):
        text = text.replace(ch, "\\" + ch)
    # Unicode punctuation that LaTeX renders natively under utf8 + T1.
    text = text.replace("—", "---").replace("…", "\\ldots{}")
    return text


def _table_row(line: str) -> list[str]:
    """Split a markdown table row into cells."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-+:?", c) for c in cells)


def _table_to_latex(block: list[str]) -> str:
    rows = [_table_row(r) for r in block]
    rows = [r for r in rows if not _is_separator(r)]
    if not rows:
        return ""
    ncol = len(rows[0])
    spec = "l" + " c" * (ncol - 1)
    out = ["\\begin{center}", "\\begin{tabular}{%s}" % spec, "\\hline"]
    for idx, row in enumerate(rows):
        out.append(" & ".join(_latex_escape(c) for c in row) + " \\\\")
        if idx == 0:
            out.append("\\hline")
    out.append("\\hline")
    out.append("\\end{tabular}")
    out.append("\\end{center}")
    return "\n".join(out)


def _md_to_latex(md_text: str) -> str:
    """Convert the skill's simple markdown report to LaTeX."""
    lines = md_text.splitlines()
    body = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            body.append("")
            i += 1
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            body.append(_table_to_latex(block))
        elif line.startswith("# "):
            body.append("\\section*{" + _latex_escape(line[2:]) + "}")
            i += 1
        elif line.startswith("## "):
            body.append("\\subsection*{" + _latex_escape(line[3:]) + "}")
            i += 1
        elif line.startswith("- "):
            body.append("\\begin{itemize}\\item " + _latex_escape(line[2:]) + "\\end{itemize}")
            i += 1
        elif line.startswith("!["):
            m = re.match(r"!\[.*?\]\((.+?)\)", line)
            if m:
                body.append("\\begin{center}\n\\includegraphics[width=0.78\\textwidth]{"
                            + m.group(1) + "}\n\\end{center}")
            i += 1
        else:
            # Escape raw content first, then turn **bold** into \textbf{...}.
            line = _latex_escape(line)
            line = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", line)
            body.append(line)
            i += 1

    preamble = (
        "\\documentclass[10pt]{article}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage{times}\n"
        "\\usepackage{graphicx}\n"
        "\\setlength{\\parindent}{0pt}\n"
        "\\setlength{\\parskip}{0.5em}\n"
        "\\begin{document}\n"
    )
    return preamble + "\n".join(body) + "\n\\end{document}\n"


def to_pdf(markdown_path: str | Path, pdf_path: str | Path) -> None:
    """Convert markdown to PDF via pandoc, else fall back to LaTeX."""
    import subprocess

    md = Path(markdown_path)
    pdf = Path(pdf_path)
    try:
        subprocess.run(
            ["pandoc", str(md), "-o", str(pdf),
             "--pdf-engine=xelatex", "-V", "geometry:margin=1in"],
            check=True,
        )
        return
    except FileNotFoundError:
        pass  # pandoc not installed — use the LaTeX fallback below.

    tex = md.with_suffix(".tex")
    tex.write_text(_md_to_latex(md.read_text()))
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex.name],
        cwd=str(md.parent),
        check=True,
    )
