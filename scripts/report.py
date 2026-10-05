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
