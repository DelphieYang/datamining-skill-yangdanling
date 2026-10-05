"""Shared utilities: loading, seeding, run-log, and environment recording."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def file_hash(path: str | Path) -> str:
    """Return an MD5 hash of a file so the run can be traced to exact data."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a tabular dataset, dispatching on file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file format: {suffix}")


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def env_info() -> dict:
    """Collect library versions for the run log."""
    import sklearn
    import matplotlib
    import seaborn
    return {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": seaborn.__version__,
    }


def init_run(output_dir: str | Path, dataset_path: str | Path,
             llm_model_name: str, llm_interface: str, seed: int = 42) -> dict:
    """Create output directories and write the initial run log."""
    out = Path(output_dir)
    for sub in ("intermediate", "intermediate/checkpoints"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    run_log = {
        "seed": seed,
        "dataset_path": str(dataset_path),
        "dataset_hash": file_hash(dataset_path),
        "llm_model_name": llm_model_name,
        "llm_interface": llm_interface,
        "environment": env_info(),
    }
    save_json(run_log, out / "run_log.json")
    return run_log
