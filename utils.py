# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ACTL3143)
#     language: python
#     name: actl3143
# ---

# %%
"""
Minimal project utilities for Capstone-Seminar-Paper.

Scope (intentionally minimal):
- Project root detection (env override supported)
- Directory helpers:
  data/raw, data/processed, artifacts/{figures,tables,meta}, models, report, notebooks, tuner_dir
- Save helpers: CSV (DataFrame), JSON, Pickle, text, bytes, matplotlib figures

No logging, no timestamps, no data cleaning or grouping tools.
"""

from __future__ import annotations
import os
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd



# =========================
# Root detection
# =========================

def _detect_project_root() -> Path:
    """
    Detect project root. Priority:
    1) env CAPSTONE_ROOT
    2) /Users/audreychang/Projects/Capstone-Seminar-Paper (Audrey's default)
    3) search upwards from cwd for a folder containing 'data' and 'notebooks'
    4) fallback to cwd
    """
    env = os.environ.get("CAPSTONE_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p

    cur = Path.cwd().resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "data").exists() and ((parent / "scripts").exists() or (parent / "report").exists()):
            return parent
    return cur


ROOT: Path = _detect_project_root()

# =========================
# Directory schema
# =========================

@dataclass(frozen=True)
class Dirs:
    root: Path = ROOT
    data: Path = ROOT / "data"
    raw: Path = ROOT / "data" / "raw"
    processed: Path = ROOT / "data" / "processed"
    artifacts: Path = ROOT / "artifacts"
    figures: Path = ROOT / "artifacts" / "figures"
    tables: Path = ROOT / "artifacts" / "tables"
    meta: Path = ROOT / "artifacts" / "meta"
    models: Path = ROOT / "models"
    notebooks: Path = ROOT / "notebooks"
    report: Path = ROOT / "report"
    tuner: Path = ROOT / "tuner_dir"

DIRS = Dirs()

def ensure_dirs() -> None:
    """Create all common project folders if missing."""
    for p in [
        DIRS.data, DIRS.raw, DIRS.processed,
        DIRS.artifacts, DIRS.figures, DIRS.tables, DIRS.meta,
        DIRS.models, DIRS.notebooks, DIRS.report, DIRS.tuner,
    ]:
        p.mkdir(parents=True, exist_ok=True)

ensure_dirs()

# =========================
# Path helpers (return Path)
# =========================

def path_in(*parts: str | Path) -> Path:
    return DIRS.root.joinpath(*parts)

def raw_path(name: str) -> Path:
    return DIRS.raw / name

def processed_path(name: str) -> Path:
    return DIRS.processed / name

def figure_path(name: str) -> Path:
    """Return artifacts/figures/<name>. If no extension, default to .png."""
    if "." not in name:
        name = f"{name}.png"
    return DIRS.figures / name

def table_path(name: str) -> Path:
    """Return artifacts/tables/<name>.csv if no extension given."""
    if not name.lower().endswith(".csv"):
        name = f"{name}.csv"
    return DIRS.tables / name

def meta_path(name: str) -> Path:
    return DIRS.meta / name

def manifest_path(run_name: str) -> Path:
    """Return manifest path for a given run."""
    return DIRS.meta / f"manifest_{run_name}.json"

def model_path(name: str) -> Path:
    return DIRS.models / name

def report_path(name: str) -> Path:
    return DIRS.report / name

def notebook_path(name: str) -> Path:
    return DIRS.notebooks / name

def tuner_path(name: str) -> Path:
    return DIRS.tuner / name



# ======== APPENDIX: Manifest writing =================
# Added near end of utils.py
from pathlib import Path
import json

_RUN_LOG = {"figures": [], "tables": []}

def _log_figure(path: Path):
    _RUN_LOG["figures"].append(path.name)

def _log_table(path: Path):
    _RUN_LOG["tables"].append(path.name)

def reset_run_log():
    """Clear the run log figures/tables (for run_all)."""
    _RUN_LOG["figures"].clear()
    _RUN_LOG["tables"].clear()


def write_manifest(run_name: str, core_figures=None, core_tables=None, reset: bool = True):
    core_figures = set(core_figures or [])
    core_tables = set(core_tables or [])

    manifest = {
        "run_name": run_name,
        "figures_main": [f for f in _RUN_LOG["figures"] if f in core_figures],
        "figures_appendix": [f for f in _RUN_LOG["figures"] if f not in core_figures],
        "tables_main": [t for t in _RUN_LOG["tables"] if t in core_tables],
        "tables_appendix": [t for t in _RUN_LOG["tables"] if t not in core_tables],
        "all_figures": _RUN_LOG["figures"],
        "all_tables": _RUN_LOG["tables"],
    }

    out_dir = DIRS.meta
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"manifest_{run_name}.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== Manifest ===")
    print("Saved to:", out_path)
    print("Figures (main):", manifest["figures_main"])
    print("Figures (appendix):", manifest["figures_appendix"])
    print("Tables  (main):", manifest["tables_main"])
    print("Tables  (appendix):", manifest["tables_appendix"])
    if reset:
        reset_run_log()








# =========================
# Save helpers
# =========================

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def save_df(df: pd.DataFrame, name: str, folder: Optional[Path] = None, index: bool = False, log_table: bool = True) -> Path:
    """
    Save DataFrame:
    - If extension is .parquet, save as parquet (preserve dtypes)
    - Otherwise save as CSV.
    Default folder: artifacts/tables.
    
    [WARN] Ensure folder exists
    """
    folder = folder or DIRS.tables
    folder.mkdir(parents=True, exist_ok=True)  # [OK] Critical line

    # Build full path
    p = folder / name

    # If no suffix is provided, default to .csv
    if p.suffix == "":
        p = p.with_suffix(".csv")

    # Choose format by suffix
    if p.suffix.lower() == ".parquet":
        df.to_parquet(p, index=index)
    else:
        df.to_csv(p, index=index)

    if log_table:
        _log_table(p)    # [OK] Log after save, before print/return
    print(f"[save_df] Saved: {p}")
    return p


def save_json(obj, name: str | Path, folder: Optional[Path] = None, ensure_ascii: bool = False, indent: int = 2) -> Path:
    """
    Save JSON with UTF-8 encoding.
    Default folder: artifacts/meta.
    """
    if isinstance(name, Path):
        p = name
    else:
        folder = folder or DIRS.meta
        p = folder / (name if name.lower().endswith(".json") else f"{name}.json")
    _ensure_parent(p)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)
    return p

def save_pickle(obj, name: str, folder: Optional[Path] = None) -> Path:
    """
    Save Pickle binary.
    Default folder: models (fix: previously meta)
    
    [WARN] Fix: models should be stored in models/, not meta/
    """
    # [FIX] Default folder changed from DIRS.meta to DIRS.models
    folder = folder or DIRS.models  # [OK] Updated default
    
    p = folder / (name if name.lower().endswith(".pkl") else f"{name}.pkl")
    _ensure_parent(p)
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    print(f"[save_pickle] Saved: {p}")  #  
    return p

def load_pickle(name: str | Path, folder: Optional[Path] = None):
    """
    Load Pickle binary.
    Accepts either a full Path or a file name (defaults to models/).
    """
    if isinstance(name, Path):
        path = name
    else:
        folder = folder or DIRS.models
        filename = name if name.lower().endswith(".pkl") else f"{name}.pkl"
        path = folder / filename

    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    joblib = None
    try:
        import joblib as _joblib  # type: ignore
        joblib = _joblib
    except ImportError:
        joblib = None

    # Prefer joblib.load because many models were saved via joblib.dump (handles zip format)
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass

    # Fallback to plain pickle.load
    with open(path, "rb") as f:
        return pickle.load(f)

def save_text(text: str, name: str, folder: Optional[Path] = None, encoding: str = "utf-8") -> Path:
    """
    Save plain text file.
    Default folder: artifacts/meta.
    """
    folder = folder or DIRS.meta
    p = folder / (name if name.lower().endswith(".txt") else f"{name}.txt")
    _ensure_parent(p)
    with open(p, "w", encoding=encoding) as f:
        f.write(text)
    return p

def save_bytes(data: bytes, name: str, folder: Optional[Path] = None) -> Path:
    """
    Save raw bytes (e.g., images already in bytes).
    Default folder: artifacts/meta.
    """
    folder = folder or DIRS.meta
    p = folder / name
    _ensure_parent(p)
    with open(p, "wb") as f:
        f.write(data)
    return p

def save_fig(fig, name: str, dpi: int = 200, tight: bool = True) -> Path:
    """
    Save a matplotlib figure under artifacts/figures.
    If no extension in `name`, PNG is used.
    """
    p = figure_path(name)
    _ensure_parent(p)
    if tight:
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(p, dpi=dpi)

    _log_figure(p)   # [OK] Log after save, before return
    return p

from typing import Union
from pathlib import Path
import json

# =========================
# Load helpers
# =========================

def load_json(name: Union[str, Path], folder: Optional[Path] = None):
    """
    Load JSON file.

    Usage 1: pass name (recommended)
        load_json("preprocess_params")
        load_json("preprocess_params.json")

    Usage 2: pass full Path (advanced)
        load_json(meta_path("preprocess_params.json"))

    folder:
        - if name is str: use folder or DIRS.meta
        - if name is Path: ignore folder and use this Path
    """
    # If a Path is given, use it directly
    if isinstance(name, Path):
        path = name
    else:
        folder = folder or DIRS.meta
        filename = name if name.lower().endswith(".json") else f"{name}.json"
        path = folder / filename

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# Model result logging helpers
# =========================
def log_model_result(
    model_name: str,
    params: Optional[dict] = None,
    metrics: Optional[dict] = None,
    notes: Optional[str] = None,
    filename: str = "model_results.csv",
):
    """
    Append each experiment result to artifacts/meta/model_results.csv.

    Parameters
    ----------
    model_name : str
        Model or experiment name, e.g. "logit_baseline_v1".
    params : dict, optional
        Hyperparameters, e.g. GridSearchCV best_params_.
    metrics : dict, optional
        Flattened metrics, e.g.
        {"auc_train": 0.61, "auc_val": 0.60, "brier_val": 0.24}
    notes : str, optional
        Notes, e.g. "with class_weight=balanced".
    filename : str, optional
        Output filename, default "model_results.csv".
    """
    import datetime
    import pandas as pd

    params = params or {}
    metrics = metrics or {}

    # 1. Build one row
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
    }

    # Prefix all hyperparams with param__
    for k, v in params.items():
        row[f"param__{k}"] = v

    # Prefix all metrics with metric__
    for k, v in metrics.items():
        row[f"metric__{k}"] = v

    if notes is not None:
        row["notes"] = notes

    df = pd.DataFrame([row])

    # 2. Output path: artifacts/meta/model_results.csv
    p = meta_path(filename)
    _ensure_parent(p)  # [OK] Ensure folder exists

    # 3. If file exists, append without header; else create new
    if p.exists():
        df.to_csv(p, mode="a", header=False, index=False)
    else:
        df.to_csv(p, mode="w", header=True, index=False)

    print(f"[log_model_result] Appended 1 row to {p}")
    return p




# =========================
# sklearn custom transformers 
# =========================

class MedianFromConfigImputer(BaseEstimator, TransformerMixin):
    """
    Use numeric_impute_values from preprocess_params.json for imputation,
    do not recompute medians from data.

    impute_values: dict, key is column name, value is median
    feature_names: list, optional, expected feature order for this transformer
    """
    def __init__(self, impute_values: dict, feature_names=None):
        self.impute_values = impute_values
        self.feature_names = feature_names

    def fit(self, X, y=None):
        # Keep column order for transform/get_feature_names_out
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        elif self.feature_names is not None:
            self.feature_names_in_ = list(self.feature_names)
        else:
            self.feature_names_in_ = [f"col_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        # Normalize to DataFrame for processing
        if hasattr(X, "columns"):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        for col in self.feature_names_in_:
            if col in self.impute_values:
                X_df[col] = X_df[col].fillna(self.impute_values[col])

        # Convert back to original type
        if hasattr(X, "columns"):
            return X_df
        else:
            return X_df.values

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return np.asarray(input_features, dtype=object)


class ToFloatTransformer(BaseEstimator, TransformerMixin):
    """
    Cast inputs to float for 0/1 flag columns,
    and keep feature_names_out unchanged.
    """
    def fit(self, X, y=None):
        self.feature_names_in_ = getattr(X, "columns", None)
        return self

    def transform(self, X):
        return X.astype(float)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        return np.asarray(input_features, dtype=object)






# =========================
# Test code (optional)
# =========================

if __name__ == "__main__":
    """
    Quick smoke test for utils
    """
    print("="*80)
    print("Testing utils.py")
    print("="*80)
    
    # Test paths
    print("\n1. Test path helpers:")
    print(f"  ROOT: {ROOT}")
    print(f"  DIRS.models: {DIRS.models}")
    print(f"  DIRS.meta: {DIRS.meta}")
    
    # Test save/load JSON
    print("\n2. Test JSON helpers:")
    test_data = {"test": "data", "value": 123}
    json_path = save_json(test_data, "test_utils", folder=DIRS.meta)
    print(f"  Saved:  {json_path}")
    
    loaded_data = load_json("test_utils", folder=DIRS.meta)
    print(f"  Loaded:  {loaded_data}")
    assert loaded_data == test_data, "JSON test failed!"
    print("  [OK] JSON test passed")
    
    # Test save_df
    print("\n3. Test DataFrame helpers:")
    import pandas as pd
    test_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df_path = save_df(test_df, "test_utils.csv", folder=DIRS.tables)
    print(f"  Saved:  {df_path}")
    print("  [OK] DataFrame test passed")
    
    # Test log_model_result
    print("\n4. Test log_model_result:")
    log_model_result(
        model_name="test_model",
        params={"param1": 0.1, "param2": 100},
        metrics={"auc_val": 0.85, "brier_val": 0.15},
        notes="test model"
    )
    print("  [OK] log_model_result test passed")
    
    print("\n" + "="*80)
    print("[OK] All tests passed!")
    print("="*80)





# =========================
# Public API
# =========================
__all__ = [
    "ROOT", "DIRS", "ensure_dirs",
    "path_in", "raw_path", "processed_path",
    "figure_path", "table_path", "meta_path", "manifest_path", "model_path", "report_path", "notebook_path", "tuner_path",
    "save_df", "save_json", "save_pickle", "save_text", "save_bytes", "save_fig",
    "load_json",  
    "MedianFromConfigImputer", "ToFloatTransformer",
    "log_model_result", "write_manifest", "reset_run_log"
]
