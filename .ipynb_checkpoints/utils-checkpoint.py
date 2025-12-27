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

    default = Path("/Users/audreychang/Projects/Capstone-Seminar-Paper")
    if default.exists():
        return default.resolve()

    cur = Path.cwd().resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "data").exists() and (parent / "notebooks").exists():
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

def model_path(name: str) -> Path:
    return DIRS.models / name

def report_path(name: str) -> Path:
    return DIRS.report / name

def notebook_path(name: str) -> Path:
    return DIRS.notebooks / name

def tuner_path(name: str) -> Path:
    return DIRS.tuner / name

# =========================
# Save helpers
# =========================

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def save_df(df: pd.DataFrame, name: str, folder: Optional[Path] = None, index: bool = False) -> Path:
    """
    儲存 DataFrame：
    - 若副檔名為 .parquet → 以 parquet 格式儲存（保留型別資訊）
    - 否則以 CSV 格式儲存。
    預設資料夾：artifacts/tables。
    
    ⚠️ 重要：確保資料夾存在
    """
    folder = folder or DIRS.tables
    folder.mkdir(parents=True, exist_ok=True)  # ✅ 這行最重要！

    # 組完整路徑
    p = folder / name

    # 若沒指定副檔名，預設加上 .csv
    if p.suffix == "":
        p = p.with_suffix(".csv")

    # 根據副檔名決定格式
    if p.suffix.lower() == ".parquet":
        df.to_parquet(p, index=index)
    else:
        df.to_csv(p, index=index)

    print(f"[save_df] 已儲存：{p}")
    return p


def save_json(obj, name: str, folder: Optional[Path] = None, ensure_ascii: bool = False, indent: int = 2) -> Path:
    """
    Save JSON with UTF-8 encoding.
    Default folder: artifacts/meta.
    """
    folder = folder or DIRS.meta
    p = folder / (name if name.lower().endswith(".json") else f"{name}.json")
    _ensure_parent(p)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)
    return p

def save_pickle(obj, name: str, folder: Optional[Path] = None) -> Path:
    """
    Save Pickle binary.
    Default folder: models (修正：原本是 meta)
    
    ⚠️ 重要修正：模型應存放在 models/ 而非 meta/
    """
    # 🔧 修正：預設資料夾從 DIRS.meta 改為 DIRS.models
    folder = folder or DIRS.models  # ✅ 修正後
    
    p = folder / (name if name.lower().endswith(".pkl") else f"{name}.pkl")
    _ensure_parent(p)
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    print(f"[save_pickle] 已儲存：{p}")  # 🔧 加入日誌
    return p

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
    return p

from typing import Optional, Union
from pathlib import Path
import json

# =========================
# Load helpers
# =========================

def load_json(name: Union[str, Path], folder: Optional[Path] = None):
    """
    讀取 JSON 檔。

    用法 1：給檔名（建議）
        load_json("preprocess_params")
        load_json("preprocess_params.json")

    用法 2：給完整 Path（進階）
        load_json(meta_path("preprocess_params.json"))

    folder:
        - 若 name 是字串：用 folder 或 DIRS.meta 當資料夾
        - 若 name 是 Path：會忽略 folder，直接用這個 Path
    """
    # 如果直接給 Path，就直接用它
    if isinstance(name, Path):
        path = name
    else:
        folder = folder or DIRS.meta
        filename = name if name.lower().endswith(".json") else f"{name}.json"
        path = folder / filename

    if not path.exists():
        raise FileNotFoundError(f"找不到 JSON 檔：{path}")

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
    把每次實驗結果 append 到 artifacts/meta/model_results.csv。

    Parameters
    ----------
    model_name : str
        模型或實驗的名稱，例如 "logit_baseline_v1"。
    params : dict, optional
        超參數，例如 GridSearchCV 的 best_params_。
    metrics : dict, optional
        已經扁平化好的指標，例如
        {"auc_train": 0.61, "auc_val": 0.60, "brier_val": 0.24}。
    notes : str, optional
        備註文字，例如 "with class_weight=balanced"。
    filename : str, optional
        存成的檔案名稱，預設 "model_results.csv"。
    """
    import datetime
    import pandas as pd

    params = params or {}
    metrics = metrics or {}

    # 1. 組一列 row 資料
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
    }

    # 超參數一律加 param__ 前綴
    for k, v in params.items():
        row[f"param__{k}"] = v

    # 指標一律加 metric__ 前綴
    for k, v in metrics.items():
        row[f"metric__{k}"] = v

    if notes is not None:
        row["notes"] = notes

    df = pd.DataFrame([row])

    # 2. 目標檔案路徑：artifacts/meta/model_results.csv
    p = meta_path(filename)
    _ensure_parent(p)  # ✅ 確保資料夾存在

    # 3. 若檔案已存在 → append 不寫 header；否則建立新檔
    if p.exists():
        df.to_csv(p, mode="a", header=False, index=False)
    else:
        df.to_csv(p, mode="w", header=True, index=False)

    print(f"[log_model_result] 已追加 1 筆結果到 {p}")
    return p




# =========================
# sklearn custom transformers 
# =========================

class MedianFromConfigImputer(BaseEstimator, TransformerMixin):
    """
    使用 preprocess_params.json 中的 numeric_impute_values 來做填補，
    不再從資料本身重新估計 median。

    impute_values: dict，key 是欄位名，value 是對應的 median
    feature_names: list，可選，指定這個 transformer 預期處理的欄位順序
    """
    def __init__(self, impute_values: dict, feature_names=None):
        self.impute_values = impute_values
        self.feature_names = feature_names

    def fit(self, X, y=None):
        # 記住欄位順序，讓 transform / get_feature_names_out 用
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        elif self.feature_names is not None:
            self.feature_names_in_ = list(self.feature_names)
        else:
            self.feature_names_in_ = [f"col_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        # 統一轉成 DataFrame 處理
        if hasattr(X, "columns"):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        for col in self.feature_names_in_:
            if col in self.impute_values:
                X_df[col] = X_df[col].fillna(self.impute_values[col])

        # 再轉回原來型態
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
    把輸入統一轉成 float，用在 0/1 flag 類欄位，
    並保持 feature_names_out 不變。
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
# 🧪 測試程式碼（可選）
# =========================

if __name__ == "__main__":
    """
    快速測試 utils 功能
    """
    print("="*80)
    print("Testing utils.py")
    print("="*80)
    
    # 測試路徑
    print("\n1. 測試路徑功能:")
    print(f"  ROOT: {ROOT}")
    print(f"  DIRS.models: {DIRS.models}")
    print(f"  DIRS.meta: {DIRS.meta}")
    
    # 測試 save/load JSON
    print("\n2. 測試 JSON 功能:")
    test_data = {"test": "data", "value": 123}
    json_path = save_json(test_data, "test_utils", folder=DIRS.meta)
    print(f"  已儲存: {json_path}")
    
    loaded_data = load_json("test_utils", folder=DIRS.meta)
    print(f"  已載入: {loaded_data}")
    assert loaded_data == test_data, "JSON 測試失敗！"
    print("  ✓ JSON 測試通過")
    
    # 測試 save_df
    print("\n3. 測試 DataFrame 功能:")
    import pandas as pd
    test_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df_path = save_df(test_df, "test_utils.csv", folder=DIRS.tables)
    print(f"  已儲存: {df_path}")
    print("  ✓ DataFrame 測試通過")
    
    # 測試 log_model_result
    print("\n4. 測試 log_model_result:")
    log_model_result(
        model_name="test_model",
        params={"param1": 0.1, "param2": 100},
        metrics={"auc_val": 0.85, "brier_val": 0.15},
        notes="測試用模型"
    )
    print("  ✓ log_model_result 測試通過")
    
    print("\n" + "="*80)
    print("✅ 所有測試通過！")
    print("="*80)

# =========================
# Public API
# =========================
__all__ = [
    "ROOT", "DIRS", "ensure_dirs",
    "path_in", "raw_path", "processed_path",
    "figure_path", "table_path", "meta_path", "model_path", "report_path", "notebook_path", "tuner_path",
    "save_df", "save_json", "save_pickle", "save_text", "save_bytes", "save_fig",
    "load_json",  # ✅ 確保這行存在
    "MedianFromConfigImputer", "ToFloatTransformer",
    "log_model_result",
]

