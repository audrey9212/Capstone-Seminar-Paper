# %% 01-final
# # ==== 0. Global Setup：匯入套件與專案目錄設定 ======================

# %%
import os
import sys

# ---- 保證專案根目錄永遠在 sys.path 裡 ----
PROJECT_ROOT = os.path.abspath("..")  # 從 notebooks/ 往上一層
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

print("PROJECT_ROOT:", PROJECT_ROOT)

import utils as U

print("U.ROOT:", U.ROOT)


# %%
# ==== 0. Global Setup：匯入套件與專案目錄設定 ======================

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils as U

# 忽略不重要的警告（例如 FutureWarning）
warnings.filterwarnings("ignore")

# ---- 專案 utils：自動偵測專案根目錄 & 圖表儲存工具 -----------------
# 確保可以匯入到 utils.py（如果 Notebook 在 notebooks/ 底下，這樣寫比較安全）
try:
    import utils
except ImportError:
    # 如果在不同資料夾，可視情況調整 .. 或 ../..
    sys.path.append("..")
    import utils

from utils import DIRS, raw_path, processed_path, figure_path, save_fig, save_df  # :contentReference[oaicite:0]{index=0}

# 確認目前專案根目錄位置（debug 用）
print(f"Project root detected by utils: {DIRS.root}")
print(f"Raw data dir:       {DIRS.raw}")
print(f"Processed data dir: {DIRS.processed}")
print(f"Figures dir:        {DIRS.figures}")

# ==== Pandas 顯示設定（讓摘要表在 Notebook 中好讀） =================

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# ==== Random Seed（之後如果有抽樣／隨機動作，結果可重現） ============

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ==== Matplotlib / Seaborn 畫圖風格 ================================

# 基本主題：paper context + colorblind 配色，適合論文用圖
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# 使用 seaborn 的 colorblind palette 做為全域預設
base_palette = sns.color_palette("colorblind")
sns.set_palette(base_palette)

# 若你有中文字標題，可視需求設定字型（不會顯示中文就拿掉這段）
# plt.rcParams["font.family"] = "Noto Sans CJK TC"  # 或你機器上有的中文字型
plt.rcParams["axes.unicode_minus"] = False  # 避免負號變成方塊

# 統一圖表尺寸 & DPI（論文插圖建議 6x4 或 6x5 英吋，300dpi 以上）
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["figure.dpi"] = 120

# ==== Churn 專用顏色與標籤 ==========================================
# 讓「流失 / 未流失」在所有圖中顏色一致
CHURN_COLORS = {
    0: base_palette[0],   # No churn
    1: base_palette[1],   # Churn
    "No": base_palette[0],
    "Yes": base_palette[1],
}

CHURN_LABELS = {
    0: "No Churn (0)",
    1: "Churn (1)",
    "No": "No Churn (0)",
    "Yes": "Churn (1)",
}

# ==== 小工具：統一存圖函式 =========================================

def save_eda_fig(fig: plt.Figure, name: str, dpi: int = 300, tight: bool = True):
    """
    將圖存到 artifacts/figures/ 底下，統一 DPI 與路徑。
    
    Parameters
    ----------
    fig : matplotlib Figure
        要儲存的圖物件 (通常由 plt.gcf() 取得)
    name : str
        檔名，可以是 "F1a_churn_distribution.png" 或不含副檔名 "F1a_churn_distribution"
    dpi : int
        解析度（論文建議 >= 300）
    tight : bool
        是否使用 bbox_inches='tight' 去除多餘空白
    """
    # utils.save_fig 會自動幫你放到 artifacts/figures，並補 .png 副檔名 :contentReference[oaicite:1]{index=1}
    return save_fig(fig, name, dpi=dpi, tight=tight)


print("Global setup completed ✅")


# %% [markdown]
# # ==== 0.x 關鍵欄位名稱與基本設定 =====================================

# %%
# ==== 0.x 關鍵欄位名稱與基本設定 =====================================

# 之後整本 Notebook 都用這三個常數，不要在下面又寫不同名字
ID_COL = "CustomerID"       # 客戶唯一ID
TARGET = "Churn"            # 原始流失欄位（Yes/No 或 1/0）
TARGET_BIN = "Churn01"      # 轉成 0/1 後的新欄位名稱

# 原始訓練資料檔名（如有不同名稱，這一行改掉就好）
RAW_FILE = "cell2celltrain.csv"

print("ID_COL   :", ID_COL)
print("TARGET   :", TARGET)
print("TARGET_BIN:", TARGET_BIN)
print("RAW_FILE :", RAW_FILE)

# ==== 0.x 讀取原始資料 ===============================================

data_path = raw_path(RAW_FILE)
print("讀取資料來源：", data_path)

df = pd.read_csv(data_path)
print("資料形狀 (rows, cols):", df.shape)

# 看前幾列確認一下
display(df.head())


# %% [markdown]
# # ==== 0.x 建立二元目標欄位 Churn01 ==================================

# %%
# ==== 0.x 建立二元目標欄位 Churn01 ==================================

def make_churn01(col: pd.Series) -> pd.Series:
    """
    將原始 Churn 欄位轉成 0/1。
    支援：
    - 數值型：>0 視為 1，其餘 0
    - 文字型：yes/no, y/n, true/false, 1/0 等常見寫法（不分大小寫）
    """
    s = col.copy()

    # 若本來就是數值欄位
    if pd.api.types.is_numeric_dtype(s):
        out = (s.astype(float) > 0).astype(int)
        return out

    # 若是字串／類別，先轉小寫去空白
    s = s.astype(str).str.strip().str.lower()

    mapping = {
        "yes": 1, "y": 1, "1": 1, "true": 1, "t": 1,
        "no": 0,  "n": 0, "0": 0, "false": 0, "f": 0,
    }
    out = s.map(mapping)

    # 如果有無法辨識的值，提醒一下（通常應該是 0 個）
    n_unknown = out.isna().sum()
    if n_unknown > 0:
        print(f"[警告] Churn 欄位中有 {n_unknown} 筆值無法轉成 0/1，暫時當作 0 處理。")
        # 視需求也可以改成 drop 或保留 NaN
        out = out.fillna(0).astype(int)
    else:
        out = out.astype(int)

    return out

# 實際建立新的 0/1 目標欄位
df[TARGET_BIN] = make_churn01(df[TARGET])

print("Churn01 分佈（0=未流失, 1=流失）：")
display(df[TARGET_BIN].value_counts(dropna=False).to_frame("count"))
print("比例：")
display((df[TARGET_BIN].value_counts(normalize=True) * 100).round(2).to_frame("percent"))


# %%
# 讀取資料檢查 (假設 df 已載入)
print(df['Homeownership'].value_counts(dropna=False))

# %% [markdown]
# # ============================================
# # Phase 0：Feature Registry + Semantic Groups

# %%
import numpy as np

# ---------------------------------
# 0.x 修正 ordinal 欄位的 dtype
# ---------------------------------
# HandsetPrice: '10','30',...,'Unknown' → float + NaN
if "HandsetPrice" in df.columns and df["HandsetPrice"].dtype == "object":
    df["HandsetPrice"] = (
        df["HandsetPrice"]
        .replace("Unknown", np.nan)   # 先把 Unknown 當成 NaN
        .astype(float)
    )
    print("HandsetPrice converted to float with NaN for 'Unknown'.")

# CreditRating: "1-Highest" ~ "7-Lowest" → 有順序的分數
credit_order = [
    "1-Highest", "2-High", "3-Good",
    "4-Medium", "5-Low", "6-VeryLow", "7-Lowest"
]
if "CreditRating" in df.columns and df["CreditRating"].dtype == "object":
    credit_map = {lab: i+1 for i, lab in enumerate(credit_order)}
    df["CreditRating"] = df["CreditRating"].map(credit_map).astype(float)
    print("CreditRating converted to ordered numeric scores 1–7.")

print(df[["HandsetPrice", "CreditRating"]].dtypes)


# %%
# ============================================
# Phase 0：Feature Registry + Primary Types + Semantic Groups + Type Tags
# 產出：
#   - T0_feature_registry.csv  （唯一 source of truth）
#   - T0_semantic_groups_table.csv（給論文看的摘要）
#   - 全域 FEATURE_REG + get_features(...)
# ============================================

import pandas as pd

print("Data shape:", df.shape)
print("Columns:", len(df.columns))

# --- 語意分組（依照你舊版 EDA） ---
SEMANTIC_GROUPS = {
    "id_target": [ID_COL, TARGET, TARGET_BIN],
    "billing_economics": [
        "MonthlyRevenue","TotalRecurringCharge","OverageMinutes",
        "PercChangeRevenues","PercChangeMinutes","HandsetPrice"
    ],
    "usage_activity": [
        "MonthlyMinutes","InboundCalls","OutboundCalls","ReceivedCalls",
        "PeakCallsInOut","OffPeakCallsInOut","ThreewayCalls","RoamingCalls",
        "DirectorAssistedCalls"
    ],
    "quality_experience": [
        "DroppedCalls","BlockedCalls","UnansweredCalls","DroppedBlockedCalls",
        "CallWaitingCalls","CallForwardingCalls"
    ],
    "support_retention": [
        "CustomerCareCalls","RetentionCalls","RetentionOffersAccepted","MadeCallToRetentionTeam"
    ],
    "account_tenure": [
        "MonthsInService","UniqueSubs","ActiveSubs","Handsets","HandsetModels",
        "CurrentEquipmentDays","ReferralsMadeBySubscriber"
    ],
    "device_flags": ["HandsetRefurbished","HandsetWebCapable"],
    "demographics_household": [
        "AgeHH1","AgeHH2","ChildrenInHH","Homeownership","IncomeGroup","Occupation",
        "MaritalStatus","OwnsComputer","HasCreditCard","OwnsMotorcycle","TruckOwner","RVOwner",
        "BuysViaMailOrder","RespondsToMailOffers","OptOutMailings","NonUSTravel",
        "NewCellphoneUser","NotNewCellphoneUser","CreditRating","AdjustmentsToCreditRating"
    ],
    "geo_segmentation": ["ServiceArea","PrizmCode"],
}

# --- Primary Types （主要型別） ---
PRIMARY_TYPE_MAP = {
    "CustomerID": "id",
    "Churn": "target",
    "Churn01": "binary_target",

    # Billing / economics
    "MonthlyRevenue": "numeric",
    "TotalRecurringCharge": "numeric",
    "OverageMinutes": "numeric",
    "PercChangeRevenues": "numeric",
    "PercChangeMinutes": "numeric",
    "HandsetPrice": "ordinal",

    # Usage
    "MonthlyMinutes": "numeric",
    "InboundCalls": "numeric",
    "OutboundCalls": "numeric",
    "ReceivedCalls": "numeric",
    "PeakCallsInOut": "numeric",
    "OffPeakCallsInOut": "numeric",
    "ThreewayCalls": "numeric",
    "RoamingCalls": "numeric",
    "DirectorAssistedCalls": "numeric",

    # Quality
    "DroppedCalls": "numeric",
    "BlockedCalls": "numeric",
    "UnansweredCalls": "numeric",
    "DroppedBlockedCalls": "numeric",
    "CallWaitingCalls": "numeric",
    "CallForwardingCalls": "numeric",

    # Support / retention
    "CustomerCareCalls": "numeric",
    "RetentionCalls": "numeric",
    "RetentionOffersAccepted": "numeric",
    "MadeCallToRetentionTeam": "binary",

    # Account tenure
    "MonthsInService": "numeric",
    "UniqueSubs": "numeric",
    "ActiveSubs": "numeric",
    "Handsets": "numeric",
    "HandsetModels": "numeric",
    "CurrentEquipmentDays": "numeric",
    "ReferralsMadeBySubscriber": "numeric",

    # Device flags
    "HandsetRefurbished": "binary",
    "HandsetWebCapable": "binary",

    # Demographics / household
    "AgeHH1": "numeric",
    "AgeHH2": "numeric",
    "ChildrenInHH": "binary",
    "Homeownership": "binary",
    "IncomeGroup": "ordinal",
    "Occupation": "nominal",
    "MaritalStatus": "nominal",
    "OwnsComputer": "binary",
    "HasCreditCard": "binary",
    "OwnsMotorcycle": "binary",
    "TruckOwner": "binary",
    "RVOwner": "binary",
    "BuysViaMailOrder": "binary",
    "RespondsToMailOffers": "binary",
    "OptOutMailings": "binary",
    "NonUSTravel": "binary",
    "NewCellphoneUser": "binary",
    "NotNewCellphoneUser": "binary",
    "CreditRating": "ordinal",
    "AdjustmentsToCreditRating": "numeric",

    # Geo
    "ServiceArea": "nominal",
    "PrizmCode": "nominal",
}

# --- Type Tags（細部標籤：影響前處理 / 視覺化 / 建模） ---
CARDINALITY_TAG = {
    "ServiceArea": "high_card",
    "PrizmCode": "low_card",
    "Occupation": "low_card",
    "MaritalStatus": "low_card",
}

# 自動偵測 sparse-zero 欄位：
# 條件：
#   zero_fraction   >= 0.8  （至少 80% 是 0，才叫「稀疏」）
#   nonzero_fraction >= 0.01（至少 1% > 0，才值得做 flag）
sparse_zero_candidates = []

for col in df.columns:
    if col in {ID_COL, TARGET, TARGET_BIN}:
        continue
    s = df[col]
    # 只針對數值欄位
    if not pd.api.types.is_numeric_dtype(s):
        continue

    zero_fraction = (s == 0).mean()
    nonzero_fraction = ((s != 0) & s.notna()).mean()

    if (zero_fraction >= 0.8) and (nonzero_fraction >= 0.01):
        sparse_zero_candidates.append(col)

SPARSE_ZERO = set(sparse_zero_candidates)
print("Auto-detected SPARSE_ZERO features:", SPARSE_ZERO)

MOMENTUM = {
    "PercChangeMinutes","PercChangeRevenues",
}

ZERO_AS_MISSING = {
    "HandsetPrice","IncomeGroup","AgeHH1",
}

# ------------------------------------------------
# 建立 feature registry 表
# ------------------------------------------------
rows = []
for col in df.columns:
    s = df[col]
    dtype = str(s.dtype)
    primary_type = PRIMARY_TYPE_MAP.get(col, "unknown")
    # 找對應語意分組
    semantic_group = next((k for k, v in SEMANTIC_GROUPS.items() if col in v), "unassigned")

    is_numeric = primary_type in {"numeric", "ordinal"}
    is_binary = primary_type in {"binary", "binary_target"}
    is_categorical = primary_type in {"binary", "binary_target", "ordinal", "nominal"}

    n_unique = s.nunique(dropna=True)
    na_rate = s.isna().mean()

    cardinality_tag = CARDINALITY_TAG.get(col, "none")
    is_sparse_zero = col in SPARSE_ZERO
    is_momentum = col in MOMENTUM
    zero_as_missing = col in ZERO_AS_MISSING

    rows.append(
        {
            "column": col,
            "dtype": dtype,
            "primary_type": primary_type,
            "semantic_group": semantic_group,
            "is_numeric": is_numeric,
            "is_binary": is_binary,
            "is_categorical": is_categorical,
            "n_unique": int(n_unique),
            "na_rate": float(na_rate),
            "cardinality_tag": cardinality_tag,
            "is_sparse_zero": is_sparse_zero,
            "is_momentum": is_momentum,
            "zero_as_missing": zero_as_missing,
        }
    )

reg = pd.DataFrame(rows)

# ------------------------------------------------
# 檢查覆蓋率
# ------------------------------------------------
feature_cols = [c for c in df.columns if c not in {ID_COL, TARGET, TARGET_BIN}]
n_features = len(feature_cols)
reg_feat = reg[reg["column"].isin(feature_cols)]

n_sem_ok = (reg_feat["semantic_group"] != "unassigned").sum()
n_type_ok = (reg_feat["primary_type"] != "unknown").sum()

print(f"[Semantic groups] assigned columns: {n_sem_ok} / total features: {n_features}")
if n_sem_ok == n_features:
    print("✅ Semantic groups fully covered.")
else:
    missing_sem = reg_feat.loc[reg_feat["semantic_group"] == "unassigned", "column"].tolist()
    print("⚠️ Missing semantic_group for:", missing_sem)

print(f"[Primary types] assigned columns: {n_type_ok} / total features: {n_features}")
if n_type_ok == n_features:
    print("✅ Primary types fully covered.")
else:
    missing_pt = reg_feat.loc[reg_feat["primary_type"] == "unknown", "column"].tolist()
    print("⚠️ Missing primary_type for:", missing_pt)

# ------------------------------------------------
# 存檔
# ------------------------------------------------
save_df(reg, "T0_feature_registry")
print("Saved feature registry ✅")
display(reg.head(15))

# ------------------------------------------------
# 語意分組摘要表（給論文用）
# ------------------------------------------------
NICE_NAMES = {
    "id_target": "識別與標的",
    "billing_economics": "帳務與金額",
    "usage_activity": "使用強度與結構",
    "quality_experience": "通話品質與體驗",
    "support_retention": "客服與挽留 / 行銷互動",
    "account_tenure": "帳戶年資與結構",
    "device_flags": "裝置與功能",
    "demographics_household": "人口與家戶特徵",
    "geo_segmentation": "地理與分群",
}

rows_group = []
for g, sub in reg_feat.groupby("semantic_group"):
    nice = NICE_NAMES.get(g, g)
    rows_group.append(
        {
            "Category (分類)": f"{nice}（{len(sub)}）",
            "semantic_group": g,
            "n_vars": len(sub),
            "Variables (columns)": ", ".join(sorted(sub["column"])),
        }
    )

sem_table = pd.DataFrame(rows_group).sort_values("semantic_group").reset_index(drop=True)
save_df(sem_table, "T0_semantic_groups_table")
print("Saved semantic groups table ✅")
display(sem_table)

# ------------------------------------------------
# 建立 FEATURE_REG + get_features 工具
# ------------------------------------------------
FEATURE_REG = reg.set_index("column")

def get_features(
    kind: str | None = None,
    semantic: str | list[str] | None = None,
    include_target: bool = False,
    exclude: str | list[str] | None = None,
) -> list[str]:
    """
    從 FEATURE_REG 選欄位。
    支援：
      kind: "numeric" / "categorical" / None
      semantic: e.g. "billing_economics" 或 ["usage_activity","account_tenure"]
      include_target: 是否保留 ID / 目標欄位
      exclude: 額外排除欄位
    """
    df_reg = FEATURE_REG.copy()
    mask = pd.Series(True, index=df_reg.index)

    if kind == "numeric":
        mask &= df_reg["is_numeric"]
    elif kind == "categorical":
        mask &= df_reg["is_categorical"]

    if semantic is not None:
        if isinstance(semantic, str):
            semantic = [semantic]
        mask &= df_reg["semantic_group"].isin(semantic)

    cols = df_reg.index[mask].tolist()

    if not include_target:
        cols = [c for c in cols if c not in {ID_COL, TARGET, TARGET_BIN}]

    if exclude is not None:
        if isinstance(exclude, str):
            exclude = [exclude]
        cols = [c for c in cols if c not in set(exclude)]

    return cols

print("Feature registry 完成，可以使用 get_features() ✅")
print("Example numeric features:", get_features(kind="numeric")[:10])
print("Example billing features:", get_features(kind="numeric", semantic="billing_economics"))
print("\n[cardinality_tag = 'high_card']",
      FEATURE_REG.index[FEATURE_REG["cardinality_tag"] == "high_card"].tolist())

print("[cardinality_tag = 'low_card']",
      FEATURE_REG.index[FEATURE_REG["cardinality_tag"] == "low_card"].tolist())

print("\n[is_sparse_zero = True]",
      FEATURE_REG.index[FEATURE_REG["is_sparse_zero"]].tolist())

print("[is_momentum = True]",
      FEATURE_REG.index[FEATURE_REG["is_momentum"]].tolist())

print("[zero_as_missing = True]",
      FEATURE_REG.index[FEATURE_REG["zero_as_missing"]].tolist())




# %% [markdown]
# # 第一章：資料健康度與稀疏性 (Data Health & Sparsity)

# %% [markdown]
# # 1.0 Quick overview

# %%
# 1.0 Quick overview of data health
# 這段只是快速看一下資料形狀與欄位，幫助你確認有沒有讀錯檔

print("Shape:", df.shape)
print("Columns:", len(df.columns))
print(df.dtypes.value_counts())

# 如果想快速看前幾列
display(df.head())


# %% [markdown]
# # 1.1 Missing & Zero-inflation overview

# %%
# ============================================
# 1.1 Missing & Zero-inflation overview
# 目標：
#   - 每個特徵計算「有效 missing、有效 zero、non-zero」比例
#   - 將 ZERO_AS_MISSING / FEATURE_REG 中設定的欄位 (0 視為 missing) 納入
#   - 產生：
#       1) T1_sparsity_overview.csv（全 58 個欄位的簡潔表）
#       2) F1a_sparsity_missing_zero.png（只畫問題最大的 Top 20）
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- 1.1.0 要分析的欄位：預設用 get_features()，沒有就 fallback ----

if "get_features" in globals():
    # 從 feature registry 抓「真正的特徵欄位」，預設排除 ID / 目標
    feature_cols = get_features(include_target=False)
else:
    # 保險做法：直接從 df.columns 排除 ID / 目標欄位
    feature_cols = [
        c for c in df.columns
        if c not in {ID_COL, TARGET, TARGET_BIN}
    ]

# ---- 1.1.1 從 FEATURE_REG / ZERO_AS_MISSING 找出「0 視為 missing」欄位 ----

zero_as_missing_set: set[str] = set()

# Phase 0 已經在 FEATURE_REG 裡標好 zero_as_missing 的話，優先使用
if "FEATURE_REG" in globals() and "zero_as_missing" in FEATURE_REG.columns:
    zero_as_missing_set = set(
        FEATURE_REG.index[FEATURE_REG["zero_as_missing"]]
    )
# 否則就退回到全域的 ZERO_AS_MISSING（若有設定）
elif "ZERO_AS_MISSING" in globals():
    zero_as_missing_set = set(ZERO_AS_MISSING)

print("ZERO_AS_MISSING used in sparsity:", zero_as_missing_set)


def compute_sparsity(
    df: pd.DataFrame,
    columns: list[str],
    zero_as_missing: set[str] | None = None
) -> pd.DataFrame:
    """
    計算每個欄位的 sparsity 結構。

    raw_missing_fraction:  只看 NaN
    raw_zero_fraction:     只看數值 0（只對 numeric 有意義）

    missing_fraction:      有效 missing 比例
                           = NaN + (若在 zero_as_missing，就把 0 也算進來)
    zero_fraction:         有效 zero 比例
                           = 真正的 0（不屬於 zero_as_missing 的欄位）
    nonzero_fraction:      非 missing 且非 zero
    """
    if zero_as_missing is None:
        zero_as_missing = set()

    records: list[dict] = []

    for col in columns:
        s = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(s)

        # 原始缺值標記
        is_missing_raw = s.isna()

        # 原始 zero 標記（只對 numeric）
        if is_numeric:
            is_zero_raw = (~is_missing_raw) & (s == 0)
        else:
            is_zero_raw = pd.Series(False, index=s.index)

        # 原始比例（主要是 debug 用，不一定要寫進論文）
        raw_missing_fraction = is_missing_raw.mean()
        raw_zero_fraction = is_zero_raw.mean()

        # 決定是否把「0」併入 missing
        if is_numeric and (col in zero_as_missing):
            # 0 也算 missing，zero 區塊變成 0
            is_missing_eff = is_missing_raw | is_zero_raw
            is_zero_eff = pd.Series(False, index=s.index)
        else:
            # 一般情況：NaN = missing, 0 = zero
            is_missing_eff = is_missing_raw
            is_zero_eff = is_zero_raw

        # 非 missing 且非 zero
        is_nonzero_eff = (~is_missing_eff) & (~is_zero_eff)

        missing_fraction = is_missing_eff.mean()
        zero_fraction = is_zero_eff.mean()
        nonzero_fraction = is_nonzero_eff.mean()

        records.append(
            {
                "feature": col,
                "dtype": s.dtype,
                "is_numeric": is_numeric,
                # 原始比例（只在 notebook 中做 sanity check 用）
                "raw_missing_fraction": raw_missing_fraction,
                "raw_zero_fraction": raw_zero_fraction,
                # 有效比例（論文與圖主要看這個）
                "missing_fraction": missing_fraction,
                "zero_fraction": zero_fraction,
                "nonzero_fraction": nonzero_fraction,
                "missing_plus_zero": missing_fraction + zero_fraction,
                "zero_as_missing": bool(is_numeric and (col in zero_as_missing)),
            }
        )

    out = pd.DataFrame(records)
    return out


# ---- 1.1.2 實際計算 sparsity_df ------------------------------------

sparsity_df = compute_sparsity(
    df,
    feature_cols,
    zero_as_missing=zero_as_missing_set,
)

print("Sparsity table shape:", sparsity_df.shape)
display(
    sparsity_df.sort_values("missing_plus_zero", ascending=False).head(20)
)

# ---- 1.1.3 輸出乾淨版 CSV（給論文 / 後續 notebook 用） -------------

# 只保留「對外公開」需要的欄位，rows = 全部 features
sparsity_export = sparsity_df[
    [
        "feature",
        "missing_fraction",
        "zero_fraction",
        "missing_plus_zero",
        "zero_as_missing",
    ]
].sort_values("missing_plus_zero", ascending=False)

try:
    save_df(sparsity_export, "T1_sparsity_overview")
    print("Saved table: T1_sparsity_overview.csv")
except Exception as e:
    print("[Warning] save_df 失敗，改用 to_csv。錯誤訊息：", e)
    sparsity_export.to_csv(
        "T1_sparsity_overview.csv",
        index=False,
        encoding="utf-8",
        float_format="%.6f",
    )
    print("Saved table locally: T1_sparsity_overview.csv")

# ---- 1.1.4 視覺化：只畫「有問題」欄位的 Top 20 --------------------

# 只保留「有效 missing 或 zero 有出現」的欄位
mask_plot = (sparsity_df["missing_fraction"] > 0) | (sparsity_df["zero_fraction"] > 0)
plot_df = sparsity_df.loc[mask_plot].copy()

if plot_df.empty:
    print("No features with missing or zero values – skip sparsity plot.")
else:
    # 依 missing_plus_zero 由大到小排序，取前 20（如果不足 20 就全畫）
    plot_df = (
        plot_df
        .sort_values("missing_plus_zero", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )

    # 轉成百分比方便畫圖
    plot_df["missing_pct"] = plot_df["missing_fraction"] * 100
    plot_df["zero_pct"] = plot_df["zero_fraction"] * 100
    plot_df["nonzero_pct"] = plot_df["nonzero_fraction"] * 100

    y_pos = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(10, 6))

    # 水平堆疊條圖：Missing -> Zero -> Non-zero
    ax.barh(
        y_pos,
        plot_df["missing_pct"],
        label="Missing (NaN or structural zero)",
        color="#e0e0e0",
    )
    ax.barh(
        y_pos,
        plot_df["zero_pct"],
        left=plot_df["missing_pct"],
        label="Zero (value = 0)",
        color="#c7dcec",
    )
    ax.barh(
        y_pos,
        plot_df["nonzero_pct"],
        left=plot_df["missing_pct"] + plot_df["zero_pct"],
        label="Non-zero (non-missing)",
        color="#346aa9",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["feature"])
    ax.set_xlabel("Percentage of records (%)")
    ax.set_title("Missing and zero-value structure by feature (top sparsity)")

    # x 軸 0–100%，留一點空間給左側註記
    ax.set_xlim(0, 100)

    # 在有 zero_as_missing 的欄位前面加上 * 標記
    for i, row in plot_df.iterrows():
        if row["zero_as_missing"]:
            ax.text(
                -0.5,
                i,
                "*",
                color="red",
                va="center",
                ha="right",
                fontsize=12,
                transform=ax.get_yaxis_transform(),  # y 用 data, x 用 axis fraction
            )

    # 在圖下方加註解說明 * 是什麼
    ax.text(
        0.0,
        -0.15,
        "* 0 is treated as missing for this feature",
        transform=ax.transAxes,
        color="red",
        fontsize=9,
        ha="left",
        va="top",
    )

    ax.legend(loc="lower right")
    plt.tight_layout()

    # 存圖
    try:
        U.save_fig(fig, "F1a_sparsity_missing_zero")
        print("Saved figure: F1a_sparsity_missing_zero.png")
    except Exception as e:
        print("[Warning] save_fig 失敗，改用 fig.savefig。錯誤訊息：", e)
        fig.savefig("F1a_sparsity_missing_zero.png", dpi=200, bbox_inches="tight")
        print("Saved figure locally: F1a_sparsity_missing_zero.png")

    plt.show()


# %%
# optional：missingno 的 matrix 圖當做補充

# Missingness matrix (supplementary figure)
# 這一格會：
# 1. 建一份 df_ms，把 ZERO_AS_MISSING 欄位中的「0」也轉成 NaN
# 2. 用 missingno 畫出 missingness matrix
# 3. 存成 F1b_missingness_matrix.png（或本地檔）

import pandas as pd

try:
    import missingno as msno
except ImportError:
    print("missingno is not installed. Run `%pip install missingno` first.")
else:
    # --- 1. 準備一份「包含結構性缺失」的 df_ms ------------------------

    df_ms = df.copy()

    # 從 FEATURE_REG 或 ZERO_AS_MISSING 取得要把 0 當作 missing 的欄位
    zero_as_missing_set = set()

    if "FEATURE_REG" in globals() and "zero_as_missing" in FEATURE_REG.columns:
        zero_as_missing_set = set(
            FEATURE_REG.index[FEATURE_REG["zero_as_missing"]]
        )
    elif "ZERO_AS_MISSING" in globals():
        zero_as_missing_set = set(ZERO_AS_MISSING)

    print("ZERO_AS_MISSING used in missingno matrix:", zero_as_missing_set)

    for col in zero_as_missing_set:
        if col in df_ms.columns:
            # 只在 numeric 欄位上，把 0 轉成 NaN
            if pd.api.types.is_numeric_dtype(df_ms[col]):
                df_ms.loc[df_ms[col] == 0, col] = np.nan

    # --- 2. 決定要畫哪些欄位 ---------------------------------------

    # A) 精簡版：只看幾個關鍵欄位（IncomeGroup / HandsetPrice / AgeHH1 / AgeHH2）
    key_cols = [c for c in ["IncomeGroup", "HandsetPrice", "AgeHH1", "AgeHH2"] if c in df_ms.columns]

    # 如果你想畫「全部特徵」，可以把下面這行改成 feature_cols 或 df_ms.columns
    cols_for_matrix = key_cols   # 或者改成 feature_cols / df_ms.columns

    # --- 3. 畫 missingno matrix -------------------------------------

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    msno.matrix(
        df_ms[cols_for_matrix],
        ax=ax,
        labels=cols_for_matrix
    )

    ax.set_title("True missingness (NaN + structural zero as missing)")

    plt.tight_layout()

    # --- 4. 存圖 -----------------------------------------------------

    try:
        import utils as U
        U.save_fig(fig, "F1b_missingness_matrix")
        print("Saved figure: F1b_missingness_matrix.png")
    except ImportError:
        fig.savefig("F1b_missingness_matrix.png", dpi=200, bbox_inches="tight")
        print("Saved figure locally: F1b_missingness_matrix.png")

    plt.show()



#AgeHH2 的 0 視為 ‘no second householder’ ，所以不轉成 NaN。

# %% [markdown]
# # 1.2a Value-domain and logic checks

# %%
# ============================================
# 1.2 Value-domain and logic checks (re-built)
#
# 目的：
#   - 檢查「數值範圍是否合理」與「基本商業邏輯是否違反」
#   - 包含：
#       [A] 年齡合理性 (AgeHH1 / AgeHH2)
#       [B] 非負值檢查（minutes / revenue / calls / subs / days ...）
#       [C] 邏輯不等式（ActiveSubs <= UniqueSubs 等）
#       [D] 整數檢查（UniqueSubs, ActiveSubs, Handsets, HandsetModels）
#       [E] 互斥欄位（NewCellphoneUser vs NotNewCellphoneUser）
#
# 輸出：
#   - T1_value_domain_checks.csv
#   - F1b_value_domain_logic_violations_refined.png（若有違規才畫）
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

n_total = len(df)

checks: list[dict] = []  # 存所有 rule 的結果


def add_check(check_type: str, rule: str, mask: pd.Series) -> int:
    """
    小工具：把每一條 rule 的違規列數 & 比例記錄下來。

    check_type : 類別，例如 "Age Validity", "Non-Negative", ...
    rule       : 人類可讀的規則描述（英文，方便當成圖表 Y 軸）
    mask       : bool Series，True = 違規的 row
    """
    mask = mask.fillna(False)
    n = int(mask.sum())
    rate = float(n / n_total) if n_total > 0 else 0.0
    checks.append(
        {
            "check_type": check_type,
            "rule": rule,
            "n_rows": n,
            "rate": rate,
        }
    )
    return n


# --------------------------------------------------------
# [A] Age reasonableness (AgeHH1 / AgeHH2)
# --------------------------------------------------------
AGE_MIN, AGE_MAX = 10, 110  # 你可以之後在報告說明這個選擇
age_cols = [c for c in ["AgeHH1", "AgeHH2"] if c in df.columns]

for col in age_cols:
    s = df[col]
    # 只檢查 >0 的值（0 在前面 ZERO_AS_MISSING / has_second-householder 的邏輯處理）
    mask_invalid_age = s.notna() & (s > 0) & ((s < AGE_MIN) | (s > AGE_MAX))
    add_check("Age Validity", f"{col} outside [{AGE_MIN}, {AGE_MAX}]", mask_invalid_age)


# --------------------------------------------------------
# [B] Non-negative checks (minutes / revenue / calls / subs / days / price)
# --------------------------------------------------------
nonneg_keywords = [
    "minutes", "mou", "revenue", "charge", "calls",
    "subs", "days", "price"
]

# 百分比變化允許負值
ALLOW_NEGATIVE_COLS = ["PercChangeRevenues", "PercChangeMinutes"]

# 即使名稱沒有關鍵字，也強制檢查非負
ALWAYS_NONNEG_COLS = ["CurrentEquipmentDays", "MonthsInService"]

for col in df.columns:
    if col in {ID_COL, TARGET, TARGET_BIN}:
        continue

    s = df[col]
    if not pd.api.types.is_numeric_dtype(s):
        continue

    # 允許負值的欄位直接跳過
    if col in ALLOW_NEGATIVE_COLS:
        continue

    col_lower = col.lower()
    should_be_nonneg = False

    if any(k in col_lower for k in nonneg_keywords):
        should_be_nonneg = True
    if col in ALWAYS_NONNEG_COLS:
        should_be_nonneg = True

    if not should_be_nonneg:
        continue

    mask_neg = s < 0
    add_check("Non-Negative", f"{col} has negative values", mask_neg)


# --------------------------------------------------------
# [C] Logical inequalities
# --------------------------------------------------------
ineq_rules = [
    # (描述, 左欄, 右欄, lambda(left,right) -> bool mask for violation)
    ("ActiveSubs > UniqueSubs", "ActiveSubs", "UniqueSubs",
     lambda a, b: a > b),
    ("HandsetModels > Handsets", "HandsetModels", "Handsets",
     lambda a, b: a > b),
    ("DroppedBlockedCalls < DroppedCalls", "DroppedBlockedCalls", "DroppedCalls",
     lambda total, part: total < part),
    ("DroppedBlockedCalls < BlockedCalls", "DroppedBlockedCalls", "BlockedCalls",
     lambda total, part: total < part),
]

for desc, col_left, col_right, cond in ineq_rules:
    if {col_left, col_right}.issubset(df.columns):
        mask_viol = cond(df[col_left], df[col_right])
        add_check("Logical Inequality", desc, mask_viol)

# ActiveSubs <= UniqueSubs
# HandsetModels <= Handsets
# DroppedBlockedCalls >= DroppedCalls
# DroppedBlockedCalls >= BlockedCalls



# --------------------------------------------------------
# [D] Integer-only checks (count-like columns)
# 依官方資料字典：下列欄位是「四個月平均」，小數合情合理 -> 不應檢查整數性
# --------------------------------------------------------
integer_check_cols = [
    col for col in [    "UniqueSubs", "ActiveSubs", "Handsets", "HandsetModels",
    "ReferralsMadeBySubscriber", "RetentionCalls", "RetentionOffersAccepted",
    "AdjustmentsToCreditRating", "CurrentEquipmentDays", "MonthsInService",
    # 依資料特性可選擇加入（通常也是整數）：
    "AgeHH1", "AgeHH2"]
    if col in df.columns
]

for col in integer_check_cols:
    s = df[col]
    # 只檢查非 NaN，且有小數部分的視為違規
    frac_part, _ = np.modf(s.astype(float))
    mask_nonint = s.notna() & (np.abs(frac_part) > 1e-9)
    add_check("Integer Validity", f"{col} has non-integer values", mask_nonint)



# --------------------------------------------------------
# [E] Mutually exclusive flags
# --------------------------------------------------------
if {"NewCellphoneUser", "NotNewCellphoneUser"}.issubset(df.columns):
    a = df["NewCellphoneUser"]
    b = df["NotNewCellphoneUser"]

    # 嘗試統一成 0/1：若是字串就轉小寫 + yes/no
    def to_binary(x: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(x):
            return (x.astype(float) > 0).astype(int)
        s = x.astype(str).str.strip().str.lower()
        return s.isin(["yes", "y", "true", "t", "1"]).astype(int)

    a_bin = to_binary(a)
    b_bin = to_binary(b)

    mask_conflict = (a_bin == 1) & (b_bin == 1)
    add_check(
        "Mutually Exclusive",
        "NewCellphoneUser = 1 and NotNewCellphoneUser = 1",
        mask_conflict,
    )


# --------------------------------------------------------
# 匯總：存成表格 + 人類可讀 summary
# --------------------------------------------------------
checks_df = pd.DataFrame(checks).sort_values(
    ["n_rows", "check_type", "rule"], ascending=[False, True, True]
)

# 存 CSV（完整，包含 0 違規的 rule）
try:
    U.save_df(checks_df, "T1_value_domain_checks")
    print("Saved table: T1_value_domain_checks.csv")
except Exception:
    checks_df.to_csv("T1_value_domain_checks.csv", index=False)
    print("Saved figure: T1_value_domain_checks.csv")

print("Value-domain & logic checks (all rules):")
display(checks_df)

# 依 check_type 做總結
summary = (
    checks_df.groupby("check_type", as_index=False)["n_rows"]
    .sum()
    .sort_values("n_rows", ascending=False)
)
print("Summary by check_type:")
display(summary)

for _, row in summary.iterrows():
    if row["n_rows"] == 0:
        print(f"✅ {row['check_type']}: no violations.")
    else:
        print(f"⚠️ {row['check_type']}: {row['n_rows']} rows violate at least one rule.")


# --------------------------------------------------------
# 視覺化：只畫「有違規」的 rule（避免一堆 0 的 bar）
# --------------------------------------------------------
viol_df = checks_df[checks_df["n_rows"] > 0].copy()

if viol_df.empty:
    print("✅ No value-domain or logic violations to plot.")
else:
    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(viol_df))))
    sns.barplot(
        data=viol_df,
        x="n_rows",
        y="rule",
        hue="check_type",
        dodge=False,
        ax=ax,
    )
    ax.set_title("Counts of value-domain and logic violations", fontsize=14)
    ax.set_xlabel("Number of rows")
    ax.set_ylabel("Rule")

    plt.tight_layout()
    try:
        U.save_fig(fig, "F1b_value_domain_logic_violations_refined")
        print("Saved figure: F1b_value_domain_logic_violations_refined.png")
    except Exception:
        fig.savefig("F1b_value_domain_logic_violations_refined.png",
                    dpi=200, bbox_inches="tight")
        print("Saved figure locally: F1b_value_domain_logic_violations_refined.png")

    plt.show()


# %% [markdown]
# # 1.2b Extreme value check (IQR × 3)

# %%
# ============================================
# 1.2b Extreme value check (IQR × 3 for numeric features)
# 目的：
#   - 找出「數值極端」的欄位與列數
#   - 幫助後續決定是否要做 winsorize / clipping / log-transform
# 做法：
#   - 對每一個數值欄位計算 Q1, Q3, IQR = Q3 - Q1
#   - 以 [Q1 - 3*IQR, Q3 + 3*IQR] 之外的觀測值視為 outlier
#   - 對於 zero_as_missing 欄位（例如 HandsetPrice, IncomeGroup, AgeHH1），先把 0 當作 NaN 再算 IQR
# 輸出：
#   - T1_outliers_iqr3.csv：每個欄位的 outlier 統計
#   - F1d_extreme_values_iqr3.png：outlier 比例最高的前幾個欄位
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype

# --- 取出數值欄位（用 Phase 0 的 registry） -------------------------
numeric_cols = get_features(kind="numeric", include_target=False)

# zero_as_missing 欄位（Phase 0 已標記）
zero_as_missing_set: set[str] = set()
if "FEATURE_REG" in globals() and "zero_as_missing" in FEATURE_REG.columns:
    zero_as_missing_set = set(
        FEATURE_REG.index[FEATURE_REG["zero_as_missing"]]
    )

print("IQR×3 extreme-value check")
print("  - numeric columns:", len(numeric_cols))
print("  - zero_as_missing columns:", zero_as_missing_set)


def compute_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str],
    k: float = 3.0,
    zero_as_missing: set[str] | None = None,
) -> pd.DataFrame:
    """
    對每個欄位計算 IQR×k 的極端值(outlier)比例。

    步驟：
    - 若欄位不是 numeric dtype，先用 to_numeric(, errors='coerce') 嘗試轉成數值；
      轉不動的變成 NaN。
    - 對於有標記在 zero_as_missing 的欄位，先把 0 視為 NaN 再計算 IQR。
    - IQR=0 或 NaN 的欄位（幾乎是常數）跳過。
    - outlier 比例以「整體樣本數」為分母（方便比較），
      另附一個 non-missing 分母的比例。
    """
    if zero_as_missing is None:
        zero_as_missing = set()

    n_total = len(df)
    records: list[dict] = []

    for col in columns:
        if col not in df.columns:
            continue

        # 原始 Series
        raw = df[col]

        # 嘗試確保是 numeric
        if not is_numeric_dtype(raw):
            s = pd.to_numeric(raw, errors="coerce")
        else:
            s = raw.astype(float)

        # 把 0 當作 missing（只對被標記的欄位）
        if col in zero_as_missing:
            s = s.mask(s == 0)

        # 用 non-missing 值算 IQR
        s_nonmissing = s.dropna()
        n_nonmissing = len(s_nonmissing)
        if n_nonmissing == 0:
            continue

        q1 = s_nonmissing.quantile(0.25)
        q3 = s_nonmissing.quantile(0.75)
        iqr = q3 - q1

        # IQR=0 或 NaN => 幾乎是常數，跳過
        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - k * iqr
        upper = q3 + k * iqr

        # 在同一個處理後的 s 上判斷 outlier
        is_outlier = (s < lower) | (s > upper)
        n_outliers = int(is_outlier.sum())

        if n_outliers == 0:
            frac_all = 0.0
            frac_nonmissing = 0.0
        else:
            frac_all = n_outliers / n_total
            frac_nonmissing = n_outliers / n_nonmissing

        records.append(
            {
                "feature": col,
                "lower_bound": lower,
                "upper_bound": upper,
                "n_outliers": n_outliers,
                "frac_outliers_all": frac_all,          # 以所有列為分母
                "frac_outliers_nonmissing": frac_nonmissing,  # 以非缺值列為分母
            }
        )

    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values("frac_outliers_all", ascending=False).reset_index(drop=True)
    return out


# --- 實際計算 --------------------------------------------------------
outlier_df = compute_outliers_iqr(
    df,
    numeric_cols,
    k=3.0,
    zero_as_missing=zero_as_missing_set,
)

print("Outlier table shape:", outlier_df.shape)
display(outlier_df.head(15))

# 存成 CSV（完整：所有有 IQR 的 numeric 欄位）
try:
    U.save_df(outlier_df, "T1_outliers_iqr3")
    print("Saved table: T1_outliers_iqr3.csv")
except Exception:
    outlier_df.to_csv("T1_outliers_iqr3.csv", index=False)
    print("Saved table locally: T1_outliers_iqr3.csv")

# --- 視覺化：只畫有 outlier 的前 10 個欄位 ------------------------
df_plot = outlier_df.query("n_outliers > 0").head(10).copy()

if df_plot.empty:
    print("✅ No features with IQR-based outliers (k=3) to plot.")
else:
    df_plot["frac_outliers_pct"] = df_plot["frac_outliers_all"] * 100

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(
        data=df_plot,
        x="frac_outliers_pct",
        y="feature",
        ax=ax,
    )

    ax.set_title("Features with IQR-based outliers (k = 3)")
    ax.set_xlabel("Percentage of rows with outliers (%)")
    ax.set_ylabel("Feature")

    for i, row in df_plot.iterrows():
        ax.text(
            row["frac_outliers_pct"] + 0.05,
            i,
            f"{row['frac_outliers_pct']:.2f}%",
            va="center",
        )

    plt.tight_layout()
    try:
        U.save_fig(fig, "F1d_extreme_values_iqr3")
        print("Saved figure: F1d_extreme_values_iqr3.png")
    except Exception:
        fig.savefig("F1d_extreme_values_iqr3.png", dpi=200, bbox_inches="tight")
        print("Saved figure locally: F1d_extreme_values_iqr3.png")

    plt.show()


# %% [markdown]
# # 1.2c Descriptive statistics（overall + by churn）

# %%
# ============================================
# 1.2c Descriptive statistics (overall & by churn)
# 目的：
#   - 給出數值特徵的基本統計量，作為論文 Table / 附錄
#   - 同時按 Churn01 分組，觀察流失與未流失族群的平均/分布差異
# 輸出：
#   - T1_numeric_summary_overall.csv
#   - T1_numeric_summary_by_churn.csv
# ============================================

# 這裡沿用上一格的 numeric_cols（get_features(kind="numeric")）
numeric_cols = get_features(kind="numeric", include_target=False)

# ---- (1) 整體描述統計（overall） ------------------------------

desc_overall = (
    df[numeric_cols]
    .describe(percentiles=[0.25, 0.5, 0.75])
    .T  # 欄位變成列，比較好讀
)

desc_overall = desc_overall.rename_axis("feature").reset_index()
U.save_df(desc_overall, "T1_numeric_summary_overall")
print("Saved table: T1_numeric_summary_overall.csv")

print("Overall numeric summary (head):")
display(desc_overall.head())

# ---- (2) 依 Churn01 分組的描述統計 -----------------------------

# 確保 TARGET_BIN 存在
assert TARGET_BIN in df.columns, f"{TARGET_BIN} not found in df!"

records = []
stats = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

for col in numeric_cols:
    for churn_val, sub in df.groupby(TARGET_BIN):
        d = sub[col].describe(percentiles=[0.25, 0.5, 0.75])
        row = {
            "feature": col,
            "churn": int(churn_val),
        }
        for s in stats:
            row[s] = d.get(s, np.nan)
        records.append(row)

desc_by_churn = pd.DataFrame(records)
U.save_df(desc_by_churn, "T1_numeric_summary_by_churn")
print("Saved table: T1_numeric_summary_by_churn.csv")

print("Numeric summary by churn (head):")
display(desc_by_churn.head(10))


# %% [markdown]
# # 1.3 Duplicate rows & ID consistency

# %%
# ============================================
# 1.3 Duplicate rows & ID consistency check
# 目的：
#   - 檢查：
#       1) 是否有「整列完全重複」的資料
#       2) 是否有「同一個 CustomerID 出現多列」
#   - 視覺化顯示每種問題有多少列
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

n_total = len(df)

issues = []

# ---- 1.3.1 整列重複（所有欄位都一樣） -----------------------------

dup_full_mask = df.duplicated(keep=False)  # True = 這列屬於某個重複群組
n_dup_full_rows = int(dup_full_mask.sum())

if n_dup_full_rows > 0:
    # 重複群組數量（unique 的重複 row 數）
    n_dup_full_groups = int(df[dup_full_mask].drop_duplicates().shape[0])
else:
    n_dup_full_groups = 0

issues.append({
    "issue": "Full-row duplicates",
    "n_rows": n_dup_full_rows,
    "n_groups": n_dup_full_groups,
})

# ---- 1.3.2 ID 是否唯一（同一個 CustomerID 出現多列） ---------------

if ID_COL in df.columns:
    dup_id_mask = df.duplicated(subset=[ID_COL], keep=False)
    n_dup_id_rows = int(dup_id_mask.sum())
    n_dup_id_values = int(df.loc[dup_id_mask, ID_COL].nunique())

    issues.append({
        "issue": f"{ID_COL} duplicated",
        "n_rows": n_dup_id_rows,
        "n_groups": n_dup_id_values,
    })
else:
    print(f"Warning: ID_COL '{ID_COL}' not found in df.columns; skip ID duplicate check.")

# ---- 1.3.3 匯總表 -----------------------------------------------

dup_df = pd.DataFrame(issues)

print(f"Total rows: {n_total}")
display(dup_df)

# 若想看具體範例，可以解開以下 display：
if n_dup_full_rows > 0:
    print("\nExample of full-row duplicates:")
    display(
        df[dup_full_mask]
        .head(10)
    )

if "dup_id_mask" in globals() and dup_id_mask.any():
    print("\nExample of duplicated IDs:")
    display(
        df[dup_id_mask]
        .sort_values(ID_COL)
        .head(10)
    )

# ---- 1.3.4 視覺化（bar chart） -----------------------------------

# ---- 1.3.4 視覺化（只有在有問題時才畫圖） ------------------------

if dup_df["n_rows"].sum() == 0:
    print("No duplicate issues detected – skip plotting.")
else:
    fig, ax = plt.subplots(figsize=(7, 3.5))

    sns.barplot(
        data=dup_df,
        x="n_rows",
        y="issue",
        ax=ax
    )

    ax.set_title("Duplicate rows and ID issues")
    ax.set_xlabel("Number of rows")
    ax.set_ylabel("Issue type")

    for i, row in dup_df.iterrows():
        ax.text(
            row["n_rows"] + max(1, 0.005 * n_total),
            i,
            f"{row['n_rows']} rows\n({row['n_groups']} groups)",
            va="center"
        )

    plt.tight_layout()

    try:
        import utils as U
        U.save_fig(fig, "F1c_duplicate_rows_id_check")
        print("Saved figure: F1c_duplicate_rows_id_check.png")
    except Exception:
        fig.savefig("F1c_duplicate_rows_id_check.png", dpi=200, bbox_inches="tight")
        print("Saved figure locally: F1c_duplicate_rows_id_check.png")

    plt.show()


# %% [markdown]
# # 第二章：目標變數基準 → 2.1 流失率分佈（Churn Distribution）

# %%
# ============================================
# 2.1 Churn distribution (Target baseline / class imbalance)
# 目的：
#   - 看整體流失率 & 類別是否不平衡
#   - 給出「全部猜不流失」的 baseline 準確率
# 輸出：
#   - T2_churn_distribution.csv
#   - F2a_churn_distribution_donut.png
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- 2.1.1 準備目標欄位 -----------------------------------------

# 優先使用二元欄位（例如 Churn01）
if TARGET_BIN in df.columns:
    y = df[TARGET_BIN]
    target_label = TARGET_BIN
    # 0 = 未流失, 1 = 流失
    class_labels = {0: "No", 1: "Yes"}
    y_for_plot = y.map(class_labels)
else:
    # 若沒有 Churn01，就直接用原始 Churn，並嘗試標準化為 Yes/No
    y = df[TARGET].astype(str).str.strip().str.lower()
    target_label = TARGET
    mapping = {"yes": "Yes", "y": "Yes", "1": "Yes",
               "no": "No",  "n": "No",  "0": "No"}
    y_for_plot = y.map(mapping)

# 統計 Yes / No
N = len(df)
count_series = (
    y_for_plot
    .value_counts(dropna=False)
    .reindex(["No", "Yes"])          # 固定順序：先 No 再 Yes
)
percent_series = (count_series / N * 100).round(1)

churn_summary = pd.DataFrame(
    {
        "count": count_series,
        "percent": percent_series,
    }
)
churn_summary.index.name = "class"

print(f"Target column used: {target_label}")
print(f"Total rows N = {N:,}")
display(churn_summary)

# 存成表格，方便論文引用
U.save_df(churn_summary, "T2_churn_distribution")

# ---- 2.1.2 Baseline：全部猜「不流失」的準確率 ---------------------

# 多數類別（通常是 "No"）
majority_class = churn_summary["percent"].idxmax()
baseline_acc = churn_summary.loc[majority_class, "percent"] / 100.0

print(
    f"Baseline accuracy (always predict '{majority_class}') "
    f"= {baseline_acc:.3%}"
)

# ---- 2.1.3 視覺化：環圈圖 (Donut Chart) --------------------------

fig, ax = plt.subplots(figsize=(5.5, 5.5))

# 顏色可以之後再微調，這裡先給溫和一點的配色
colors = ["#9BBBD4", "#F4A582"]  # No, Yes

# 畫 pie，但把 width 調小變成「環」
wedges, _ = ax.pie(
    count_series,
    startangle=90,
    counterclock=False,
    colors=colors,
    wedgeprops=dict(width=0.4, edgecolor="white"),
)

# 在每一塊扇形上標註 class / % / n
for i, w in enumerate(wedges):
    angle = (w.theta2 + w.theta1) / 2
    x = 0.75 * np.cos(np.deg2rad(angle))
    y_ = 0.75 * np.sin(np.deg2rad(angle))

    cls = churn_summary.index[i]
    pct = churn_summary.iloc[i]["percent"]
    cnt = churn_summary.iloc[i]["count"]

    ax.text(
        x,
        y_,
        f"{cls}\n{pct:.1f}%\n(n={cnt:,})",
        ha="center",
        va="center",
        fontsize=10,
    )

# 中心文字：N + baseline
center_text = (
    f"N = {N:,}\n"
    f"Baseline ≈ {baseline_acc*100:.1f}%\n"
    f"(all '{majority_class}')"
)
ax.text(0, 0, center_text, ha="center", va="center", fontsize=11)

ax.set_title("Churn distribution and baseline accuracy", fontsize=13)
ax.axis("equal")  # 讓圓形不會變橢圓

plt.tight_layout()
U.save_fig(fig, "F2a_churn_distribution_donut")
print("Saved figure: F2a_churn_distribution_donut.png")
plt.show()


# %% [markdown]
# # 3.1 Key numeric features vs Churn (boxplot grid)
# # 第三章：關鍵數值特徵 vs 流失

# %% [markdown]
# # 3.1 核心財務 / 使用量 / 年資分佈

# %%
# ============================================
# 第三章：關鍵數值特徵 vs 流失
# 3.1 核心財務 / 使用量 / 年資分佈
# ============================================

import matplotlib.pyplot as plt
import seaborn as sns

# ---- 3.1.0 Churn 標籤欄位（畫圖好看一點） ------------------------

CHURN_LABELS = {0: "No churn", 1: "Churn"}
CHURN_ORDER = [0, 1]

df["ChurnLabel"] = df[TARGET_BIN].map(CHURN_LABELS)

# 若有自訂配色就用，沒有就 fallback
try:
    CHURN_COLORS = ["#4C72B0", "#DD8452"]  # 藍 / 橘
except NameError:
    CHURN_COLORS = ["#4C72B0", "#DD8452"]

# ---- 3.1.1 核心四大 KPI：Revenue / Minutes / Tenure --------------

core_numeric_1 = [
    "MonthlyRevenue",
    "TotalRecurringCharge",
    "MonthlyMinutes",
    "MonthsInService",
]

fig, axes = plt.subplots(2, 2, figsize=(11, 7))

for ax, col in zip(axes.flat, core_numeric_1):
    if col not in df.columns:
        ax.set_visible(False)
        continue

    sns.violinplot(
        data=df,
        x="ChurnLabel",
        y=col,
        order=CHURN_LABELS.values(),
        palette=CHURN_COLORS,
        inner="quartile",
        cut=0,
        ax=ax,
    )

    ax.set_title(col)
    ax.set_xlabel("")
    ax.set_ylabel(col)

    # 對明顯有長尾的欄位，可以用 log scale（視覺比較好看）
    if col in {"MonthlyRevenue", "TotalRecurringCharge", "MonthlyMinutes"}:
        ax.set_yscale("log")
        ax.set_ylabel(f"{col} (log scale)")

fig.suptitle("Core financial / usage / tenure features by churn", fontsize=14)
plt.tight_layout()

U.save_fig(fig, "F3a_violin_core_financial_usage_tenure")
print("Saved figure: F3a_violin_core_financial_usage_tenure.png")
plt.show()


# %% [markdown]
# # ---- 3.1.2 延伸：OverageMinutes + CurrentEquipmentDays -------------
# 

# %%
# ---- 3.1.2 延伸：OverageMinutes + CurrentEquipmentDays -------------

core_numeric_2 = [
    "OverageMinutes",
    "CurrentEquipmentDays",
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, col in zip(axes.flat, core_numeric_2):
    if col not in df.columns:
        ax.set_visible(False)
        continue

    sns.violinplot(
        data=df,
        x="ChurnLabel",
        y=col,
        order=CHURN_LABELS.values(),
        palette=CHURN_COLORS,
        inner="quartile",
        cut=0,
        ax=ax,
    )

    ax.set_title(col)
    ax.set_xlabel("")
    ax.set_ylabel(col)

    # 這兩個都常常右偏，可以視情況上 log
    if col in {"OverageMinutes", "CurrentEquipmentDays"}:
        ax.set_yscale("log")
        ax.set_ylabel(f"{col} (log scale)")

fig.suptitle("Extended numeric features by churn (overage & equipment age)", fontsize=14)
plt.tight_layout()

U.save_fig(fig, "F3b_violin_overage_equipment")
print("Saved figure: F3b_violin_overage_equipment.png")
plt.show()


# %% [markdown]
# # 3.2 零膨脹數值特徵的流失風險（二元風險條圖）

# %%
# ============================================
# 3.2 零膨脹數值特徵的流失風險（二元風險條圖）
# DroppedCalls, OverageMinutes, CurrentEquipmentDays
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_binary_risk(df, source_col, threshold, ax, title=None):
    """
    把連續欄位轉成 0/1 flag，計算每組的流失率並畫條圖。

    source_col : 原始數值欄位
    threshold  : > threshold 視為 1（例如 >0）
    """
    tmp = df[[source_col, TARGET_BIN]].copy()

    flag_col = f"{source_col}_flag"
    tmp[flag_col] = (tmp[source_col] > threshold).astype(int)

    summary = (
        tmp.groupby(flag_col)[TARGET_BIN]
        .agg(churn_rate="mean", n="size")
        .reset_index()
    )
    summary["churn_rate_pct"] = summary["churn_rate"] * 100
    summary[flag_col] = summary[flag_col].map({0: "0 / No", 1: f">{threshold} / Yes"})

    sns.barplot(
        data=summary,
        x=flag_col,
        y="churn_rate_pct",
        ax=ax,
    )

    for i, row in summary.iterrows():
        ax.text(
            i,
            row["churn_rate_pct"] + 0.5,
            f"{row['churn_rate_pct']:.1f}%\n(n={row['n']})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("")
    ax.set_ylabel("Churn rate (%)")
    ax.set_ylim(0, max(summary["churn_rate_pct"]) * 1.25)

    if title is not None:
        ax.set_title(title)


fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# 1) 斷話有無
if "DroppedCalls" in df.columns:
    plot_binary_risk(
        df,
        source_col="DroppedCalls",
        threshold=0,
        ax=axes[0],
        title="Churn rate by DroppedCalls>0",
    )
else:
    axes[0].set_visible(False)

# 2) 是否有超額分鐘
if "OverageMinutes" in df.columns:
    plot_binary_risk(
        df,
        source_col="OverageMinutes",
        threshold=0,
        ax=axes[1],
        title="Churn rate by OverageMinutes>0",
    )
else:
    axes[1].set_visible(False)

# 3) 設備是否非常老（門檻可依你之後的特徵工程調整）
if "CurrentEquipmentDays" in df.columns:
    EQUIP_OLD_THRESHOLD = 500  # TODO: 之後可調整或改成分位數
    plot_binary_risk(
        df,
        source_col="CurrentEquipmentDays",
        threshold=EQUIP_OLD_THRESHOLD,
        ax=axes[2],
        title=f"Churn rate by EquipmentDays>{EQUIP_OLD_THRESHOLD}",
    )
else:
    axes[2].set_visible(False)

fig.suptitle("Risk lift of zero-inflated numeric features", fontsize=14)
plt.tight_layout()

U.save_fig(fig, "F3c_binary_risk_zero_inflated")
print("Saved figure: F3c_binary_risk_zero_inflated.png")
plt.show()


# %% [markdown]
# # Appendix
# # ============================================
# # 3.1 Key numeric features vs Churn (boxplot grid)

# %%
# Appendix
# ============================================
# 3.1 Key numeric features vs Churn (boxplot grid)
# 目的：
#   - 比較流失 vs 未流失在主要數值特徵上的分佈差異
#   - 幫助挑出「形狀差很多」的強特徵
# 輸出：
#   - F3a_numeric_vs_churn_boxplots.png
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Appendix: Boxplots of additional numeric features by churn status (diagnostic plots)

# --- 3.1.0 先選出要畫的數值特徵 ------------------------------------
# 這裡先手動挑 9 個「財務 + 使用量 + 年資」的重要欄位做 3×3 grid
# 如果有欄位不存在，會自動被過濾掉，不會當掉。
CANDIDATE_NUM_COLS = [
    "MonthlyRevenue",
    "TotalRecurringCharge",
    "MonthlyMinutes",
    "OverageMinutes",
    "MonthsInService",
    "CurrentEquipmentDays",
    "PercChangeRevenues",
    "PercChangeMinutes",
    "UniqueSubs",
]

num_features_for_plot = [c for c in CANDIDATE_NUM_COLS if c in df.columns]
print("Numeric features included in boxplot grid:", num_features_for_plot)

if len(num_features_for_plot) == 0:
    raise ValueError("num_features_for_plot is empty – please adjust CANDIDATE_NUM_COLS.")

# --- 3.1.1 準備繪圖用的資料（把 0/1 目標改成 Yes/No label 比較好看） ----
plot_df = df[[TARGET_BIN] + num_features_for_plot].copy()

# 只保留目標 & 特徵都不是 NaN 的資料
plot_df = plot_df.dropna(subset=[TARGET_BIN])

# 建一個好看的標籤欄位（0=No, 1=Yes）
plot_df["ChurnLabel"] = plot_df[TARGET_BIN].map({0: "No", 1: "Yes"})

# --- 3.1.2 3×3 盒鬚圖 grid -----------------------------------------
n_features = len(num_features_for_plot)
n_cols = 3
n_rows = int(np.ceil(n_features / n_cols))

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(4 * n_cols, 3.5 * n_rows),
    sharex=True
)

axes = np.array(axes).reshape(n_rows, n_cols)  # 方便迴圈

for idx, col in enumerate(num_features_for_plot):
    r = idx // n_cols
    c = idx % n_cols
    ax = axes[r, c]

    # 去掉 NaN，避免 boxplot 畫出奇怪的 warning
    sub = plot_df[["ChurnLabel", col]].dropna()

    # 盒鬚圖（如果你比較喜歡 violin，下面那行改成 sns.violinplot）
    sns.boxplot(
        data=sub,
        x="ChurnLabel",
        y=col,
        ax=ax,
        showfliers=False  # 不畫極端值點，避免遮住主體
    )

    ax.set_title(col, fontsize=11)
    ax.set_xlabel("")  # 用整體 X label 即可
    ax.set_ylabel("")

# 把多出來的空白子圖關掉（如果不是剛好 9 個）
for j in range(idx + 1, n_rows * n_cols):
    r = j // n_cols
    c = j % n_cols
    fig.delaxes(axes[r, c])

# 全圖標題 & 共用標籤
fig.suptitle("Distribution of key numeric features by Churn", fontsize=14, y=0.98)
fig.text(0.5, 0.02, "Churn status", ha="center")
fig.text(0.04, 0.5, "Feature value", va="center", rotation="vertical")

plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

# 儲存圖檔
try:
    U.save_fig(fig, "F3a_numeric_vs_churn_boxplots")
    print("Saved figure: F3a_numeric_vs_churn_boxplots.png")
except Exception:
    fig.savefig("F3a_numeric_vs_churn_boxplots.png", dpi=200, bbox_inches="tight")
    print("Saved figure locally: F3a_numeric_vs_churn_boxplots.png")

plt.show()


#有和 F3a,b 重覆的：MonthlyRevenue, TotalRecurringCharge, MonthlyMinutes, MonthsInService, CurrentEquipmentDays 也有只有這張才有的：PercChangeRevenues, PercChangeMinutes, UniqueSubs

# %% [markdown]
# # 第四章：類別特徵深度洞察
# # 4.0 準備：從 FEATURE_REG 抓出類別特徵

# %%
# ============================================
# Chapter 4: Categorical Feature Insights
# 4.0 用 FEATURE_REG 抓出類別特徵清單 + 重要標籤列表
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("FEATURE_REG columns:", FEATURE_REG.columns.tolist())

# ---- 全域基準流失率 -------------------------------------------------
GLOBAL_CHURN_RATE = df[TARGET_BIN].mean()
print(f"Global churn rate: {GLOBAL_CHURN_RATE:.3%}")

# ---- 4.0.1 抓出 ordinal / nominal + cardinality --------------------

ordinal_cols: list[str] = []
low_card_nominal_cols: list[str] = []
high_card_nominal_cols: list[str] = []

if "primary_type" in FEATURE_REG.columns:
    # 直接用你在 Phase 0 已經標好的 primary_type + cardinality_tag
    ordinal_cols = FEATURE_REG.query("primary_type == 'ordinal'").index.tolist()

    low_card_nominal_cols = FEATURE_REG.query(
        "primary_type == 'nominal' and cardinality_tag == 'low_card'"
    ).index.tolist()

    high_card_nominal_cols = FEATURE_REG.query(
        "primary_type == 'nominal' and cardinality_tag == 'high_card'"
    ).index.tolist()
else:
    # 安全備援：如果沒 primary_type，就用 n_unique 粗略判斷
    for col, row in FEATURE_REG.iterrows():
        if not row.get("is_numeric", False):
            n_uni = row.get("n_unique", np.nan)
            try:
                if n_uni <= 10:
                    low_card_nominal_cols.append(col)
                elif n_uni <= 50:
                    high_card_nominal_cols.append(col)
            except TypeError:
                # n_unique 若不是數字就跳過
                continue

print("[Ordinal cols]        ", ordinal_cols)
print("[Low-card nominal]    ", low_card_nominal_cols)
print("[High-card nominal]   ", high_card_nominal_cols)

# ---- 4.0.2 列出 cardinality_tag / is_sparse_zero / is_momentum / zero_as_missing ----

# 1) cardinality_tag：列出各種 tag 對應到的欄位清單
if "cardinality_tag" in FEATURE_REG.columns:
    print("\n[cardinality_tag groups]")
    for tag in FEATURE_REG["cardinality_tag"].dropna().unique():
        cols = FEATURE_REG.index[FEATURE_REG["cardinality_tag"] == tag].tolist()
        print(f"  - {tag}: {cols}")
else:
    print("\n[cardinality_tag] column not found in FEATURE_REG.")

# 2) is_sparse_zero：大多數是 0 的零膨脹欄位
sparse_zero_cols = []
if "is_sparse_zero" in FEATURE_REG.columns:
    sparse_zero_cols = FEATURE_REG.index[FEATURE_REG["is_sparse_zero"]].tolist()
    print("\n[is_sparse_zero == True] (zero-inflated numeric features)")
    print(" ", sparse_zero_cols)
else:
    print("\n[is_sparse_zero] column not found in FEATURE_REG.")

# 3) is_momentum：變動率 / 動量類欄位（例如 PercChangeRevenues）
momentum_cols = []
if "is_momentum" in FEATURE_REG.columns:
    momentum_cols = FEATURE_REG.index[FEATURE_REG["is_momentum"]].tolist()
    print("\n[is_momentum == True] (rate-of-change / momentum features)")
    print(" ", momentum_cols)
else:
    print("\n[is_momentum] column not found in FEATURE_REG.")

# 4) zero_as_missing：0 被視為缺失的欄位（如 HandsetPrice, IncomeGroup, AgeHH1）
zero_as_missing_cols = []
if "zero_as_missing" in FEATURE_REG.columns:
    zero_as_missing_cols = FEATURE_REG.index[FEATURE_REG["zero_as_missing"]].tolist()
    print("\n[zero_as_missing == True] (0 treated as missing)")
    print(" ", zero_as_missing_cols)
else:
    print("\n[zero_as_missing] column not found in FEATURE_REG.")


# %% [markdown]
# # 4.1 序位變數的趨勢 (Ordinal Trends)

# %% [markdown]
# # 4.1.1 工具函式：計算每個等級的流失率

# %%
# ============================================
# 4.1 Ordinal trends: 等級 vs 流失率
# ============================================

def churn_rate_by_category(df: pd.DataFrame,
                           cat_col: str,
                           target_col: str = TARGET_BIN) -> pd.DataFrame:
    """
    給定一個類別欄位，回傳每一類別的：
      - n: 樣本數
      - churn_rate: 流失率（target=1 的比例）
    """
    g = df.groupby(cat_col)[target_col]
    out = g.agg(
        n="size",
        churn_rate="mean",
    ).reset_index()

    # 排序：若是純數值等級就照數值排；否則照 churn_rate 排
    if pd.api.types.is_numeric_dtype(df[cat_col]):
        out = out.sort_values(cat_col)
    else:
        out = out.sort_values("churn_rate", ascending=False)

    out["churn_rate_pct"] = out["churn_rate"] * 100
    return out


# %% [markdown]
# # 4.1.2 畫出每個 ordinal 欄位的點圖 + 表格

# %%
# 對每個 ordinal 變數畫一張 point plot
n_ord = len(ordinal_cols)
fig, axes = plt.subplots(
    nrows=n_ord,
    ncols=1,
    figsize=(7, max(3, 2.8 * n_ord)),
    sharey=False
)

if n_ord == 1:
    axes = [axes]

ordinal_summaries = []

for ax, col in zip(axes, ordinal_cols):
    summary = churn_rate_by_category(df, col, TARGET_BIN)
    ordinal_summaries.append((col, summary))

    sns.pointplot(
        data=summary,
        x=col,
        y="churn_rate",
        ax=ax,
        marker="o",
    )
    ax.axhline(GLOBAL_CHURN_RATE, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Churn rate")
    ax.set_xlabel(col)
    ax.set_title(f"Churn rate by {col} (ordinal)")

    # 在點上標 n
    for i, row in summary.iterrows():
        ax.text(
            i,
            row["churn_rate"] + 0.005,
            f"n={row['n']}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

plt.tight_layout()
U.save_fig(fig, "F4a_ordinal_trends")
print("Saved figure: F4a_ordinal_trends.png")

# 把各 ordinal 的 summary 合併存成一張長表
rows = []
for col, s in ordinal_summaries:
    tmp = s.copy()
    tmp.insert(0, "feature", col)
    rows.append(tmp)

ordinal_table = pd.concat(rows, ignore_index=True)
U.save_df(ordinal_table, "T4_ordinal_trends")
print("Saved table: T4_ordinal_trends.csv")
display(ordinal_table.head())


# %% [markdown]
# # 4.2 低基數類別：100% 堆疊圖看流失率差異

# %% [markdown]
# # 4.2.1 小工具：把 Churn01 轉成 Yes / No label

# %%
# 建一個字串版的 Churn 標籤，畫堆疊圖用
if "ChurnLabel" not in df.columns:
    df["ChurnLabel"] = df[TARGET_BIN].map({0: "No churn", 1: "Churn"})


# %% [markdown]
# # 4.2.2 工具函式：畫 100% stacked bar

# %%
def plot_stacked_churn_bar(df: pd.DataFrame,
                           cat_col: str,
                           target_label_col: str = "ChurnLabel",
                           min_count: int = 50,
                           ax=None):
    """
    對 low-card nominal 類別畫 100% 堆疊圖：
      - X: 類別
      - Y: 組內 churn / no churn 比例
    只保留樣本數 >= min_count 的類別。
    """
    tmp = (
        df
        .groupby([cat_col, target_label_col])
        .size()
        .rename("n")
        .reset_index()
    )

    # 總數與篩選樣本太少的類別
    total = tmp.groupby(cat_col)["n"].sum().rename("total")
    tmp = tmp.merge(total, on=cat_col)
    tmp = tmp[tmp["total"] >= min_count].copy()

    if tmp.empty:
        print(f"[{cat_col}] has no category with n >= {min_count}, skip.")
        return None

    tmp["prop"] = tmp["n"] / tmp["total"]

    # Pivot 成 100% 堆疊格式
    pivot = tmp.pivot(index=cat_col, columns=target_label_col, values="prop").fillna(0)

    # 排序：依照 churn 率由高到低
    if "Churn" in pivot.columns:
        pivot = pivot.sort_values("Churn", ascending=False)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))

    pivot.plot(kind="bar", stacked=True, ax=ax)

    ax.axhline(GLOBAL_CHURN_RATE, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Proportion within category")
    ax.set_xlabel(cat_col)
    ax.set_title(f"Churn composition by {cat_col} (categories with n ≥ {min_count})")
    ax.legend(title="Status", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    return ax


# %% [markdown]
# # 4.2.3 實際畫出幾個重要的 low-card nominal

# %%
# F4b：Binary / 少數類 flag（Homeownership, HasCreditCard, NewCellphoneUser…）

# 選幾個你最關心的人口/方案類特徵
# 明確指定：想看的「flag / 少數類」特徵
focus_cats = [
    "Homeownership", "MaritalStatus", "HasCreditCard",
    "NonUSTravel", "NewCellphoneUser", "HandsetWebCapable",
]

focus_cats = [c for c in focus_cats if c in df.columns]

n_low = len(focus_cats)
fig, axes = plt.subplots(
    nrows=n_low,
    ncols=1,
    figsize=(7, max(3, 2.7 * n_low)),
)

if n_low == 1:
    axes = [axes]

for ax, col in zip(axes, focus_cats):
    plot_stacked_churn_bar(df, col, target_label_col="ChurnLabel", min_count=100, ax=ax)

plt.tight_layout()
U.save_fig(fig, "F4b_low_card_categorical_stacked")
print("Saved figure: F4b_low_card_categorical_stacked.png")


# %% [markdown]
# # F4c / F4d：Occupation / PrizmCode 的 Top-N churn rate 條圖

# %%
def plot_topN_churn_rate(df, col, target_bin=TARGET_BIN, top_n=8, min_count=200, fig_name=None):
    tmp = (
        df.groupby(col)[target_bin]
        .agg(churn_rate="mean", n="count")
        .reset_index()
    )
    # 只保留樣本數夠多的類別
    tmp = tmp[tmp["n"] >= min_count]

    if tmp.empty:
        print(f"[{col}] has no category with n >= {min_count}, skip.")
        return

    tmp = tmp.sort_values("churn_rate", ascending=False).head(top_n)
    tmp["churn_pct"] = (tmp["churn_rate"] * 100).round(1)

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(tmp) + 2))
    sns.barplot(data=tmp, x="churn_rate", y=col, ax=ax)

    ax.set_xlabel("Churn rate")
    ax.set_ylabel(col)
    ax.set_title(f"Top {len(tmp)} {col} categories by churn rate")

    # 畫一條全體平均的垂直線
    ax.axvline(GLOBAL_CHURN_RATE, linestyle="--", linewidth=1, label="Overall churn rate")
    ax.legend()

    for i, row in tmp.iterrows():
        ax.text(
            row["churn_rate"] + 0.002,
            i,
            f"{row['churn_pct']}% (n={row['n']})",
            va="center",
        )

    plt.tight_layout()
    if fig_name is not None:
        U.save_fig(fig, fig_name)
        print(f"Saved figure: {fig_name}.png")
    plt.show()


# 實際畫圖
plot_topN_churn_rate(df, "Occupation", fig_name="F4c_topN_churn_by_occupation")
plot_topN_churn_rate(df, "PrizmCode", fig_name="F4d_topN_churn_by_prizm")




# %% [markdown]
# # 在表中加入 lift（相對於整體流失率的差距）

# %% [markdown]
# # 風險偏差圖：左右向的 lift（Deviation plot）

# %%
# ============================================
# 4.3 Low-card categorical: churn lift vs global rate
# 重新建立 T4_low_card_categorical_churn + 畫 deviation plot
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

GLOBAL_CHURN_RATE = df[TARGET_BIN].mean()

# 想要分析的低基數 / flag 類別
focus_for_deviation = [
    "Homeownership",
    "MaritalStatus",
    "HasCreditCard",
    "NonUSTravel",
    "NewCellphoneUser",
    "HandsetWebCapable",
    "Occupation",
    "PrizmCode",
]
focus_for_deviation = [c for c in focus_for_deviation if c in df.columns]


def churn_rate_by_category(df: pd.DataFrame,
                           cat_col: str,
                           target_bin: str = TARGET_BIN,
                           min_count: int = 50) -> pd.DataFrame:
    """
    給一個類別欄位，計算：
      - 每一類的樣本數 n
      - 流失率 churn_rate
      - churn_rate_pct (%)
      - lift_pctpt: churn_rate - global_rate (百分點差距)
    只保留 n >= min_count 的類別。
    """
    g = df.groupby(cat_col)[target_bin]
    out = g.agg(n="size", churn_rate="mean").reset_index()
    out = out[out["n"] >= min_count].copy()
    if out.empty:
        return out

    out["churn_rate_pct"] = out["churn_rate"] * 100
    out["lift_pctpt"] = (out["churn_rate"] - GLOBAL_CHURN_RATE) * 100
    out = out.rename(columns={cat_col: "category"})
    return out


# -------- 建 T4_low_card_categorical_churn 表 --------
rows = []
for col in focus_for_deviation:
    tmp = churn_rate_by_category(df, col, TARGET_BIN, min_count=50)
    if tmp.empty:
        continue
    tmp.insert(0, "feature", col)
    rows.append(tmp)

tbl_low = pd.concat(rows, ignore_index=True)
U.save_df(tbl_low, "T4_low_card_categorical_churn")
print("Saved T4_low_card_categorical_churn.csv")
display(tbl_low.head())


# -------- F4e: deviation plot（修好文字位置） --------
n_f = len(focus_for_deviation)
fig, axes = plt.subplots(
    nrows=n_f,
    ncols=1,
    figsize=(6, 2.4 * n_f),
)
if n_f == 1:
    axes = [axes]

for ax, feat in zip(axes, focus_for_deviation):
    df_plot = tbl_low.query("feature == @feat").copy()
    if df_plot.empty:
        ax.axis("off")
        ax.set_title(f"{feat}: no category with enough samples")
        continue

    # 依 lift 排序
    df_plot = df_plot.sort_values("lift_pctpt")

    sns.barplot(
        data=df_plot,
        x="lift_pctpt",
        y="category",
        ax=ax,
        color="#3274A1",
    )

    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{feat}: churn lift vs global rate")
    ax.set_xlabel("Lift vs global churn rate (percentage points)")
    ax.set_ylabel("")

    # 取得 y 位置，確保文字對齊 bar
    y_labels = [t.get_text() for t in ax.get_yticklabels()]
    y_pos = ax.get_yticks()
    pos_map = dict(zip(y_labels, y_pos))

    max_abs_lift = df_plot["lift_pctpt"].abs().max()
    pad = max_abs_lift * 0.08  # bar 與文字的水平距離

    for _, row in df_plot.iterrows():
        lift = row["lift_pctpt"]
        cat = str(row["category"])
        y = pos_map.get(cat, 0)

        # 右邊（>0）文字放右側、左對齊；左邊（<0）文字放左側、右對齊
        if lift >= 0:
            x_text = lift + pad
            ha = "left"
        else:
            x_text = lift - pad
            ha = "right"

        ax.text(
            x_text,
            y,
            f"{row['churn_rate_pct']:.1f}% (n={row['n']})",
            va="center",
            ha=ha,
            fontsize=8,
        )

    # x 軸預留空間給文字
    lim = max_abs_lift * 1.4
    ax.set_xlim(-lim, lim)

plt.tight_layout()
U.save_fig(fig, "F4e_low_card_deviation_lift")
print("Saved figure: F4e_low_card_deviation_lift.png")
plt.show()


# seprate Occupation and PrizmCode from the plot to a single graph

# %% [markdown]
# # 4.3 高基數類別：Top N 高風險類別條圖

# %% [markdown]
# # 4.3.1 工具函式：找出 Top N 高流失類別

# %%
def top_churn_categories(df: pd.DataFrame,
                         cat_col: str,
                         target_col: str = TARGET_BIN,
                         min_count: int = 200,
                         top_n: int = 8) -> pd.DataFrame:
    """
    對高基數類別欄位：
      - 計算每一類的樣本數與流失率
      - 只保留樣本數 >= min_count
      - 回傳流失率最高的 top_n 類別
    """
    g = df.groupby(cat_col)[target_col]
    out = g.agg(n="size", churn_rate="mean").reset_index()
    out = out[out["n"] >= min_count].copy()
    out["churn_rate_pct"] = out["churn_rate"] * 100
    out = out.sort_values("churn_rate", ascending=False).head(top_n)
    return out


# %% [markdown]
# # 4.3.2 畫出ServiceArea  的 Top N

# %%
# ============================================
# 4.3 High-card categorical feature: ServiceArea
#    Head vs Tail churn analysis
# ============================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def churn_head_tail(
    df: pd.DataFrame,
    cat_col: str,
    target_col: str = TARGET_BIN,
    min_count: int = 200,
    top_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    對高基數類別欄位：
      - 計算每一類的樣本數與流失率
      - 只保留樣本數 >= min_count
      - 回傳：
          bottom_df: 流失率最低的前 top_n 類別
          top_df   : 流失率最高的前 top_n 類別
    """
    if cat_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    g = df.groupby(cat_col)[target_col]
    out = g.agg(n="size", churn_rate="mean").reset_index()

    # 過濾樣本數不足的類別
    out = out[out["n"] >= min_count].copy()
    if out.empty:
        return pd.DataFrame(), pd.DataFrame()

    out["churn_rate_pct"] = out["churn_rate"] * 100

    # 依流失率由低到高排序
    out = out.sort_values("churn_rate", ascending=True)

    # bottom N（最穩定）
    bottom_df = out.head(top_n).copy().reset_index(drop=True)

    # top N（最危險）
    top_df = out.tail(top_n).copy().iloc[::-1].reset_index(drop=True)

    return bottom_df, top_df


# 用 registry 標記的 high-card 欄位（目前就是 ['ServiceArea']）
focus_high_card = high_card_nominal_cols[:]   # 例如 ['ServiceArea']
print("[High-card nominal used in F4c/F4d]:", focus_high_card)

if not focus_high_card:
    print("No high-card nominal categorical features tagged; skip 4.3.")
else:
    for col in focus_high_card:
        bottom_tbl, top_tbl = churn_head_tail(
            df,
            cat_col=col,
            target_col=TARGET_BIN,
            min_count=200,
            top_n=5,
        )

        if bottom_tbl.empty or top_tbl.empty:
            print(f"{col}: insufficient data (after min_count filter).")
            continue

        # ---- 繪圖：左邊 bottom 5，右邊 top 5 ------------------------
        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(12, 4.5),
            constrained_layout=True,
        )

        # 方便之後設定共同 x 範圍
        max_val = max(bottom_tbl["churn_rate_pct"].max(),
                      top_tbl["churn_rate_pct"].max())

        # ===== 左：流失率最低的 5 個類別 =====
        ax_left = axes[0]
        sns.barplot(
            data=bottom_tbl,
            x="churn_rate_pct",
            y=col,
            ax=ax_left,
            color="#4C72B0",
        )
        ax_left.axvline(
            GLOBAL_CHURN_RATE * 100,
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Global churn rate",
        )
        ax_left.set_title(f"Lowest churn {col} categories (bottom {len(bottom_tbl)})")
        ax_left.set_xlabel("Churn rate (%)")
        ax_left.set_ylabel(col)

        for i, (_, row) in enumerate(bottom_tbl.iterrows()):
            ax_left.text(
                row["churn_rate_pct"] + max_val * 0.02,
                i,
                f"{row['churn_rate_pct']:.1f}% (n={row['n']})",
                va="center",
                fontsize=8,
            )

        # ===== 右：流失率最高的 5 個類別 =====
        ax_right = axes[1]
        sns.barplot(
            data=top_tbl,
            x="churn_rate_pct",
            y=col,
            ax=ax_right,
            color="#C44E52",
        )
        ax_right.axvline(
            GLOBAL_CHURN_RATE * 100,
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Global churn rate",
        )
        ax_right.set_title(f"Highest churn {col} categories (top {len(top_tbl)})")
        ax_right.set_xlabel("Churn rate (%)")
        ax_right.set_ylabel("")

        for i, (_, row) in enumerate(top_tbl.iterrows()):
            ax_right.text(
                row["churn_rate_pct"] + max_val * 0.02,
                i,
                f"{row['churn_rate_pct']:.1f}% (n={row['n']})",
                va="center",
                fontsize=8,
            )

        # 兩邊共用一個比較合理的 x 軸範圍
        for ax in axes:
            ax.set_xlim(0, max_val * 1.25)

        fig.suptitle(f"Head vs Tail churn analysis for {col}", fontsize=14)
        U.save_fig(fig, f"F4c_high_card_head_tail_{col}")
        print(f"Saved figure: F4c_high_card_head_tail_{col}.png")
        plt.show()
        plt.close(fig)


# %%
rows = []
for col in high_card_nominal_cols:
    tmp = churn_rate_by_category(df, col, TARGET_BIN)
    tmp.insert(0, "feature", col)
    rows.append(tmp)

high_card_table = pd.concat(rows, ignore_index=True)
U.save_df(high_card_table, "T4_high_card_categorical_churn")
print("T4_high_card_categorical_churn.csv")
display(high_card_table.head())


# %% [markdown]
# # 4.4 Categorical–target association (Cramer's V)

# %%
# ============================================
# 4.4 Categorical–target association (Cramer's V)
# 目的：
#   - 量化每個類別特徵跟 Churn01 的關聯強度
#   - Cramer's V ∈ [0,1]，越接近 1 代表關聯越強
# 輸出：
#   - T4_cramers_v_vs_churn.csv：所有 is_categorical 特徵的 Cramer's V
#   - F4f_cramers_v_top_categorical.png：Top-K 排名長條圖
#   - F4g_cramers_v_heatmap.png：同一組 Top-K 的熱圖版本（可選）
# ============================================

import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    計算兩個類別變數之間的 Cramer's V.
    這裡只拿 x vs Churn01 用，所以是「單一特徵 vs 目標」的關聯度量。
    """
    tab = pd.crosstab(x, y)

    # 如果 contingency table 太小（某一邊只有一類），就沒有意義，回傳 NaN
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan

    chi2, _, _, _ = chi2_contingency(tab)
    n = tab.to_numpy().sum()
    r, k = tab.shape

    # 最常見的簡單公式：sqrt(chi2 / (n * (min(r-1, k-1))))
    denom = n * (min(r - 1, k - 1))
    if denom <= 0:
        return np.nan

    return float(np.sqrt(chi2 / denom))


# 1) 從 FEATURE_REG 抓出所有「類別特徵」
cat_cols = FEATURE_REG.index[FEATURE_REG["is_categorical"]].tolist()

# 排除 ID / 目標欄位
cat_cols = [c for c in cat_cols if c not in {ID_COL, TARGET, TARGET_BIN}]
print("[Categorical cols for Cramer's V]:", cat_cols)

# 2) 對每個類別特徵計算 Cramer's V
records = []
for col in cat_cols:
    v = cramers_v(df[col], df[TARGET_BIN])

    records.append(
        {
            "feature": col,
            "cramers_v": v,
            "primary_type": FEATURE_REG.loc[col, "primary_type"],
            "semantic_group": FEATURE_REG.loc[col, "semantic_group"],
            "cardinality_tag": FEATURE_REG.loc[col, "cardinality_tag"],
        }
    )

assoc_tbl = pd.DataFrame(records).dropna()
assoc_tbl = assoc_tbl.sort_values("cramers_v", ascending=False).reset_index(drop=True)

print("Cramer's V table shape:", assoc_tbl.shape)
display(assoc_tbl.head(20))

# 存成表格，方便在論文或後續 notebook 用數字
U.save_df(assoc_tbl, "T4_cramers_v_vs_churn")
print("Saved table: T4_cramers_v_vs_churn.csv")


# 3) 視覺化：Top-K 類別特徵的 Cramer's V 長條圖
top_k = 15   # 你可以改成 10, 20 等
plot_tbl = assoc_tbl.head(top_k).copy()

fig, ax = plt.subplots(figsize=(7, 0.45 * len(plot_tbl) + 1.5))

sns.barplot(
    data=plot_tbl,
    x="cramers_v",
    y="feature",
    ax=ax,
)

ax.set_title("Association of categorical features with churn (Cramer's V)")
ax.set_xlabel("Cramer's V (0–1)")
ax.set_ylabel("Feature")

# 在每條 bar 右側標出數值
for i, row in plot_tbl.iterrows():
    ax.text(
        row["cramers_v"] + 0.01,
        i,
        f"{row['cramers_v']:.3f}",
        va="center",
        fontsize=8,
    )

plt.tight_layout()
U.save_fig(fig, "F4f_cramers_v_top_categorical")
print("Saved figure: F4f_cramers_v_top_categorical.png")
plt.show()


# 4)（可選）把同一組 Top-K 做成「一列 heatmap」
heat_data = plot_tbl.set_index("feature")[["cramers_v"]]

fig, ax = plt.subplots(figsize=(4, 0.45 * len(heat_data) + 1.5))

sns.heatmap(
    heat_data,
    annot=True,
    fmt=".3f",
    cmap="viridis",
    cbar_kws={"label": "Cramer's V"},
    ax=ax,
)

ax.set_title("Churn association heatmap (Cramer's V)")
ax.set_xlabel("")
ax.set_ylabel("")

plt.tight_layout()
U.save_fig(fig, "F4g_cramers_v_heatmap")
print("Saved figure: F4g_cramers_v_heatmap.png")
plt.show()


# %% [markdown]
# # 第五章：多變數 & 特徵工程驗證

# %%
# ============================================
# 5.1 Numeric correlation & redundancy
# 5.1.1 選出要檢查的數值特徵（依與 Churn01 的相關性）
# ============================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1) 從 FEATURE_REG 抓出 numeric features（排除 ID / target）
numeric_all = (
    FEATURE_REG
    .query("is_numeric == True")
    .index.tolist()
)

numeric_all = [
    c for c in numeric_all
    if c not in {ID_COL, TARGET, TARGET_BIN}
]

print(f"[All numeric candidates] {len(numeric_all)} columns")
print(numeric_all)

# 2) 計算這些 numeric features 與 Churn01 的 Spearman 相關
#    只留絕對值最大的前 top_k 個，避免圖太亂
top_k = 25

corr_with_target = (
    df[numeric_all + [TARGET_BIN]]
    .corr(method="spearman")[TARGET_BIN]
    .drop(TARGET_BIN)          # 拿掉自己
    .dropna()
)

corr_with_target_abs = corr_with_target.abs().sort_values(ascending=False)
top_numeric = corr_with_target_abs.head(top_k).index.tolist()

print(f"\n[Top {len(top_numeric)} numeric features by |Spearman with {TARGET_BIN}|]:")
for c in top_numeric:
    print(f"  {c:25s}  Spearman={corr_with_target[c]: .3f}")

# 之後相關性矩陣與熱圖都用 top_numeric 這個清單


# %% [markdown]
# # # 5.1.2 Spearman correlation matrix + triangle heatmap (F5a)

# %%
# ============================================
# 5.1.2 Spearman correlation matrix + triangle heatmap (F5a)
# ============================================

# 計算 Spearman 相關係數矩陣
corr_mat = df[top_numeric].corr(method="spearman")

# 也可以把整個矩陣存起來（選擇性）
# U.save_df(corr_mat, "T5_numeric_spearman_corr_matrix")
# print("Saved: T5_numeric_spearman_corr_matrix.csv")

# 建立「遮住上三角」的 mask，只顯示下三角
mask = np.triu(np.ones_like(corr_mat, dtype=bool))

fig, ax = plt.subplots(
    figsize=(0.6 * len(top_numeric) + 4, 0.6 * len(top_numeric) + 2)
)

sns.heatmap(
    corr_mat,
    mask=mask,           # 遮上半部，只看下三角
    cmap="coolwarm",
    vmin=-1, vmax=1,
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"label": "Spearman correlation"},
    ax=ax,
)

ax.set_title("Spearman correlation (top numeric features)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
U.save_fig(fig, "F5a_numeric_correlation_triangle")
print("Saved figure: F5a_numeric_correlation_triangle.png")
plt.show()
plt.close(fig)


# %% [markdown]
# # 5.1.3 高相關 pair 摘要表 (T5a_high_corr_pairs)

# %%
# ============================================
# 5.1.3 高相關 pair 摘要表 (T5a_high_corr_pairs)
#   - 從 corr_mat 抓出 |corr| >= 0.7 的成對特徵
# ============================================

threshold = 0.7

# 把對角線設成 NaN，避免被抓到
corr_no_diag = corr_mat.copy()
np.fill_diagonal(corr_no_diag.values, np.nan)

# 轉成 long format
corr_long = (
    corr_no_diag
    .stack()                  # 變成 (row,col,value)
    .reset_index()
)
corr_long.columns = ["feature_1", "feature_2", "spearman_corr"]

# 為了只保留一次 (A,B)，丟掉重複的 (B,A)：
# 可以要求 feature_1 < feature_2（字典順序）
corr_long["pair_sorted"] = corr_long.apply(
    lambda r: tuple(sorted([r["feature_1"], r["feature_2"]])),
    axis=1,
)
corr_long = (
    corr_long
    .drop_duplicates("pair_sorted")
    .drop(columns="pair_sorted")
)

# 套用門檻
high_corr_pairs = corr_long[
    corr_long["spearman_corr"].abs() >= threshold
].sort_values(
    "spearman_corr",
    ascending=False
).reset_index(drop=True)

print(f"High-corr pairs (|Spearman| >= {threshold}): {len(high_corr_pairs)}")
display(high_corr_pairs)

# 存成 T5a，方便論文引用
U.save_df(high_corr_pairs, "T5a_high_corr_pairs")
print("Saved table: T5a_high_corr_pairs.csv")


# %% [markdown]
# # 5.2.1 建立工程特徵欄位（只在 EDA 用）

# %%
# ============================================
# 5.2.1 建立工程特徵（只在 EDA notebook 內使用）
# ============================================
import numpy as np

# InactiveSubs = UniqueSubs - ActiveSubs，理論上不應為負數，所以 clip 到 0
df["InactiveSubs"] = (df["UniqueSubs"] - df["ActiveSubs"]).clip(lower=0)

# RevenuePerMinute = 每分鐘帳單金額
# +1 避免除以 0，同時不會對大多數樣本產生太大影響
df["RevenuePerMinute"] = df["MonthlyRevenue"] / (df["MonthlyMinutes"] + 1)

# MOU_perActive = 每一個 active subscriber 的平均通話分鐘數
df["MOU_perActive"] = df["MonthlyMinutes"] / (df["ActiveSubs"] + 1)

print("Engineered features created:")
for col in ["InactiveSubs", "RevenuePerMinute", "MOU_perActive"]:
    print(f"  - {col}, na_rate={df[col].isna().mean():.3f}")


# %% [markdown]
# # 5.2.2 通用 decile risk curve 函式

# %%
# ============================================
# 5.2.2 Decile risk curve helper
# ============================================
import pandas as pd
import matplotlib.pyplot as plt

def make_decile_risk_table(
    df: pd.DataFrame,
    feature: str,
    target: str = TARGET_BIN,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    將 feature 依分位數切成 n_bins（deciles），
    回傳每一 bin 的：
      - n: 樣本數
      - churn_rate: 平均流失率
      - feature_min / max / median: 該 bin 的數值範圍
      - decile: 1 ~ n_bins
    """
    tmp = df[[feature, target]].copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()

    if tmp.empty:
        return pd.DataFrame()

    # 依分位數切 bin；duplicates="drop" 避免極端常數導致 bin 數不足時報錯
    tmp["bin"] = pd.qcut(
        tmp[feature],
        q=n_bins,
        labels=False,
        duplicates="drop",
    )

    grouped = tmp.groupby("bin").agg(
        n=(target, "size"),
        churn_rate=(target, "mean"),
        feature_min=(feature, "min"),
        feature_max=(feature, "max"),
        feature_median=(feature, "median"),
    ).reset_index(drop=True)

    grouped["decile"] = np.arange(1, len(grouped) + 1)
    grouped["churn_pct"] = grouped["churn_rate"] * 100

    return grouped[[
        "decile", "n", "churn_rate", "churn_pct",
        "feature_min", "feature_max", "feature_median"
    ]]


def plot_decile_risk_curve(
    df: pd.DataFrame,
    feature: str,
    target: str = TARGET_BIN,
    n_bins: int = 10,
    ax: plt.Axes | None = None,
    label: str | None = None,
):
    """
    畫出某個數值特徵的 decile risk curve。
    X 軸：decile (1~10)
    Y 軸：churn rate (%)
    """
    tbl = make_decile_risk_table(df, feature, target, n_bins)

    if tbl.empty:
        print(f"[{feature}] has no valid data for decile curve, skip.")
        return None, None

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    lbl = label if label is not None else feature

    ax.plot(
        tbl["decile"],
        tbl["churn_pct"],
        marker="o",
        linestyle="-",
        label=lbl,
    )

    # 全體平均 churn 當 baseline
    ax.axhline(GLOBAL_CHURN_RATE * 100, color="gray", linestyle="--", linewidth=1)

    ax.set_xticks(tbl["decile"])
    ax.set_xlabel("Decile (1 = lowest, 10 = highest)")
    ax.set_ylabel("Churn rate (%)")

    # 簡單標註幾個點（例如第 1 / 5 / 10 decile）
    for k in [0, len(tbl)//2, len(tbl)-1]:
        row = tbl.iloc[k]
        ax.text(
            row["decile"],
            row["churn_pct"] + 0.3,
            f"{row['churn_pct']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    return fig, ax


# %%


# %%


# %% [markdown]
# # 5.2.3 原始特徵 vs 工程特徵：風險曲線對比

# %%
# ============================================
# 5.2.3 (1) MonthlyRevenue vs RevenuePerMinute
# F5b：月費 vs 每分鐘費用
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

plot_decile_risk_curve(df, "MonthlyRevenue", TARGET_BIN, n_bins=10, ax=axes[0],
                       label="MonthlyRevenue")
axes[0].set_title("Risk curve: MonthlyRevenue")

plot_decile_risk_curve(df, "RevenuePerMinute", TARGET_BIN, n_bins=10, ax=axes[1],
                       label="RevenuePerMinute")
axes[1].set_title("Risk curve: RevenuePerMinute")

for ax in axes:
    ax.legend()

plt.tight_layout()
U.save_fig(fig, "F5b_risk_curves_revenue_vs_rpm")
print("Saved figure: F5b_risk_curves_revenue_vs_rpm.png")
plt.show()


# %%
# ============================================
# 5.2.x Create engineered features for risk curves
# ============================================
import numpy as np

# InactiveSubs = UniqueSubs - ActiveSubs
df["InactiveSubs"] = df["UniqueSubs"] - df["ActiveSubs"]

# MOU_perActive = MonthlyMinutes / ActiveSubs (avoid divide-by-zero)
df["MOU_perActive"] = np.where(
    df["ActiveSubs"] > 0,
    df["MonthlyMinutes"] / df["ActiveSubs"],
    np.nan
)

print("NA rates:")
print(df[["InactiveSubs", "MOU_perActive"]].isna().mean().round(4))


# %% [markdown]
# # 5.2.x InactiveSubs decile risk curve

# %%
# ============================================
# 5.2.x InactiveSubs decile risk curve
# F5c_inactivesubs_decile_risk.png
# ============================================

fig, ax = plt.subplots(figsize=(6, 4))

fig, ax = plot_decile_risk_curve(
    df=df,
    feature="InactiveSubs",
    target=TARGET_BIN,      # <-- use 'target', not 'target_col'
    n_bins=10,
    ax=ax,
    label="InactiveSubs"
)

ax.set_title("Risk curve: InactiveSubs")

plt.tight_layout()
U.save_fig(fig, "F5c_inactivesubs_decile_risk")
print("Saved figure: F5c_inactivesubs_decile_risk.png")
plt.show()


# %% [markdown]
# # 5.2.x MOU_perActive decile risk curve

# %%
# ============================================
# 5.2.x MOU_perActive decile risk curve
# F5c_mou_per_active_decile_risk.png
# ============================================

fig, ax = plt.subplots(figsize=(6, 4))

fig, ax = plot_decile_risk_curve(
    df=df,
    feature="MOU_perActive",
    target=TARGET_BIN,      # <-- same here
    n_bins=10,
    ax=ax,
    label="MOU_perActive"
)

ax.set_title("Risk curve: MOU_perActive")

plt.tight_layout()
U.save_fig(fig, "F5c_mou_per_active_decile_risk")
print("Saved figure: F5c_mou_per_active_decile_risk.png")
plt.show()


# %%


# %% [markdown]
# # 5.3.1 合約 × 手機功能：2×2 風險熱圖

# %%
# ============================================
# 5.3.1 Contract / device flags: 2x2 churn heatmap
#   Example: UpgradeOverdue × HandsetWebCapable
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 建立 UpgradeOverdue flag（合約到期 & 未換機） ---
# 條件可以依你之後模型微調，這裡先用一個直覺版本：
#   - MonthsInService >= 24 個月
#   - CurrentEquipmentDays >= 500 天
if "UpgradeOverdue" not in df.columns:
    df["UpgradeOverdue"] = np.where(
        (df["MonthsInService"] >= 24) & (df["CurrentEquipmentDays"] >= 500),
        "Yes",
        "No",
    )

print(df["UpgradeOverdue"].value_counts(dropna=False))

def plot_flag_2x2_heatmap(
    df: pd.DataFrame,
    row_flag: str,
    col_flag: str,
    target_col: str = TARGET_BIN,
    min_count: int = 100,
    fig_name: str | None = None,
):
    """
    繪製 2x2 / 小型類別 × 類別 的流失率熱圖：
      - 每一格顏色 = churn rate (%)
      - 標註 "xx.x% (n=xxx)"
    min_count：樣本數太少的 cell 會標成空白，不強行解讀
    """
    tmp = df[[row_flag, col_flag, target_col]].copy()
    tmp[row_flag] = tmp[row_flag].fillna("Unknown")
    tmp[col_flag] = tmp[col_flag].fillna("Unknown")

    # 流失率表 & 樣本數表
    tbl_rate = pd.crosstab(
        tmp[row_flag], tmp[col_flag],
        values=tmp[target_col],
        aggfunc="mean",
    )
    tbl_n = pd.crosstab(tmp[row_flag], tmp[col_flag])

    # 轉成百分比
    tbl_pct = tbl_rate * 100

    # 樣本數太少的格子用 NaN 蓋掉，不畫顏色
    tbl_plot = tbl_pct.where(tbl_n >= min_count)

    # 準備 annotation 文字
    annot = np.empty_like(tbl_plot.values, dtype=object)
    for i in range(tbl_plot.shape[0]):
        for j in range(tbl_plot.shape[1]):
            val = tbl_plot.iloc[i, j]
            n_ij = tbl_n.iloc[i, j]
            if pd.isna(val):
                annot[i, j] = ""
            else:
                annot[i, j] = f"{val:.1f}%\n(n={n_ij})"

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    sns.heatmap(
        tbl_plot,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=max(40, np.nanmax(tbl_plot.values)),
        cbar_kws={"label": "Churn rate (%)"},
        ax=ax,
    )

    ax.set_xlabel(col_flag)
    ax.set_ylabel(row_flag)
    ax.set_title(f"{row_flag} × {col_flag} : churn heatmap")

    plt.tight_layout()
    if fig_name is not None:
        U.save_fig(fig, fig_name)
        print(f"Saved figure: {fig_name}.png")
    plt.show()
    plt.close(fig)


# --- 實際畫幾個關鍵組合 ---
plot_flag_2x2_heatmap(
    df,
    row_flag="UpgradeOverdue",
    col_flag="HandsetWebCapable",
    fig_name="F5d_upgrade_overdue_x_webcapable",
)

# Optional：再加一組你有興趣的交互作用，例如：
plot_flag_2x2_heatmap(
    df,
    row_flag="UpgradeOverdue",
    col_flag="HandsetRefurbished",
    fig_name="F5e_upgrade_overdue_x_refurbished",
)


# %% [markdown]
# # 5.3.2 RevenuePerMinute × IncomeBand 的分群風險曲線

# %%
# ============================================
# 5.3.2 Numeric × Categorical:
#   RevenuePerMinute × IncomeBand 的分群 decile risk curve
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 確保工程特徵存在（若 5.2 已建立，這裡會直接跳過） ---
if "RevenuePerMinute" not in df.columns:
    df["RevenuePerMinute"] = df["MonthlyRevenue"] / (df["MonthlyMinutes"] + 1)

if "MOU_perActive" not in df.columns:
    df["MOU_perActive"] = df["MonthlyMinutes"] / (df["ActiveSubs"] + 1)

if "InactiveSubs" not in df.columns:
    tmp_diff = df["UniqueSubs"] - df["ActiveSubs"]
    df["InactiveSubs"] = tmp_diff.clip(lower=0)

# --- 建立 IncomeBand（Low / Mid / High / Unknown） ---
def _map_income_band(v):
    if pd.isna(v) or v <= 0:
        return "Unknown"
    elif v <= 3:
        return "Low"
    elif v <= 6:
        return "Mid"
    else:
        return "High"

df["IncomeBand"] = df["IncomeGroup"].apply(_map_income_band)
print(df["IncomeBand"].value_counts(dropna=False))


def plot_segmented_decile_curve(
    df: pd.DataFrame,
    feature: str,
    segment_col: str,
    target_col: str = TARGET_BIN,
    n_bins: int = 10,
    segment_order: list[str] | None = None,
    fig_name: str | None = None,
    title: str | None = None,
):
    """
    Numeric × Categorical 的分群 decile risk curve：
      - 全體一起依 feature 做 qcut 分成 n_bins 個等分位
      - 每個 decile × segment 計算 churn rate
    """
    tmp = df[[feature, segment_col, target_col]].dropna().copy()
    if tmp.empty:
        print(f"[{feature}] no data after dropna; skip.")
        return

    # 用全體資料對 feature 做 qcut
    try:
        tmp["decile"] = pd.qcut(
            tmp[feature],
            q=n_bins,
            labels=range(1, n_bins + 1),
            duplicates="drop",
        )
    except ValueError as e:
        print(f"qcut failed for {feature}: {e}")
        return

    # 計算每個 decile × segment 的流失率
    grp = (
        tmp.groupby(["decile", segment_col])[target_col]
        .agg(churn_rate="mean", n="size")
        .reset_index()
    )
    grp["churn_pct"] = grp["churn_rate"] * 100

    # 決定 decile 序
    deciles = sorted(grp["decile"].unique())
    decile_int = {d: i + 1 for i, d in enumerate(deciles)}
    grp["decile_idx"] = grp["decile"].map(decile_int)

    # segment 顯示順序
    if segment_order is None:
        segment_order = ["Low", "Mid", "High", "Unknown"]
    segment_order = [s for s in segment_order if s in grp[segment_col].unique()]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for seg in segment_order:
        sub = grp[grp[segment_col] == seg].sort_values("decile_idx")
        if sub.empty:
            continue
        ax.plot(
            sub["decile_idx"],
            sub["churn_rate"],
            marker="o",
            label=f"{seg}",
        )

        # 在最後一個點標註大約的流失率區間
        last = sub.iloc[-1]
        ax.text(
            last["decile_idx"] + 0.1,
            last["churn_rate"],
            f"{seg}",
            va="center",
            fontsize=9,
        )

    ax.axhline(GLOBAL_CHURN_RATE, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(range(1, len(deciles) + 1))
    ax.set_xlabel("Decile (Low → High)")
    ax.set_ylabel("Churn rate")

    if title is None:
        title = f"{feature} decile risk curves by {segment_col}"
    ax.set_title(title)
    ax.legend(
        title=segment_col,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
    )

    plt.tight_layout()
    if fig_name is not None:
        U.save_fig(fig, fig_name)
        print(f"Saved figure: {fig_name}.png")
    plt.show()
    plt.close(fig)


# --- 實際畫：RevenuePerMinute × IncomeBand ---
plot_segmented_decile_curve(
    df,
    feature="RevenuePerMinute",
    segment_col="IncomeBand",
    n_bins=10,
    segment_order=["Low", "Mid", "High"],
    fig_name="F5f_rpm_deciles_by_income_band",
    title="RevenuePerMinute decile risk by income band",
)


# %% [markdown]
# # 5.3.3 類別 × 類別：2D 風險熱圖（例：HandsetWebCapable × PrizmCode）

# %%
# ============================================
# 5.3.3 Categorical × Categorical:
#   Example: HandsetWebCapable × PrizmCode 的 churn heatmap
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_categorical_interaction_heatmap(
    df: pd.DataFrame,
    row_cat: str,
    col_cat: str,
    target_col: str = TARGET_BIN,
    min_count: int = 200,
    fig_name: str | None = None,
):
    """
    類別 × 類別 的 2D churn heatmap：
      - index = row_cat, columns = col_cat
      - 顏色 = churn rate (%)
      - 標註 "xx.x% (n=xxx)"，樣本數 < min_count 則留白
    """
    tmp = df[[row_cat, col_cat, target_col]].copy()
    tmp[row_cat] = tmp[row_cat].fillna("Unknown")
    tmp[col_cat] = tmp[col_cat].fillna("Unknown")

    rate_tbl = pd.crosstab(
        tmp[row_cat], tmp[col_cat],
        values=tmp[target_col],
        aggfunc="mean",
    )
    n_tbl = pd.crosstab(tmp[row_cat], tmp[col_cat])

    pct_tbl = rate_tbl * 100
    plot_tbl = pct_tbl.where(n_tbl >= min_count)

    annot = np.empty_like(plot_tbl.values, dtype=object)
    for i in range(plot_tbl.shape[0]):
        for j in range(plot_tbl.shape[1]):
            val = plot_tbl.iloc[i, j]
            n_ij = n_tbl.iloc[i, j]
            if pd.isna(val):
                annot[i, j] = ""
            else:
                annot[i, j] = f"{val:.1f}%\n(n={n_ij})"

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    sns.heatmap(
        plot_tbl,
        annot=annot,
        fmt="",
        cmap="PuBu",
        vmin=0,
        vmax=max(40, np.nanmax(plot_tbl.values)),
        cbar_kws={"label": "Churn rate (%)"},
        ax=ax,
    )

    ax.set_xlabel(col_cat)
    ax.set_ylabel(row_cat)
    ax.set_title(f"{row_cat} × {col_cat} : churn heatmap")

    plt.tight_layout()
    if fig_name is not None:
        U.save_fig(fig, fig_name)
        print(f"Saved figure: {fig_name}.png")
    plt.show()
    plt.close(fig)


# --- 實際畫一個有 business 意義的組合 ---
plot_categorical_interaction_heatmap(
    df,
    row_cat="HandsetWebCapable",
    col_cat="PrizmCode",
    min_count=200,
    fig_name="F5g_hwcap_x_prizm_heatmap",
)


# %% [markdown]
# # 5.3.4 F5h：HandsetPrice_q × HandsetWebCapable churn heatmap

# %%
# ============================================
# 5.3.4 HandsetPrice_q × HandsetWebCapable 交互作用
# ============================================

import numpy as np

# -- 1) 建立 HandsetPrice_q（價格分組：Low / Mid / High / Unknown） --

hp = df["HandsetPrice"].copy()

# 若還沒轉成 numeric，先轉；無法轉的視為 NaN
hp_num = pd.to_numeric(hp, errors="coerce")

# 0 代表 Unknown，在這裡當作缺失，不參與 qcut
hp_num = hp_num.replace(0, np.nan)

# 只對非 NaN 做三等分（Low / Mid / High）
valid_mask = hp_num.notna()
labels = ["Low", "Mid", "High"]

hp_band = pd.Series(index=df.index, dtype="object")
hp_band.loc[valid_mask] = pd.qcut(
    hp_num[valid_mask],
    q=3,
    labels=labels
).astype(str)
hp_band.loc[~valid_mask] = "Unknown"

df["HandsetPrice_q"] = hp_band

print(df["HandsetPrice_q"].value_counts(dropna=False))

# -- 2) 計算 2D churn 表 --

tmp = (
    df.groupby(["HandsetPrice_q", "HandsetWebCapable"])[TARGET_BIN]
      .agg(churn_rate="mean", n="size")
      .reset_index()
)

table = tmp.pivot(index="HandsetPrice_q",
                  columns="HandsetWebCapable",
                  values="churn_rate") * 100
n_table = tmp.pivot(index="HandsetPrice_q",
                    columns="HandsetWebCapable",
                    values="n")

# 固定列順序比較好讀
row_order = ["Low", "Mid", "High", "Unknown"]
row_order = [r for r in row_order if r in table.index]
table = table.loc[row_order]
n_table = n_table.loc[row_order]

fig, ax = plt.subplots(figsize=(6, 4.5))

sns.heatmap(
    table,
    ax=ax,
    cmap="Blues",
    vmin=0,
    vmax=40,
    cbar_kws={"label": "Churn rate (%)"},
    annot=False,
)

# 在每格上面寫上 % 與 n
for i, row_lab in enumerate(table.index):
    for j, col_lab in enumerate(table.columns):
        val = table.loc[row_lab, col_lab]
        n = n_table.loc[row_lab, col_lab]
        ax.text(
            j + 0.5,
            i + 0.5,
            f"{val:.1f}%\n(n={n})",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )

ax.set_title("HandsetPrice_q × HandsetWebCapable : churn heatmap")
ax.set_xlabel("HandsetWebCapable")
ax.set_ylabel("HandsetPrice_q")

plt.tight_layout()
U.save_fig(fig, "F5h_handsetpriceq_x_webcapable")
print("Saved figure: F5h_handsetpriceq_x_webcapable.png")
plt.show()
plt.close(fig)


# %% [markdown]
# # 5.3.5 F5i：MonthsInService_bucket × RetentionOffersAccepted_flag churn heatmap

# %%
# ============================================
# 5.3.5 MonthsInService_bucket × RetentionOffersAccepted_flag 交互作用
# ============================================

# -- 1) 建立 tenure bucket & RetentionOffersAccepted_flag --

# Tenure 分桶：0–6, 6–12, 12–24, 24+（你可以之後調整 cutpoint）
bins = [0, 6, 12, 24, np.inf]
labels = ["0–6", "6–12", "12–24", "24+"]

df["MonthsInService_bucket"] = pd.cut(
    df["MonthsInService"],
    bins=bins,
    labels=labels,
    right=False
)

# RetentionOffersAccepted 是否有接受≥1個 offer
df["RetentionOffersAccepted_flag"] = np.where(
    df["RetentionOffersAccepted"] > 0,
    ">=1 accepted",
    "0 accepted"
)

print(df["MonthsInService_bucket"].value_counts(dropna=False))
print(df["RetentionOffersAccepted_flag"].value_counts(dropna=False))

# -- 2) 計算 2D churn 表 --

tmp2 = (
    df.groupby(["MonthsInService_bucket", "RetentionOffersAccepted_flag"])[TARGET_BIN]
      .agg(churn_rate="mean", n="size")
      .reset_index()
)

table2 = tmp2.pivot(index="MonthsInService_bucket",
                    columns="RetentionOffersAccepted_flag",
                    values="churn_rate") * 100
n_table2 = tmp2.pivot(index="MonthsInService_bucket",
                      columns="RetentionOffersAccepted_flag",
                      values="n")

fig, ax = plt.subplots(figsize=(6, 5))

sns.heatmap(
    table2,
    ax=ax,
    cmap="Blues",
    vmin=0,
    vmax=50,
    cbar_kws={"label": "Churn rate (%)"},
    annot=False,
)

for i, row_lab in enumerate(table2.index):
    for j, col_lab in enumerate(table2.columns):
        val = table2.loc[row_lab, col_lab]
        n = n_table2.loc[row_lab, col_lab]
        ax.text(
            j + 0.5,
            i + 0.5,
            f"{val:.1f}%\n(n={n})",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )

ax.set_title("MonthsInService_bucket × RetentionOffersAccepted_flag : churn heatmap")
ax.set_xlabel("RetentionOffersAccepted_flag")
ax.set_ylabel("MonthsInService_bucket")

plt.tight_layout()
U.save_fig(fig, "F5i_tenure_bucket_x_retentionoffer_flag")
print("Saved figure: F5i_tenure_bucket_x_retentionoffer_flag.png")
plt.show()
plt.close(fig)


# %% [markdown]
# # 5.4 ServiceArea head / tail & interaction

# %%
# ============================================
# 5.4 ServiceArea head / tail & interaction
#   - F5j: Top/Bottom ServiceArea churn 條圖
#   - T5b: head vs tail profile 表
#   - F5k: head vs tail profile 差異條圖
#   - F5l: ServiceArea × CustomerCareCalls 互動線圖
# ============================================

import numpy as np

# ---- 5.4.1 讀取 / 準備 ServiceArea churn 資訊 ----------------------------

# T4_high_card_categorical_churn 之前在第 4 章已經存過
try:
    high_card_table = high_card_table.copy()
except NameError:
    high_card_table = U.load_df("T4_high_card_categorical_churn")

sa_tbl = high_card_table.query("feature == 'ServiceArea'").copy()

if sa_tbl.empty:
    print("⚠️ No ServiceArea rows found in T4_high_card_categorical_churn, skip 5.4.")
else:
    # 依 churn_rate 排序
    sa_tbl = sa_tbl.sort_values("churn_rate", ascending=True).reset_index(drop=True)

    top_n = 5
    sa_top = sa_tbl.sort_values("churn_rate", ascending=False).head(top_n).copy()
    sa_bot = sa_tbl.sort_values("churn_rate", ascending=True).head(top_n).copy()

    sa_top["risk_group"] = "High-risk (top 5)"
    sa_bot["risk_group"] = "Low-risk (bottom 5)"

    sa_head_tail = pd.concat([sa_top, sa_bot], ignore_index=True)

    print("[ServiceArea top 5 by churn rate]")
    display(sa_top)
    print("[ServiceArea bottom 5 by churn rate]")
    display(sa_bot)

    # 存成一張小表（方便在論文中引用）
    U.save_df(sa_head_tail, "T5a_servicearea_head_tail_churn")
    print("Saved table: T5a_servicearea_head_tail_churn.csv")

    # ---- 5.4.1 F5j：Top/Bottom ServiceArea churn bar --------------------

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(sa_head_tail) + 2))

    sns.barplot(
        data=sa_head_tail,
        x="churn_rate_pct",
        y="category",
        hue="risk_group",
        ax=ax,
    )

    ax.axvline(GLOBAL_CHURN_RATE * 100,
               linestyle="--", color="gray", linewidth=1)

    ax.set_xlabel("Churn rate (%)")
    ax.set_ylabel("ServiceArea")
    ax.set_title("Top / Bottom ServiceAreas by churn rate")

    # 標上百分比與樣本數
    for i, row in sa_head_tail.iterrows():
        ax.text(
            row["churn_rate_pct"] + 0.4,
            i,
            f"{row['churn_rate_pct']:.1f}%\n(n={row['n']})",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    U.save_fig(fig, "F5j_servicearea_head_tail_churn")
    print("Saved figure: F5j_servicearea_head_tail_churn.png")
    plt.show()
    plt.close(fig)


# %% [markdown]
# # ---- 5.4.2 T5b + F5k：Head vs Tail profile -------------------------

# %%
    # ---- 5.4.2 T5b + F5k：Head vs Tail profile -------------------------

    # 用 top5 當作 High-risk area
    high_risk_areas = sa_top["category"].tolist()

    df["HighRiskServiceArea"] = np.where(
        df["ServiceArea"].isin(high_risk_areas),
        "High-risk",
        "Others",
    )

    # 想比較的數值特徵（存在才算）
    profile_features = [
        "MonthlyRevenue",
        "MonthlyMinutes",
        "RevenuePerMinute",      # 前面 5.2 已建立
        "CustomerCareCalls",
        "DroppedCalls",
        "DroppedBlockedCalls",
        "MonthsInService",
        "CurrentEquipmentDays",
        "IncomeGroup",
        "HandsetPrice",
        "CreditRating",
    ]
    profile_features = [c for c in profile_features if c in df.columns]

    rows = []
    for col in profile_features:
        hi = df.loc[df["HighRiskServiceArea"] == "High-risk", col].mean()
        lo = df.loc[df["HighRiskServiceArea"] == "Others", col].mean()
        rows.append(
            {
                "feature": col,
                "mean_high": hi,
                "mean_others": lo,
                "diff_high_minus_others": hi - lo,
            }
        )

    profile_df = pd.DataFrame(rows)
    profile_df = profile_df.sort_values(
        "diff_high_minus_others", ascending=False
    ).reset_index(drop=True)

    U.save_df(profile_df, "T5b_servicearea_head_tail_profile")
    print("Saved table: T5b_servicearea_head_tail_profile.csv")
    display(profile_df)

    # F5k：差異條圖（High-risk mean – Others mean）
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(profile_df) + 2))

    sns.barplot(
        data=profile_df,
        x="diff_high_minus_others",
        y="feature",
        ax=ax,
    )

    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Difference in mean (High-risk − Others)")
    ax.set_ylabel("Feature")
    ax.set_title("ServiceArea head vs tail profile differences")

    plt.tight_layout()
    U.save_fig(fig, "F5k_servicearea_head_tail_profile_diff")
    print("Saved figure: F5k_servicearea_head_tail_profile_diff.png")
    plt.show()
    plt.close(fig)



# %% [markdown]
# # ---- 5.4.3 F5l：ServiceArea × CustomerCareCalls 互動線圖 ----------

# %%
    # ---- 5.4.3 F5l：ServiceArea × CustomerCareCalls 互動線圖 ----------

    # CustomerCareCalls 分段：0 / 1–2 / 3+
    def _ccalls_bin(x: float) -> str:
        if pd.isna(x) or x <= 0:
            return "0"
        elif x <= 2:
            return "1–2"
        else:
            return "3+"

    df["CustomerCareCalls_band"] = df["CustomerCareCalls"].apply(_ccalls_bin)

    band_order = ["0", "1–2", "3+"]

    tmp = (
        df.groupby(["CustomerCareCalls_band", "HighRiskServiceArea"])[TARGET_BIN]
        .agg(churn_rate="mean", n="size")
        .reset_index()
    )
    tmp["churn_rate_pct"] = tmp["churn_rate"] * 100

    print("[ServiceArea × CustomerCareCalls_band churn table]")
    display(tmp)

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.lineplot(
        data=tmp,
        x="CustomerCareCalls_band",
        y="churn_rate_pct",
        hue="HighRiskServiceArea",
        marker="o",
        sort=False,
        ax=ax,
    )

    ax.set_xlabel("CustomerCareCalls band")
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("CustomerCareCalls × ServiceArea risk (High-risk vs Others)")
    ax.axhline(GLOBAL_CHURN_RATE * 100, linestyle="--", color="gray", linewidth=1)

    # 保證 x 軸順序
    ax.set_xticks(range(len(band_order)))
    ax.set_xticklabels(band_order)

    # 每個點加上 n
    for _, row in tmp.iterrows():
        ax.text(
            band_order.index(row["CustomerCareCalls_band"]),
            row["churn_rate_pct"] + 0.3,
            f"{row['churn_rate_pct']:.1f}%\n(n={row['n']})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    U.save_fig(fig, "F5l_ccalls_band_x_highriskarea")
    print("Saved figure: F5l_ccalls_band_x_highriskarea.png")
    plt.show()
    plt.close(fig)


# %% [markdown]
# # 附錄用互動圖

# %% [markdown]
# # ServiceArea Head/Tail × CustomerCareCalls_flag

# %%
# ============================================
# A5-1 ServiceArea Head/Tail × CustomerCareCalls_flag
# ============================================

import numpy as np

# --- 1) 建立 CustomerCareCalls_flag ---
df["CustomerCareCalls_flag"] = np.where(
    df["CustomerCareCalls"] > 0,
    "> 0 calls",
    "0 calls"
)

# --- 2) 找出 ServiceArea 的 head / tail（依 churn 率） ---
sa_stats = (
    df.groupby("ServiceArea")[TARGET_BIN]
      .agg(churn_rate="mean", n="size")
      .reset_index()
)

# 可以視情況加一個樣本數門檻，例如 n >= 200
sa_stats = sa_stats[sa_stats["n"] >= 200]

sa_stats_sorted = sa_stats.sort_values("churn_rate", ascending=False)

top_k = 5
top_areas = sa_stats_sorted.head(top_k)["ServiceArea"].tolist()
bottom_areas = sa_stats_sorted.tail(top_k)["ServiceArea"].tolist()

print("Top churn ServiceAreas:", top_areas)
print("Bottom churn ServiceAreas:", bottom_areas)

def map_sa_bucket(sa: str) -> str:
    if sa in top_areas:
        return "Top-5 high churn"
    elif sa in bottom_areas:
        return "Bottom-5 low churn"
    else:
        return "Others"

df["ServiceArea_headtail"] = df["ServiceArea"].map(map_sa_bucket)

print(df["ServiceArea_headtail"].value_counts())

# --- 3) 2D churn 熱圖 ---

tmp = (
    df.groupby(["ServiceArea_headtail", "CustomerCareCalls_flag"])[TARGET_BIN]
      .agg(churn_rate="mean", n="size")
      .reset_index()
)

table = tmp.pivot(index="ServiceArea_headtail",
                  columns="CustomerCareCalls_flag",
                  values="churn_rate") * 100
n_table = tmp.pivot(index="ServiceArea_headtail",
                    columns="CustomerCareCalls_flag",
                    values="n")

# 固定列順序
row_order = ["Top-5 high churn", "Bottom-5 low churn", "Others"]
row_order = [r for r in row_order if r in table.index]
table = table.loc[row_order]
n_table = n_table.loc[row_order]

fig, ax = plt.subplots(figsize=(6, 4.5))

sns.heatmap(
    table,
    ax=ax,
    cmap="Blues",
    vmin=0,
    vmax=50,
    cbar_kws={"label": "Churn rate (%)"},
    annot=False,
)

for i, row_lab in enumerate(table.index):
    for j, col_lab in enumerate(table.columns):
        val = table.loc[row_lab, col_lab]
        n = n_table.loc[row_lab, col_lab]
        ax.text(
            j + 0.5,
            i + 0.5,
            f"{val:.1f}%\n(n={n})",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )

ax.set_title("ServiceArea head/tail × CustomerCareCalls_flag : churn heatmap")
ax.set_xlabel("CustomerCareCalls_flag")
ax.set_ylabel("ServiceArea head/tail group")

plt.tight_layout()
U.save_fig(fig, "A5_servicearea_headtail_x_custcare_flag")
print("Saved figure: A5_servicearea_headtail_x_custcare_flag.png")
plt.show()
plt.close(fig)


# %% [markdown]
# # DroppedQuality_flag × HighUsage_flag（品質 × 使用量）

# %%
# ============================================
# A5-2 DroppedQuality_flag × HighUsage_flag
#  - 品質 × 使用量的交互作用
# ============================================

import numpy as np

# --- 1) 建立 DroppedQuality_flag & HighUsage_flag ---

quality_sum = (
    df["DroppedCalls"].fillna(0) +
    df["BlockedCalls"].fillna(0) +
    df["UnansweredCalls"].fillna(0) +
    df["DroppedBlockedCalls"].fillna(0)
)

df["DroppedQuality_flag"] = np.where(
    quality_sum > 0,
    "Any quality issue",
    "No issue"
)

# 使用量高低：以上四分位數作為 high usage
usage_q3 = df["MonthlyMinutes"].quantile(0.75)
print("MonthlyMinutes 75th percentile:", usage_q3)

df["HighUsage_flag"] = np.where(
    df["MonthlyMinutes"] >= usage_q3,
    "High usage",
    "Normal/low"
)

print(df["DroppedQuality_flag"].value_counts())
print(df["HighUsage_flag"].value_counts())

# --- 2) 2D churn 熱圖 ---

tmp = (
    df.groupby(["DroppedQuality_flag", "HighUsage_flag"])[TARGET_BIN]
      .agg(churn_rate="mean", n="size")
      .reset_index()
)

table = tmp.pivot(index="DroppedQuality_flag",
                  columns="HighUsage_flag",
                  values="churn_rate") * 100
n_table = tmp.pivot(index="DroppedQuality_flag",
                    columns="HighUsage_flag",
                    values="n")

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(
    table,
    ax=ax,
    cmap="Blues",
    vmin=0,
    vmax=50,
    cbar_kws={"label": "Churn rate (%)"},
    annot=False,
)

for i, row_lab in enumerate(table.index):
    for j, col_lab in enumerate(table.columns):
        val = table.loc[row_lab, col_lab]
        n = n_table.loc[row_lab, col_lab]
        ax.text(
            j + 0.5,
            i + 0.5,
            f"{val:.1f}%\n(n={n})",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )

ax.set_title("DroppedQuality_flag × HighUsage_flag : churn heatmap")
ax.set_xlabel("HighUsage_flag")
ax.set_ylabel("DroppedQuality_flag")

plt.tight_layout()
U.save_fig(fig, "A5_droppedquality_x_highusage_flag")
print("Saved figure: A5_droppedquality_x_highusage_flag.png")
plt.show()
plt.close(fig)


# %% [markdown]
# # 建立 SilentChurn_flag（定義)

# %%
import numpy as np
import pandas as pd

# 1. Build SilentChurn_flag
df["SilentChurn_flag"] = np.where(  (df["Churn01"] == 1) &
    (df["HighUsage_flag"] == 0) & (df["DroppedQuality_flag"] == 0),
    1,
    0
)

# 2. Aggregate sample size and churn rate
silent_counts = (
    df.groupby("SilentChurn_flag")["Churn01"]
      .agg(n_customers="size", churn_rate="mean")
      .reset_index()
)

# 3. Save table via utils helper (this should write to
#    /Users/audreychang/Projects/Capstone-Seminar-Paper/artifacts/tables)
U.save_df(silent_counts, "T5x_silentchurn_flag_summary")
print("Saved table: T5x_silentchurn_flag_summary.csv")

# 4. Show the table in the notebook
silent_counts


# %%


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# 計算 churn 率與樣本數
silent_summary = (
    df.groupby("SilentChurn_flag")["Churn01"]
      .agg(churn_rate="mean", n_customers="size")
      .reset_index()
)

# 把 0 / 1 換成比較好看的標籤
silent_summary["group"] = silent_summary["SilentChurn_flag"].map({
    0: "Non-silent (others)",
    1: "Silent churners\n(low usage & no issue)"
})

fig, ax = plt.subplots(figsize=(6, 5))
sns.barplot(
    data=silent_summary,
    x="group",
    y="churn_rate",
    ax=ax
)

ax.set_ylabel("Churn rate")
ax.set_xlabel("")
ax.set_title("Churn rate by SilentChurn_flag")

# 在柱子上標上百分比與樣本數
for p, (_, row) in zip(ax.patches, silent_summary.iterrows()):
    height = p.get_height()
    ax.annotate(
        f"{height*100:.1f}%\n(n={row['n_customers']:,})",
        (p.get_x() + p.get_width() / 2., height),
        ha="center",
        va="bottom",
        fontsize=10
    )

fig.tight_layout()

# 儲存圖檔
U.save_fig(fig, "A5_silentchurn_flag_churn.png")
print("Saved figure: A5_silentchurn_flag_churn.png")

plt.show()


# %%


# %%
# 年齡分組（你可以依自己的偏好調整 cut 點）
df["Age_band"] = pd.cut(
    df["AgeHH1"],
    bins=[0, 30, 45, 60, 120],
    labels=["<30", "30–44", "45–59", "60+"]
)

# 資歷分組（MonthsInService）
df["Tenure_band"] = pd.cut(
    df["MonthsInService"],
    bins=[0, 6, 12, 24, 48, df["MonthsInService"].max()],
    labels=["<6m", "6–11m", "12–23m", "24–47m", "48m+"]
)
def plot_stratified_silent_churn(df, group_col, fig_name, figsize=(8, 5)):
    temp = (
        df.dropna(subset=[group_col])
          .groupby([group_col, "SilentChurn_flag"])["Churn01"]
          .mean()
          .reset_index()
    )

    temp["SilentChurn_flag"] = temp["SilentChurn_flag"].map({
        0: "Non-silent",
        1: "Silent"
    })

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=temp,
        x=group_col,
        y="Churn01",
        hue="SilentChurn_flag",
        ax=ax
    )

    ax.set_ylabel("Churn rate")
    ax.set_xlabel(group_col)
    ax.set_title(f"Churn by SilentChurn_flag within {group_col}")

    # 百分比標籤
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height*100:.1f}%",
            (p.get_x() + p.get_width() / 2., height),
            ha="center",
            va="bottom",
            fontsize=8
        )

    ax.legend(title="")
    fig.tight_layout()

    # 儲存圖檔
    U.save_fig(fig, fig_name)
    print(f"Saved figure: {fig_name}")

    plt.show()

# 分層檢查：年齡 & 資歷
plot_stratified_silent_churn(df, "Age_band", "A5_silentchurn_x_age_band.png")
plot_stratified_silent_churn(df, "Tenure_band", "A5_silentchurn_x_tenure_band.png")


# %%


# %%
# Define which revenue columns to summarise
cols = {
    "MonthlyRevenue": "MonthlyRevenue",
    "RevenuePerMinute": "RevenuePerMinute"   # change name here if needed
}

silent_revenue_summary = (
    df.groupby("SilentChurn_flag")
      .agg(
          churn_rate=("Churn01", "mean"),
          n_customers=("Churn01", "size"),
          **{k: (v, "mean") for k, v in cols.items()}
      )
      .reset_index()
)

# Save using utils helper → goes to artifacts/tables
U.save_df(silent_revenue_summary, "T5x_silentchurn_revenue_summary")
print("Saved table: T5x_silentchurn_revenue_summary.csv")

# Show the table in the notebook
silent_revenue_summary



# %%


# %%
# MonthlyRevenue boxplot
fig, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(
    data=df,
    x="SilentChurn_flag",
    y="MonthlyRevenue",
    ax=ax
)
ax.set_xlabel("SilentChurn_flag (0 = others, 1 = silent)")
ax.set_title("MonthlyRevenue distribution by SilentChurn_flag")

fig.tight_layout()
U.save_fig(fig, "A5_silentchurn_monthlyrevenue_box.png")
print("Saved figure: A5_silentchurn_monthlyrevenue_box.png")
plt.show()

# RevenuePerMinute boxplot（如果有 RPM 欄位）
fig, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(
    data=df,
    x="SilentChurn_flag",
    y="RevenuePerMinute",  # 如果欄名不同，改這裡
    ax=ax
)
ax.set_xlabel("SilentChurn_flag (0 = others, 1 = silent)")
ax.set_title("RevenuePerMinute distribution by SilentChurn_flag")

fig.tight_layout()
U.save_fig(fig, "A5_silentchurn_rpm_box.png")
print("Saved figure: A5_silentchurn_rpm_box.png")
plt.show()


# %%


# %% [markdown]
# # InactiveSubs deciles × IncomeBand

# %%
# ============================================
# A5-3 InactiveSubs deciles × IncomeBand
# ============================================

import numpy as np

# --- 1) 建立 InactiveSubs ---
df["InactiveSubs"] = (df["UniqueSubs"] - df["ActiveSubs"]).clip(lower=0)
print(df["InactiveSubs"].describe())

# --- 2) IncomeBand（如果前面已經建過，就不會改變） ---
def make_income_band(x):
    """
    IncomeGroup:
      0 = Unknown
      1–3 = Low
      4–5 = Mid
      6–7 = High
    """
    if pd.isna(x):
        return "Unknown"
    try:
        x = int(x)
    except ValueError:
        return "Unknown"

    if x == 0:
        return "Unknown"
    elif 1 <= x <= 3:
        return "Low"
    elif 4 <= x <= 5:
        return "Mid"
    else:
        return "High"

df["IncomeBand"] = df["IncomeGroup"].apply(make_income_band)
print(df["IncomeBand"].value_counts(dropna=False))

# --- 3) 建立 InactiveSubs deciles（用 rank 避免同值太多的問題） ---

mask_valid = df["InactiveSubs"].notna()
ranks = df.loc[mask_valid, "InactiveSubs"].rank(method="first")

df.loc[mask_valid, "InactiveSubs_decile"] = pd.qcut(
    ranks,
    q=10,
    labels=[str(i) for i in range(1, 11)]
)

print(df["InactiveSubs_decile"].value_counts(dropna=False))

# --- 4) 算每個 decile × IncomeBand 的 churn 率 ---

tmp = (
    df.dropna(subset=["InactiveSubs_decile", "IncomeBand"])
      .groupby(["InactiveSubs_decile", "IncomeBand"])[TARGET_BIN]
      .agg(churn_rate="mean", n="size")
      .reset_index()
)

pivot_tbl = tmp.pivot(index="InactiveSubs_decile",
                      columns="IncomeBand",
                      values="churn_rate")

fig, ax = plt.subplots(figsize=(8, 4.5))

for band in ["Low", "Mid", "High"]:
    if band in pivot_tbl.columns:
        ax.plot(
            pivot_tbl.index.astype(int),
            pivot_tbl[band],
            marker="o",
            label=band,
        )

ax.axhline(GLOBAL_CHURN_RATE, color="gray", linestyle="--", linewidth=1)

ax.set_xlabel("InactiveSubs decile (Low → High)")
ax.set_ylabel("Churn rate")
ax.set_title("InactiveSubs decile risk by IncomeBand")
ax.set_xticks(range(1, 11))
ax.legend(title="IncomeBand", loc="upper left")

plt.tight_layout()
U.save_fig(fig, "A5_inactive_deciles_by_incomeband")
print("Saved figure: A5_inactive_deciles_by_incomeband.png")
plt.show()
plt.close(fig)


# %% [markdown]
# # 5.5 Leakage Scan

# %%
# ============================================
# 5.5 Leakage Scan：Retention 系列欄位的 0/1 風險比較
#   F5j_leakage_flags_churn.png
# ============================================

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 5.5.1 建立 Retention 相關的 flag 欄位
# --------------------------------------------

leak_flag_name_map = {}

# 1) RetentionCalls_flag: >0 視為有留客紀錄
if "RetentionCalls" in df.columns:
    df["RetentionCalls_flag"] = (df["RetentionCalls"].fillna(0) > 0).astype(int)
    leak_flag_name_map["RetentionCalls_flag"] = "RetentionCalls > 0"

# 2) RetentionOffersAccepted_flag: >0 視為有接受挽留
if "RetentionOffersAccepted" in df.columns:
    df["RetentionOffersAccepted_flag"] = (df["RetentionOffersAccepted"].fillna(0) > 0).astype(int)
    leak_flag_name_map["RetentionOffersAccepted_flag"] = "RetentionOffersAccepted > 0"

# 3) MadeCallToRetentionTeam：原始是 Yes/No，要轉成 0/1
if "MadeCallToRetentionTeam" in df.columns:
    col = df["MadeCallToRetentionTeam"]

    if pd.api.types.is_numeric_dtype(col):
        # 如果你之後前處理已經變成 0/1 了，就直接確保是 int
        df["MadeCallToRetentionTeam_flag"] = col.fillna(0).astype(int)
    else:
        # 一般情況：Yes / No / '  Yes ' 之類
        ser = col.fillna("No").astype(str).str.strip()
        df["MadeCallToRetentionTeam_flag"] = ser.isin(["Yes", "Y", "1", "True"]).astype(int)

    leak_flag_name_map["MadeCallToRetentionTeam_flag"] = "MadeCallToRetentionTeam = 1"

# 最終拿這些 flag 來畫圖
flag_cols = leak_flag_name_map
print("[Leakage flags to plot]:", flag_cols)


# 5.5.2 共用繪圖 helper：單一 flag 的 0/1 流失率比較
# --------------------------------------------

def plot_leakage_flag_risk(df: pd.DataFrame,
                           flag_col: str,
                           nice_name: str,
                           target_col: str = TARGET_BIN,
                           ax=None):
    """
    對一個 0/1 flag 欄位畫兩根 bar：
      x: flag = 0 / 1
      y: 該群體的 churn rate (%)
    並標註人數與百分比，外加全體 baseline 虛線。
    """
    if ax is None:
        ax = plt.gca()

    # groupby 計算樣本數與流失率
    tmp = (
        df.groupby(flag_col)[target_col]
        .agg(n="size", churn_rate="mean")
        .reset_index()
        .sort_values(flag_col)
    )
    # 確保 flag 是 0/1，轉成字串方便標籤
    tmp[flag_col] = tmp[flag_col].astype(int).astype(str)
    tmp["churn_rate_pct"] = tmp["churn_rate"] * 100

    # 畫 bar chart：兩根柱子 0 / 1
    sns.barplot(
        data=tmp,
        x=flag_col,
        y="churn_rate_pct",
        ax=ax,
    )

    # 全體平均流失率 baseline
    ax.axhline(
        GLOBAL_CHURN_RATE * 100,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Overall churn rate"
    )

    ax.set_xlabel(f"{nice_name} flag (0 / 1)")
    ax.set_ylabel("Churn rate (%)")
    ax.set_title(f"{nice_name}: churn rate by flag (0 vs 1)")

    # 設定 y 範圍 & 在每根柱子上標註百分比與樣本數
    ymax = max(tmp["churn_rate_pct"].max(), GLOBAL_CHURN_RATE * 100) * 1.25

    for i, row in tmp.iterrows():
        ax.text(
            i,                              # x 位置：第 i 根柱子
            row["churn_rate_pct"] + ymax*0.03,
            f"{row['churn_rate_pct']:.1f}%\n(n={row['n']})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylim(0, ymax)
    ax.legend(loc="upper left", fontsize=8)


# 5.5.3 一次畫出所有 leakage flags 的 2-bar 圖
# --------------------------------------------

n_flags = len(flag_cols)
if n_flags == 0:
    print("⚠️ No leakage flags found to plot; skip F5j.")
else:
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_flags,
        figsize=(5 * n_flags, 4.5),
        squeeze=False,
        constrained_layout=True,
    )

    for j, (col, nice_name) in enumerate(flag_cols.items()):
        ax = axes[0, j]
        plot_leakage_flag_risk(
            df,
            flag_col=col,
            nice_name=nice_name,
            target_col=TARGET_BIN,
            ax=ax,
        )

    U.save_fig(fig, "F5j_leakage_flags_churn")
    print("Saved figure: F5j_leakage_flags_churn.png")
    plt.show()
    plt.close(fig)


# %%



