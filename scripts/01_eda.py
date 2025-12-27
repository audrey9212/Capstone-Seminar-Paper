#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_eda.py: Data Health & Exploratory Data Analysis (Production Version)

Outputs:
  - T0_feature_registry.csv
  - T0_semantic_groups_table.csv
  - T2_churn_distribution.csv + F2a_churn_distribution_donut.png
  - T1_sparsity_overview.csv + F1a_sparsity_missing_zero.png
  - T1_value_domain_checks.csv
  - T1_outliers_iqr3.csv + (F1d in appendix)
  - T4_ordinal_trends.csv + F4a_ordinal_trends.png
  - T4_cramers_v_vs_churn.csv + F4f_cramers_v_top_categorical.png
  - T9_leakage_scan.csv + F5j_leakage_flags_churn.png
"""
import os
import sys
from pathlib import Path
import warnings

# --- Project root / imports (must come before importing utils) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # scripts/.. -> project root
os.environ["CAPSTONE_ROOT"] = str(PROJECT_ROOT)     # force utils to use this repo as root
sys.path.insert(0, str(PROJECT_ROOT))               # allow: import utils
# If you need src modules later, you can either:
#   from src import optuna_utils
# (No need to add PROJECT_ROOT/"src" to sys.path if src is a package.)

import utils as U

# --- Third-party libs ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Config ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


# =============================================================================
# 1. Load raw data
# =============================================================================
def load_data():
    """Load Cell2Cell training data"""
    raw_file = U.raw_path("cell2celltrain.csv")
    print(f"Loading data from {raw_file}...")
    
    df = pd.read_csv(raw_file, low_memory=False)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    
    return df


# =============================================================================
# 2. Create binary target Churn01
# =============================================================================
def create_churn01(df):
    """Convert Churn column to binary 0/1"""
    df = df.copy()
    
    # Map common Yes/No variants to 1/0
    churn_col = df["Churn"].astype(str).str.lower().str.strip()
    df["Churn01"] = churn_col.map({
        "yes": 1, "y": 1, "true": 1, "1": 1, "1.0": 1,
        "no": 0, "n": 0, "false": 0, "0": 0, "0.0": 0
    })
    
    # Handle missing
    df["Churn01"] = df["Churn01"].fillna(-1).astype(int)
    df = df[df["Churn01"] != -1]
    
    print(f"Churn01 value counts:\n{df['Churn01'].value_counts()}")
    
    return df


# =============================================================================
# 3. Feature registry & semantic groups (T0_feature_registry, T0_semantic_groups_table)
# =============================================================================
def create_feature_registry(df):
    """Build feature metadata table"""
    features = []
    
    for col in df.columns:
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        n_missing = df[col].isna().sum()
        pct_missing = 100 * n_missing / len(df)
        
        # Infer type
        if col == "Churn01":
            feat_type = "target"
        elif col == "CustomerID":
            feat_type = "identifier"
        elif dtype in ["int64", "float64"]:
            # Ordinal if <20 unique values
            feat_type = "ordinal" if (n_unique <= 20 and col not in ["AdjustmentMonth", "AdjustmentYear"]) else "continuous"
        else:
            feat_type = "categorical"
        
        features.append({
            "feature": col,
            "dtype": str(dtype),
            "n_unique": n_unique,
            "n_missing": n_missing,
            "pct_missing": round(pct_missing, 2),
            "type": feat_type
        })
    
    reg = pd.DataFrame(features)
    U.save_df(reg, "T0_feature_registry")
    
    # Summary by type
    summary = reg.groupby("type").size().reset_index(name="count")
    print(f"\nFeature type summary:\n{summary}")
    
    return reg


def create_semantic_groups(reg):
    """Create semantic grouping of features"""
    semantic_map = {
        "Demographic": ["Age", "Gender", "Marital", "Homeownership"],
        "Service": ["Service", "AreaCode"],
        "Billing": ["CreditCard", "PaperlessBill", "Phone"],
        "Account": ["AccountWeeks", "CustServCalls"],
        "Usage": ["DayMins", "DayCharge", "EveCharge", "NightCharge", "IntlCharge"],
        "Customer Value": ["CreditRating", "IncomeGroup", "HandsetPrice"]
    }
    
    # Flatten
    sem_table = []
    for group, feats in semantic_map.items():
        for feat in feats:
            if feat in reg["feature"].values:
                feat_info = reg[reg["feature"] == feat].iloc[0]
                sem_table.append({
                    "semantic_group": group,
                    "feature": feat,
                    "type": feat_info["type"],
                    "n_unique": feat_info["n_unique"]
                })
    
    sem_df = pd.DataFrame(sem_table)
    U.save_df(sem_df, "T0_semantic_groups_table")
    print(f"\nSemantic groups:\n{sem_df.groupby('semantic_group').size()}")
    
    return sem_df


# =============================================================================
# 4. Target distribution (T2_churn_distribution + F2a)
# =============================================================================
def analyze_target(df):
    """Analyze Churn distribution"""
    churn_counts = df["Churn01"].value_counts().sort_index()
    pct = 100 * churn_counts / len(df)
    
    summary = pd.DataFrame({
        "churn_value": churn_counts.index,
        "count": churn_counts.values,
        "percentage": pct.values
    })
    
    U.save_df(summary, "T2_churn_distribution")
    print(f"\nChurn distribution:\n{summary}")
    
    # Plot: donut chart
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2ecc71", "#e74c3c"]
    wedges, texts, autotexts = ax.pie(
        churn_counts.values,
        labels=["No Churn", "Churn"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 12, "weight": "bold"}
    )
    
    # Draw circle for donut
    centre_circle = plt.Circle((0, 0), 0.70, fc="white")
    ax.add_artist(centre_circle)
    ax.set_title("Churn Distribution (Class Imbalance)", fontsize=14, weight="bold", pad=20)
    
    plt.tight_layout()
    U.save_fig(fig, "F2a_churn_distribution_donut")
    plt.close()


# =============================================================================
# 5. Sparsity & Missing (T1_sparsity_overview + F1a)
# =============================================================================
def analyze_sparsity(df):
    """Analyze missing values and zero inflation"""
    sparsity = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in ["Churn01", "AdjustmentMonth", "AdjustmentYear"]:
            continue
        
        n_missing = df[col].isna().sum()
        n_zero = (df[col] == 0).sum()
        n_neg = (df[col] < 0).sum()
        
        sparsity.append({
            "feature": col,
            "n_missing": n_missing,
            "pct_missing": round(100 * n_missing / len(df), 2),
            "n_zero": n_zero,
            "pct_zero": round(100 * n_zero / len(df), 2),
            "n_negative": n_neg,
            "pct_negative": round(100 * n_neg / len(df), 2)
        })
    
    sparse_df = pd.DataFrame(sparsity).sort_values("pct_missing", ascending=False)
    U.save_df(sparse_df, "T1_sparsity_overview")
    print(f"\nTop 10 sparse features:\n{sparse_df.head(10)}")
    
    # Plot: sparsity vs zero inflation
    fig, ax = plt.subplots(figsize=(12, 6))
    
    plot_df = sparse_df[sparse_df["pct_missing"] + sparse_df["pct_zero"] > 0].copy()
    plot_df = plot_df.sort_values("pct_missing", ascending=True).tail(15)
    
    x = np.arange(len(plot_df))
    width = 0.35
    
    ax.barh(x - width/2, plot_df["pct_missing"], width, label="Missing", color="#3498db")
    ax.barh(x + width/2, plot_df["pct_zero"], width, label="Zero", color="#e74c3c")
    
    ax.set_yticks(x)
    ax.set_yticklabels(plot_df["feature"])
    ax.set_xlabel("Percentage (%)", fontsize=11)
    ax.set_title("Data Sparsity: Missing & Zero Inflation (Top 15)", fontsize=12, weight="bold")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    U.save_fig(fig, "F1a_sparsity_missing_zero")
    plt.close()


# =============================================================================
# 6. Value domain & logic checks (T1_value_domain_checks)
# =============================================================================
def check_value_domain(df):
    """Check value ranges and logic consistency"""
    checks = []
    
    # Define expected ranges
    domain_rules = {
        "Age": {"min": 18, "max": 100},
        "AccountWeeks": {"min": 0, "max": 1000},
        "DayMins": {"min": 0, "max": 1440},
        "DayCharge": {"min": 0, "max": 200},
        "CustServCalls": {"min": 0, "max": 20},
    }
    
    for col, rule in domain_rules.items():
        if col not in df.columns:
            continue
        
        out_of_range = ((df[col] < rule["min"]) | (df[col] > rule["max"])).sum()
        checks.append({
            "feature": col,
            "expected_range": f"[{rule['min']}, {rule['max']}]",
            "n_out_of_range": out_of_range,
            "pct_out_of_range": round(100 * out_of_range / len(df), 2)
        })
    
    check_df = pd.DataFrame(checks)
    U.save_df(check_df, "T1_value_domain_checks")
    print(f"\nValue domain check:\n{check_df}")


# =============================================================================
# 7. Outliers (T1_outliers_iqr3 + optional F1d)
# =============================================================================
def detect_outliers(df):
    """Detect extreme values using IQR × 3"""
    outliers = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in ["Churn01", "AdjustmentMonth", "AdjustmentYear"]:
            continue
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        if n_outliers > 0:
            outliers.append({
                "feature": col,
                "Q1": round(Q1, 2),
                "Q3": round(Q3, 2),
                "IQR": round(IQR, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "n_outliers": n_outliers,
                "pct_outliers": round(100 * n_outliers / len(df), 2)
            })
    
    outlier_df = pd.DataFrame(outliers).sort_values("pct_outliers", ascending=False)
    U.save_df(outlier_df, "T1_outliers_iqr3")
    print(f"\nOutliers (IQR×3) summary:\n{outlier_df.head(10)}")


# =============================================================================
# 8. Ordinal feature trends (T4_ordinal_trends + F4a)
# =============================================================================
def analyze_ordinal(df):
    """Analyze ordinal features vs churn"""
    ordinal_features = ["CreditRating", "IncomeGroup", "HandsetPrice"]
    
    ordinal_trends = []
    
    for feat in ordinal_features:
        if feat not in df.columns:
            continue
        
        # Create churn rate by ordinal value
        grouped = df.groupby(feat)["Churn01"].agg(["sum", "count", "mean"]).reset_index()
        grouped.columns = [feat, "churn_count", "n_total", "churn_rate"]
        
        for _, row in grouped.iterrows():
            ordinal_trends.append({
                "feature": feat,
                "value": row[feat],
                "churn_count": int(row["churn_count"]),
                "n_total": int(row["n_total"]),
                "churn_rate": round(row["churn_rate"], 4)
            })
    
    trend_df = pd.DataFrame(ordinal_trends)
    U.save_df(trend_df, "T4_ordinal_trends")
    
    # Plot: Focus on CreditRating (key example in thesis)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, feat in enumerate(ordinal_features):
        if feat not in df.columns:
            continue
        
        feat_data = trend_df[trend_df["feature"] == feat].sort_values("churn_rate")
        
        axes[idx].bar(range(len(feat_data)), feat_data["churn_rate"], color="#3498db")
        axes[idx].set_xticks(range(len(feat_data)))
        axes[idx].set_xticklabels(feat_data["value"], rotation=45, ha="right")
        axes[idx].set_ylabel("Churn Rate", fontsize=10)
        axes[idx].set_title(f"{feat} Trend", fontsize=11, weight="bold")
        axes[idx].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    U.save_fig(fig, "F4a_ordinal_trends")
    plt.close()
    
    print(f"\nOrdinal feature trends:\n{trend_df}")


# =============================================================================
# 9. Categorical association: Cramér's V (T4_cramers_v_vs_churn + F4f)
# =============================================================================
def cramers_v(x, y):
    """Calculate Cramér's V statistic"""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    
    if min_dim == 0:
        return 0.0
    
    return np.sqrt(chi2 / (n * min_dim))


def analyze_categorical_association(df):
    """Analyze categorical features vs churn using Cramér's V"""
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    if "Churn" in cat_cols:
        cat_cols.remove("Churn")
    
    assoc_results = []
    
    for col in cat_cols:
        v = cramers_v(df[col].fillna("Missing"), df["Churn01"])
        assoc_results.append({
            "feature": col,
            "cramers_v": round(v, 4)
        })
    
    assoc_df = pd.DataFrame(assoc_results).sort_values("cramers_v", ascending=False)
    U.save_df(assoc_df, "T4_cramers_v_vs_churn")
    print(f"\nCramér's V (categorical association):\n{assoc_df}")
    
    # Plot: Top categorical features
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top_assoc = assoc_df.head(10)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.8, len(top_assoc)))
    
    ax.barh(range(len(top_assoc)), top_assoc["cramers_v"], color=colors)
    ax.set_yticks(range(len(top_assoc)))
    ax.set_yticklabels(top_assoc["feature"])
    ax.set_xlabel("Cramér's V", fontsize=11)
    ax.set_title("Categorical Features Association with Churn (Top 10)", fontsize=12, weight="bold")
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    U.save_fig(fig, "F4f_cramers_v_top_categorical")
    plt.close()


# =============================================================================
# 10. Leakage detection (T9_leakage_scan + F5j)
# =============================================================================
def detect_leakage(df):
    """Detect potential data leakage flags"""
    
    # Define leakage flags using columns present in Cell2Cell
    leakage_specs = [
        ("has_cust_care_calls_gt_5", ["CustomerCareCalls"],
         lambda d: pd.to_numeric(d["CustomerCareCalls"], errors="coerce") > 5),
        ("has_high_monthly_minutes", ["MonthlyMinutes"],
         lambda d: pd.to_numeric(d["MonthlyMinutes"], errors="coerce") > pd.to_numeric(d["MonthlyMinutes"], errors="coerce").quantile(0.95)),
        ("has_high_monthly_revenue", ["MonthlyRevenue"],
         lambda d: pd.to_numeric(d["MonthlyRevenue"], errors="coerce") > pd.to_numeric(d["MonthlyRevenue"], errors="coerce").quantile(0.95)),
        ("retention_offers_accepted", ["RetentionOffersAccepted"],
         lambda d: pd.to_numeric(d["RetentionOffersAccepted"], errors="coerce") > 0),
        ("called_retention_team", ["MadeCallToRetentionTeam"],
         lambda d: d["MadeCallToRetentionTeam"].astype(str).str.contains("yes", case=False, na=False)),
    ]
    
    leakage_results = []
    
    for flag_name, required_cols, flag_builder in leakage_specs:
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"  Skipped {flag_name}: missing columns {missing_cols}")
            continue
        
        try:
            flag_col = flag_builder(df).fillna(False)
            churn_with_flag = df[flag_col]["Churn01"].mean()
            churn_without_flag = df[~flag_col]["Churn01"].mean()
            
            delta = churn_with_flag - churn_without_flag
            
            leakage_results.append({
                "flag_name": flag_name,
                "n_flag_true": flag_col.sum(),
                "n_flag_false": (~flag_col).sum(),
                "churn_rate_flag_true": round(churn_with_flag, 4),
                "churn_rate_flag_false": round(churn_without_flag, 4),
                "delta_churn_rate": round(delta, 4),
                "leakage_risk": "HIGH" if abs(delta) > 0.15 else "MEDIUM" if abs(delta) > 0.05 else "LOW"
            })
        except Exception as e:
            print(f"  Skipped {flag_name}: {e}")
            continue
    
    if not leakage_results:
        print("\nLeakage detection skipped: no eligible flags (columns missing).\n")
        return
    
    leakage_df = pd.DataFrame(leakage_results).sort_values("delta_churn_rate", key=lambda s: s.abs(), ascending=False)
    U.save_df(leakage_df, "T9_leakage_scan")
    print(f"\nLeakage detection:\n{leakage_df}")
    
    # Plot: Leakage flags impact
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(leakage_df))
    width = 0.35
    
    ax.bar(x - width/2, leakage_df["churn_rate_flag_true"], width, label="Flag=True", color="#e74c3c")
    ax.bar(x + width/2, leakage_df["churn_rate_flag_false"], width, label="Flag=False", color="#3498db")
    
    ax.set_xticks(x)
    ax.set_xticklabels(leakage_df["flag_name"], rotation=45, ha="right")
    ax.set_ylabel("Churn Rate", fontsize=11)
    ax.set_title("Data Leakage Risk: Feature Flags Impact on Churn", fontsize=12, weight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    U.save_fig(fig, "F5j_leakage_flags_churn")
    plt.close()


# =============================================================================
# 11. Summary & manifest
# =============================================================================
import json

def print_summary(run_name: str = "01_eda"):
    """
    Print summary from manifest JSON.
    Falls back to scanning artifacts/ if manifest is missing.
    """
    manifest_path = PROJECT_ROOT / "artifacts" / "meta" / f"manifest_{run_name}.json"

    print("\n" + "=" * 80)
    print(f"EDA COMPLETE - Manifest Summary ({run_name})")
    print("=" * 80)

    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))

        def _print_list(title, items):
            print(f"\n{title}:")
            if not items:
                print("  (none)")
            else:
                for x in items:
                    print(f"  ✓ {x}")

        _print_list("Tables (main)", m.get("tables_main", []))
        _print_list("Tables (appendix)", m.get("tables_appendix", []))
        _print_list("Figures (main)", m.get("figures_main", []))
        _print_list("Figures (appendix)", m.get("figures_appendix", []))

        print("\nManifest file:", manifest_path)
        print("=" * 80)
        return

    # ---- fallback: no manifest ----
    fig_dir = PROJECT_ROOT / "artifacts" / "figures"
    tab_dir = PROJECT_ROOT / "artifacts" / "tables"

    figs = sorted([p.name for p in fig_dir.glob("*.png")]) if fig_dir.exists() else []
    tabs = sorted([p.name for p in tab_dir.glob("*.csv")]) if tab_dir.exists() else []

    print("\n(Manifest not found, fallback to scanning artifacts/)")
    print("\nTables:")
    for t in tabs:
        print("  ✓", t)
    print("\nFigures:")
    for f in figs:
        print("  ✓", f)

    print("=" * 80)



# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    U.reset_run_log()
    print("Starting EDA pipeline...\n")
    
    # 1. Load
    df = load_data()
    
    # 2. Create binary target
    df = create_churn01(df)
    
    # 3. Feature registry
    reg = create_feature_registry(df)
    create_semantic_groups(reg)
    
    # 4. Target analysis
    analyze_target(df)
    
    # 5. Sparsity analysis
    analyze_sparsity(df)
    
    # 6. Value domain checks
    check_value_domain(df)
    
    # 7. Outlier detection
    detect_outliers(df)
    
    # 8. Ordinal analysis
    analyze_ordinal(df)
    
    # 9. Categorical association
    analyze_categorical_association(df)
    
    # 10. Leakage detection
    detect_leakage(df)

    # 10.5 Manifest (main vs appendix)
    U.write_manifest(
        run_name="01_eda",
        core_figures=[
            "F2a_churn_distribution_donut.png",
            "F1a_sparsity_missing_zero.png",
            "F4a_ordinal_trends.png",
            "F4f_cramers_v_top_categorical.png",
            "F5j_leakage_flags_churn.png",
        ],
        core_tables=[
            "T0_feature_registry.csv",
            "T0_semantic_groups_table.csv",
            "T2_churn_distribution.csv",
            "T1_sparsity_overview.csv",
            "T1_value_domain_checks.csv",
            "T1_outliers_iqr3.csv",
            "T4_ordinal_trends.csv",
            "T4_cramers_v_vs_churn.csv",
            "T9_leakage_scan.csv",
        ],
    )

    # 11. Summary (read manifest)
    print_summary("01_eda")

    print("\n All artifacts saved to artifacts/ directory")
