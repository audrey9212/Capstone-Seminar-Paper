
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================================
# Robust ROC AUC Scorer for Cross-Validation
# ============================================================================
def create_robust_roc_auc_scorer():
    """
     ROC AUC scorer cross_val_score
    
    
    -  (y_true, y_score)  (estimator, X, y_true)
    -  needs_proba=True  sklearn  predict_proba
    """
    def robust_roc_auc_score_func(y_true, y_score, *args, **kwargs):
        """
         ROC AUC
        
        :
            y_true: 
            y_score: predict_proba sklearn 
            *args, **kwargs:  sklearn  needs_proba 
        """
        try:
            #  2D 
            if isinstance(y_score, np.ndarray) and y_score.ndim == 2:
                if y_score.shape[1] == 2:
                    y_score = y_score[:, 1]  # 
                else:
                    y_score = y_score.max(axis=1)  # 
            
            #  AUC
            if len(np.unique(y_true)) < 2:
                return 0.5
            if len(np.unique(y_score)) < 2:
                return 0.5
            
            return roc_auc_score(y_true, y_score)
        
        except Exception as e:
            print(f"[WARN] Scorer error: {str(e)[:80]}")
            return 0.5
    
    # needs_proba=True
    return make_scorer(
        robust_roc_auc_score_func,
        needs_proba=True,  # sklearn  predict_proba
        greater_is_better=True,
    )


# ============================================================================
# Optuna Objective Functions
# ============================================================================

def suggest_rf_params(trial):
    """Random Forest """
    return {
        "model__n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "model__max_depth": trial.suggest_int("max_depth", 6, 15),
        "model__min_samples_split": trial.suggest_int("min_samples_split", 4, 20),
        "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
        "model__max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }


def suggest_xgb_params(trial):
    """XGBoost """
    return {
        "model__max_depth": trial.suggest_int("max_depth", 3, 10),
        "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "model__n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "model__subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "model__min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "model__gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "model__reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "model__reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
    }


def suggest_lgbm_params(trial):
    """LightGBM """
    return {
        "model__num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "model__max_depth": trial.suggest_int("max_depth", 3, 15),
        "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "model__n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "model__subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "model__min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "model__reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "model__reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
    }


PARAM_SUGGEST_FUNCS = {
    "rf": suggest_rf_params,
    "xgb": suggest_xgb_params,
    "lgbm": suggest_lgbm_params,
}


# ============================================================================
# 
# ============================================================================

def run_optuna_optimization(
    model_name: str,
    model_type: str,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    preprocessor,
    tree_features: list,
    base_estimator,
    n_trials: int = 50,
    cv_folds: int = 5,
    checkpoint_dir: str = "./checkpoints",
    force_retrain: bool = False,
):
    """
     Optuna 
    
    :
        model_name: 
        model_type:  ('rf', 'xgb', 'lgbm')
        X_train, y_train: 
        X_val, y_val: 
        X_test, y_test: 
        preprocessor: sklearn preprocessor
        tree_features: 
        base_estimator: 
        n_trials: Optuna 
        cv_folds: Cross-validation folds
        checkpoint_dir: checkpoint 
        force_retrain: 
    
    :
        (best_model, info_dict)
    """
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        brier_score_loss,
    )
    
    print("\\n" + "="*80)
    print(f" Optuna : {model_name}")
    print("="*80)

    # tree_features 
    missing_features = set(tree_features) - set(X_train.columns)
    if missing_features:
        raise ValueError(f"Missing features in X_train: {missing_features}")
    
    #  checkpoint 
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    study_file = checkpoint_path / f"{model_name}_study.pkl"
    model_file = checkpoint_path / f"{model_name}_best_model.pkl"
    
    # -------------------------------------------------------------------
    # 
    # -------------------------------------------------------------------
    if study_file.exists() and model_file.exists() and not force_retrain:
        print(f" : {model_file}")
        print("    force_retrain=True")

        try:
            with open(model_file, "rb") as f:
                best_model = pickle.load(f)

            with open(study_file, "rb") as f:
                study = pickle.load(f)

            # 
            def _eval_cached(X, y, split_name):
                proba = best_model.predict_proba(X[tree_features])[:, 1]
                return {
                    "auc": roc_auc_score(y, proba),
                    "pr_auc": average_precision_score(y, proba),
                    "brier": brier_score_loss(y, proba),
                }

            metrics = {
                "train": _eval_cached(X_train, y_train, "Train"),
                "val": _eval_cached(X_val, y_val, "Val"),
                "test": _eval_cached(X_test, y_test, "Test"),
            }

            info = {
                "best_params": study.best_params,
                "best_cv_auc": study.best_value,
                "train_auc": metrics["train"]["auc"],
                "val_auc": metrics["val"]["auc"],
                "test_auc": metrics["test"]["auc"],
                "n_trials": len(study.trials),
                "study": study,
            }

            print(f"[OK]  | Val AUC: {info['val_auc']:.4f}")
            return best_model, info

        except (AttributeError, ModuleNotFoundError, ImportError) as e:
            print(f"[WARN] sklearn : {e}")
            print("   ...")
            # 
    
    # -------------------------------------------------------------------
    #  scorer cross_val_score CV
    # -------------------------------------------------------------------
    roc_auc_scorer = create_robust_roc_auc_scorer()
    cv = cv_folds
    if isinstance(cv_folds, int):
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # 
    if model_type not in PARAM_SUGGEST_FUNCS:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(PARAM_SUGGEST_FUNCS.keys())}")
    
    suggest_params = PARAM_SUGGEST_FUNCS[model_type]
    
    # -------------------------------------------------------------------
    #  Optuna Objective
    # -------------------------------------------------------------------
    def objective(trial):
        """
        Optuna 
        """
        # 1. 
        params = suggest_params(trial)
        
        # 2.  RF  class_weight  balanced
        base_params = base_estimator.get_params()
        if model_type == "rf" and base_params.get("class_weight") is None:
            base_params["class_weight"] = "balanced"
        model = base_estimator.__class__(**{
            **base_params,
            **{k.replace("model__", ""): v for k, v in params.items()}
        })
        
        # 3.  pipeline
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", model),
        ])
        
        # 4. Cross-validation scorer
        try:
            scores = cross_val_score(
                pipeline,
                X_train[tree_features],
                y_train,
                cv=cv,
                scoring=roc_auc_scorer,
                n_jobs=-1,
                error_score="raise",
            )
            
            mean_score = scores.mean()
            
            # 
            trial.set_user_attr("cv_scores", scores.tolist())
            trial.set_user_attr("cv_std", scores.std())
            
            return mean_score
        
        except Exception as e:
            print(f"[WARN] Trial {trial.number} failed: {e}")
            return 0.5  # 
    
    # -------------------------------------------------------------------
    #  Optuna 
    # -------------------------------------------------------------------
    print(f"  Optuna {n_trials} trials, {cv_folds}-fold CV...")
    
    study = optuna.create_study(
        direction="maximize",
        study_name=model_name,
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,  # Optuna  cross_val_score 
    )
    
    print(f"\\n[OK] ")
    print(f"   CV AUC: {study.best_value:.4f}")
    print(f"  : {study.best_params}")
    
    # -------------------------------------------------------------------
    # 
    # -------------------------------------------------------------------
    print("\\n ...")
    
    best_params = {k.replace("model__", ""): v for k, v in study.best_params.items()}
    final_model = base_estimator.__class__(**{
        **base_estimator.get_params(),
        **best_params
    })
    
    best_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", final_model),
    ])
    
    best_pipeline.fit(X_train[tree_features], y_train)
    
    # -------------------------------------------------------------------
    # 
    # -------------------------------------------------------------------
    def _eval(X, y, split_name):
        proba = best_pipeline.predict_proba(X[tree_features])[:, 1]
        auc = roc_auc_score(y, proba)
        pr_auc = average_precision_score(y, proba)
        brier = brier_score_loss(y, proba)
        print(f"  {split_name:5s} | AUC={auc:.4f} PR-AUC={pr_auc:.4f} Brier={brier:.4f}")
        return {"auc": auc, "pr_auc": pr_auc, "brier": brier}
    
    print(f"\\n[TABLES] {model_name} ")
    metrics = {
        "train": _eval(X_train, y_train, "Train"),
        "val": _eval(X_val, y_val, "Val"),
        "test": _eval(X_test, y_test, "Test"),
    }
    
    # -------------------------------------------------------------------
    # 
    # -------------------------------------------------------------------
    print("\\n  study...")
    with open(model_file, "wb") as f:
        pickle.dump(best_pipeline, f)
    
    with open(study_file, "wb") as f:
        pickle.dump(study, f)
    
    print(f"[OK] : {model_file}")
    print(f"[OK] : {study_file}")
    
    # -------------------------------------------------------------------
    # 
    # -------------------------------------------------------------------
    info = {
        "best_params": study.best_params,
        "best_cv_auc": study.best_value,
        "train_auc": metrics["train"]["auc"],
        "val_auc": metrics["val"]["auc"],
        "test_auc": metrics["test"]["auc"],
        "n_trials": len(study.trials),
        "study": study,
    }
    
    print("="*80)
    print(f" Optuna : {model_name}")
    print("="*80)
    
    return best_pipeline, info


# ============================================================================
# 
# ============================================================================

def plot_optuna_optimization(info_dict, save_dir="./figures", show_inline=True):
    """
     Optuna 
    
    :
        info_dict: run_optuna_optimization  info 
        save_dir: 
        show_inline:  Notebook 
    """
    study = info_dict.get("study")
    if study is None:
        print("[WARN]  study ")
        return
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    model_name = study.study_name

    def _safe_write(fig, filename):
        png_path = save_path / filename
        try:
            fig.write_image(str(png_path))
            print(f"[OK] : {png_path}")
        except Exception as e:
            html_path = png_path.with_suffix(".html")
            fig.write_html(str(html_path))
            print(f"[WARN]  PNG  kaleido: {e};  HTML: {html_path}")
    
    # 1. Optimization History
    print("[TABLES] ...")
    fig = plot_optimization_history(study)
    _safe_write(fig, f"{model_name}_optimization_history.png")
    if show_inline:
        fig.show()
    
    # 2. Parameter Importances
    print("[TABLES] ...")
    fig = plot_param_importances(study)
    _safe_write(fig, f"{model_name}_param_importances.png")
    if show_inline:
        fig.show()
    
    # 3. Parallel Coordinate
    print("[TABLES] ...")
    fig = plot_parallel_coordinate(study)
    _safe_write(fig, f"{model_name}_parallel_coordinate.png")
    if show_inline:
        fig.show()
    
    print(f"[OK] : {save_path}")
