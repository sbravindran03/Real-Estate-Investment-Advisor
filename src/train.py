"""Steps 3 & 4 - Model development + MLflow tracking / registry.

Trains 6 classifiers (Good_Investment) and 6 regressors (Future_Price_5Y), logs parameters,
metrics, plots and model artifacts to MLflow, registers the best model of each task in the
MLflow Model Registry (alias "champion") and saves them to models/ for the Streamlit app.

Run (after preprocess.py):  python src/train.py            # full data
                             python src/train.py --sample 50000   # quicker experiment
"""
import argparse
import json
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from config import (CATEGORICAL_FEATURES, CLF_TARGET, EXPERIMENT_CLF, EXPERIMENT_REG,
                    MLFLOW_ARTIFACT_ROOT, MLFLOW_TRACKING_URI, MODELS_DIR, NUMERIC_FEATURES,
                    PROCESSED_CSV, RANDOM_STATE, REG_TARGET, REGISTERED_CLF, REGISTERED_REG,
                    REPORTS_DIR)

warnings.filterwarnings("ignore")
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RS = RANDOM_STATE

CLASSIFIERS = {
    "LogisticRegression": LogisticRegression(max_iter=2000, C=1.0),
    "DecisionTree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=RS),
    "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=5,
                                           n_jobs=-1, random_state=RS),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=RS),
    "XGBoost": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9,
                             colsample_bytree=0.9, n_jobs=-1, random_state=RS, eval_metric="logloss"),
    "LightGBM": LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                               random_state=RS, verbose=-1),
}
REGRESSORS = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "DecisionTree": DecisionTreeRegressor(max_depth=14, min_samples_leaf=10, random_state=RS),
    "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=20, min_samples_leaf=3,
                                          n_jobs=-1, random_state=RS),
    "XGBoost": XGBRegressor(n_estimators=600, max_depth=7, learning_rate=0.05, subsample=0.9,
                            colsample_bytree=0.9, n_jobs=-1, random_state=RS),
    "LightGBM": LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                              random_state=RS, verbose=-1),
}


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def feature_importance(pipe: Pipeline) -> pd.Series:
    """Importance per *original* feature (one-hot columns summed back to their source)."""
    model = pipe.named_steps["model"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        imp = np.abs(np.ravel(model.coef_))
    else:
        return pd.Series(dtype=float)
    src = []
    for n in names:
        kind, col = n.split("__", 1)
        if kind == "cat":
            col = next(c for c in CATEGORICAL_FEATURES if col.startswith(c + "_"))
        src.append(col)
    s = pd.Series(imp, index=src).groupby(level=0).sum()
    return (s / s.sum()).sort_values(ascending=False)


def log_model(pipe, X_sample, preds):
    sig = infer_signature(X_sample, preds)
    try:   # MLflow >= 3
        return mlflow.sklearn.log_model(pipe, name="model", signature=sig, input_example=X_sample.head(3))
    except TypeError:   # MLflow 2.x
        return mlflow.sklearn.log_model(pipe, artifact_path="model", signature=sig,
                                        input_example=X_sample.head(3))


def get_experiment(name):
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return mlflow.create_experiment(name, artifact_location=f"{MLFLOW_ARTIFACT_ROOT}/{name}")
    return exp.experiment_id


def bar_importance(imp: pd.Series, title: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.head(15)[::-1].plot.barh(ax=ax, color="#2a9d8f")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- classification
def run_classification(X_tr, X_te, y_tr, y_te):
    mlflow.set_experiment(experiment_id=get_experiment(EXPERIMENT_CLF))
    results = []
    for name, est in CLASSIFIERS.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", est)])
        with mlflow.start_run(run_name=name) as run:
            t0 = time.time()
            pipe.fit(X_tr, y_tr)
            train_time = time.time() - t0
            pred = pipe.predict(X_te)
            proba = pipe.predict_proba(X_te)[:, 1]
            m = {"accuracy": accuracy_score(y_te, pred), "precision": precision_score(y_te, pred),
                 "recall": recall_score(y_te, pred), "f1": f1_score(y_te, pred),
                 "roc_auc": roc_auc_score(y_te, proba), "train_time_sec": train_time}
            mlflow.log_params({"model_type": name, "n_train": len(X_tr), "n_features": len(FEATURES),
                               **{k: v for k, v in est.get_params().items()
                                  if isinstance(v, (int, float, str, bool)) and v is not None}})
            mlflow.log_metrics(m)

            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(confusion_matrix(y_te, pred), display_labels=["Not good", "Good"]).plot(
                ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(f"{name} — confusion matrix")
            fig.tight_layout(); mlflow.log_figure(fig, "confusion_matrix.png"); plt.close(fig)
            imp = feature_importance(pipe)
            if len(imp):
                f2 = bar_importance(imp, f"{name} — feature importance")
                mlflow.log_figure(f2, "feature_importance.png"); plt.close(f2)
                mlflow.log_dict(imp.round(5).to_dict(), "feature_importance.json")
            info = log_model(pipe, X_te.head(200), pred[:200])
            results.append({"model": name, **m, "run_id": run.info.run_id, "model_uri": info.model_uri,
                            "pipe": pipe, "cm": confusion_matrix(y_te, pred).tolist(), "imp": imp})
            print(f"[CLF] {name:22s} acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f} "
                  f"({train_time:.0f}s)")
    return results


# --------------------------------------------------------------------------- regression
def run_regression(X_tr, X_te, y_tr, y_te):
    mlflow.set_experiment(experiment_id=get_experiment(EXPERIMENT_REG))
    results = []
    for name, est in REGRESSORS.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", est)])
        with mlflow.start_run(run_name=name) as run:
            t0 = time.time()
            pipe.fit(X_tr, y_tr)
            train_time = time.time() - t0
            pred = pipe.predict(X_te)
            mse = mean_squared_error(y_te, pred)
            m = {"rmse": float(np.sqrt(mse)), "mse": mse, "mae": mean_absolute_error(y_te, pred),
                 "r2": r2_score(y_te, pred),
                 "mape": float(np.mean(np.abs((y_te - pred) / y_te)) * 100), "train_time_sec": train_time}
            mlflow.log_params({"model_type": name, "n_train": len(X_tr), "n_features": len(FEATURES),
                               **{k: v for k, v in est.get_params().items()
                                  if isinstance(v, (int, float, str, bool)) and v is not None}})
            mlflow.log_metrics(m)

            fig, ax = plt.subplots(figsize=(5, 5))
            idx = np.random.default_rng(RS).choice(len(y_te), min(5000, len(y_te)), replace=False)
            ax.scatter(np.asarray(y_te)[idx], pred[idx], s=4, alpha=0.3)
            lim = [float(np.min(y_te)), float(np.max(y_te))]
            ax.plot(lim, lim, "r--"); ax.set_xlabel("Actual (Lakhs)"); ax.set_ylabel("Predicted (Lakhs)")
            ax.set_title(f"{name} — actual vs predicted")
            fig.tight_layout(); mlflow.log_figure(fig, "actual_vs_predicted.png"); plt.close(fig)
            imp = feature_importance(pipe)
            if len(imp):
                f2 = bar_importance(imp, f"{name} — feature importance")
                mlflow.log_figure(f2, "feature_importance.png"); plt.close(f2)
                mlflow.log_dict(imp.round(5).to_dict(), "feature_importance.json")
            info = log_model(pipe, X_te.head(200), pred[:200])
            results.append({"model": name, **m, "run_id": run.info.run_id, "model_uri": info.model_uri,
                            "pipe": pipe, "imp": imp})
            print(f"[REG] {name:22s} rmse={m['rmse']:.3f} mae={m['mae']:.3f} r2={m['r2']:.5f} ({train_time:.0f}s)")
    return results


def register_best(client, results, registered_name, metric, higher_is_better, exclude=()):
    candidates = [r for r in results if r["model"] not in exclude]
    best = sorted(candidates, key=lambda r: r[metric], reverse=higher_is_better)[0]
    mv = mlflow.register_model(best["model_uri"], registered_name)
    client.set_registered_model_alias(registered_name, "champion", mv.version)
    client.update_model_version(registered_name, mv.version,
                                description=f"Best {best['model']} by {metric} = {best[metric]:.4f}")
    print(f"Registered {registered_name} v{mv.version} <- {best['model']} ({metric}={best[metric]:.4f})")
    return best, mv.version


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="train on a random subset of N rows")
    args = ap.parse_args()

    df = pd.read_csv(PROCESSED_CSV)
    if args.sample:
        df = df.sample(args.sample, random_state=RS)
    X = df[FEATURES]
    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, df[CLF_TARGET], df[REG_TARGET], test_size=0.2, random_state=RS, stratify=df[CLF_TARGET])
    print(f"Train {len(X_tr):,} / test {len(X_te):,}  features={len(FEATURES)}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    clf_res = run_classification(X_tr, X_te, yc_tr, yc_te)
    reg_res = run_regression(X_tr, X_te, yr_tr, yr_te)
    # A single decision tree has pure leaves, so its predict_proba is almost always 0 or 1 and
    # the app's "confidence" would be meaningless. Pick the champion among models with graded
    # probabilities (the tree's F1 is only ~0.0003 higher anyway).
    best_clf, clf_ver = register_best(client, clf_res, REGISTERED_CLF, "f1", True, exclude=("DecisionTree",))
    best_reg, reg_ver = register_best(client, reg_res, REGISTERED_REG, "rmse", False)

    # Local copies for the Streamlit app (fast load, no MLflow server needed)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_clf["pipe"], MODELS_DIR / "classifier.joblib", compress=3)
    joblib.dump(best_reg["pipe"], MODELS_DIR / "regressor.joblib", compress=3)
    strip = lambda rs, keys: [{k: v for k, v in r.items() if k in keys} for r in rs]
    meta = {
        "features": FEATURES,
        "classification": {
            "best_model": best_clf["model"], "registry": f"{REGISTERED_CLF} v{clf_ver}",
            "confusion_matrix": best_clf["cm"],
            "feature_importance": best_clf["imp"].round(5).to_dict(),
            "leaderboard": strip(clf_res, ["model", "accuracy", "precision", "recall", "f1", "roc_auc",
                                            "train_time_sec"]),
        },
        "regression": {
            "best_model": best_reg["model"], "registry": f"{REGISTERED_REG} v{reg_ver}",
            "feature_importance": best_reg["imp"].round(5).to_dict(),
            "leaderboard": strip(reg_res, ["model", "rmse", "mse", "mae", "r2", "mape", "train_time_sec"]),
            "test_rmse": best_reg["rmse"],
        },
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=1, default=float)

    REPORTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(meta["classification"]["leaderboard"]).to_csv(REPORTS_DIR / "classification_results.csv", index=False)
    pd.DataFrame(meta["regression"]["leaderboard"]).to_csv(REPORTS_DIR / "regression_results.csv", index=False)
    print("Saved models + metadata to", MODELS_DIR)


if __name__ == "__main__":
    main()
