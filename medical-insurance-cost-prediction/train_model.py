"""Train and compare regression models for medical insurance charge prediction.

Run from the repository root:
    python train_model.py --data data/insurance.csv
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "charges"
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
ASSETS_DIR = ROOT / "assets" / "figures"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nDownload insurance.csv from "
            "https://www.kaggle.com/datasets/mirichoi0218/insurance and place it in data/."
        )
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[REQUIRED_COLUMNS].copy()
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string").str.strip().str.lower()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    df = df.drop_duplicates().dropna(subset=REQUIRED_COLUMNS)
    if df.empty:
        raise ValueError("No usable rows remain after removing duplicates and missing values.")
    if (df[TARGET] < 0).any():
        raise ValueError("'charges' contains negative values; please check the source data.")
    if not set(df["sex"].unique()).issubset({"female", "male"}):
        warnings.warn("Unexpected values found in 'sex'; they will be one-hot encoded.")
    if not set(df["smoker"].unique()).issubset({"yes", "no"}):
        warnings.warn("Unexpected values found in 'smoker'; they will be one-hot encoded.")
    return df.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def build_candidates() -> dict:
    # Each candidate includes its own preprocessing to avoid data leakage in CV.
    return {
        "Linear Regression": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", LinearRegression()),
        ]),
        "Ridge Regression": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=350, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
        "Extra Trees": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", ExtraTreesRegressor(
                n_estimators=350, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
        "Histogram Gradient Boosting": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", HistGradientBoostingRegressor(
                max_iter=250, learning_rate=0.06, l2_regularization=1.0,
                random_state=RANDOM_STATE
            )),
        ]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/insurance.csv", help="Path to insurance.csv")
    parser.add_argument("--test-size", type=float, default=0.20, help="Hold-out test fraction")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    args = parser.parse_args()

    if not 0.1 <= args.test_size <= 0.4:
        raise ValueError("--test-size should be between 0.1 and 0.4.")
    if args.cv < 2:
        raise ValueError("--cv must be at least 2.")

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    df = load_dataset(data_path)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE
    )

    candidates = build_candidates()
    cv = KFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    fitted = {}
    print(f"Rows used: {len(df)} | train: {len(X_train)} | test: {len(X_test)}")
    print(f"Comparing {len(candidates)} models with {args.cv}-fold CV on training data...")

    for name, estimator in candidates.items():
        scores = cross_val_score(
            estimator, X_train, y_train, cv=cv,
            scoring="neg_mean_absolute_error", n_jobs=1
        )
        mean_mae = float(-scores.mean())
        std_mae = float(scores.std())
        rows.append({"model": name, "cv_mae_mean": mean_mae, "cv_mae_std": std_mae})
        fitted[name] = estimator
        print(f"{name:30s} CV MAE: {mean_mae:,.2f} (+/- {std_mae:,.2f})")

    comparison = pd.DataFrame(rows).sort_values("cv_mae_mean").reset_index(drop=True)
    best_name = str(comparison.loc[0, "model"])
    best_model = fitted[best_name]
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    metrics = {
        "selected_model": best_name,
        "test_rows": int(len(y_test)),
        "train_rows": int(len(y_train)),
        "dataset_rows_after_cleaning": int(len(df)),
        "test_size_fraction": args.test_size,
        "cv_folds": args.cv,
        "random_state": RANDOM_STATE,
        "test_mae": float(mean_absolute_error(y_test, predictions)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "test_r2": float(r2_score(y_test, predictions)),
        "note": "Metrics are generated by running train_model.py on the supplied dataset; they are not pre-filled estimates.",
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "insurance_model.joblib")
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    with (REPORTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({
        "actual_charges": y_test.to_numpy(),
        "predicted_charges": predictions,
        "absolute_error": np.abs(y_test.to_numpy() - predictions),
    }).to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    # Permutation importance is model-agnostic and computed only on the hold-out set.
    from sklearn.inspection import permutation_importance
    importance = permutation_importance(
        best_model, X_test, y_test, n_repeats=20,
        random_state=RANDOM_STATE, scoring="neg_mean_absolute_error"
    )
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)

    # Plot actual vs predicted.
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, predictions, alpha=0.65)
    limits = [min(float(y_test.min()), float(predictions.min())),
              max(float(y_test.max()), float(predictions.max()))]
    ax.plot(limits, limits, linestyle="--")
    ax.set(xlabel="Actual charges", ylabel="Predicted charges",
           title=f"Actual vs predicted — {best_name}")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    # Plot permutation importance.
    plot_df = importance_df.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df["feature"], plot_df["importance_mean"], xerr=plot_df["importance_std"])
    ax.set(xlabel="Decrease in score when feature is permuted (MAE-based)",
           title="Permutation feature importance")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "permutation_importance.png", dpi=160)
    plt.close(fig)

    print("\nHold-out test results (computed from this run):")
    print(f"Selected model: {best_name}")
    print(f"MAE : {metrics['test_mae']:,.2f}")
    print(f"RMSE: {metrics['test_rmse']:,.2f}")
    print(f"R²  : {metrics['test_r2']:.4f}")
    print(f"\nArtifacts saved in: {ROOT}")


if __name__ == "__main__":
    main()
