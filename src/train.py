"""Train and validate the freight-rate model.

Validation philosophy
----------------------
The validation set (``data/validation.csv``) is November-December 2025, which is
strictly *after* the training window (January-October 2025). To get an honest
read on how the model will do on that future period, the local validation split
has to imitate it: we hold out the **last 20% of the training window by date**
(a temporal forward split) rather than a random shuffle. A random split would
leak information from future dates into the model's score and flatter the
numbers.

Target transform
-----------------
``posted_rate`` is right-skewed (skew ~1.9). We model ``log(posted_rate)`` so
the booster optimises relative error, then exponentiate back. This both
stabilises variance and guarantees strictly positive predictions, which the
scorer requires.

Run
---
    python -m src.train
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

from .features import FEATURE_COLUMNS, FeatureReference, build_features

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
TRAIN_PATH = DATA_DIR / "train_test.csv"

VALIDATION_FRACTION = 0.20
RANDOM_STATE = 42

LGB_PARAMS = {
    "objective": "regression_l2",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_estimators": 3000,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}


def temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split so the newest ``VALIDATION_FRACTION`` of dates becomes validation.

    This mirrors the real task: predict a period that comes after everything
    the model trained on.
    """
    frame = frame.sort_values("date").reset_index(drop=True)
    cutoff = frame["date"].quantile(1.0 - VALIDATION_FRACTION)
    train = frame[frame["date"] <= cutoff]
    valid = frame[frame["date"] > cutoff]
    return train, valid


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def load_training_frame() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    # Drop rows with a non-positive or missing target: they cannot be used to
    # learn a rate and the scorer forbids non-positive rates anyway.
    df = df[df["posted_rate"].notna() & (df["posted_rate"] > 0)].copy()
    return df


def fit_model(X: pd.DataFrame, y_log: np.ndarray, X_val=None, y_val_log=None) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    fit_kwargs: dict = {"categorical_feature": ["equipment", "lane"]}
    if X_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val_log)]
        fit_kwargs["callbacks"] = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
    model.fit(X, y_log, **fit_kwargs)
    return model


def main() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    frame = load_training_frame()
    print(f"Loaded {len(frame):,} usable training rows "
          f"({frame['date'].min().date()} to {frame['date'].max().date()}).")

    # Build and persist the reference table (city coords + median signals) so the
    # December file -- which lacks those columns -- can run the same features.
    reference = FeatureReference.from_training(frame)
    reference.save(ARTIFACT_DIR / "feature_reference.json")

    # 1) Temporal holdout to estimate forward-looking performance.
    train_df, valid_df = temporal_split(frame)
    print(f"Temporal split -> train {len(train_df):,} rows "
          f"(<= {train_df['date'].max().date()}), "
          f"holdout {len(valid_df):,} rows (>= {valid_df['date'].min().date()}).")

    X_train = build_features(train_df)
    X_valid = build_features(valid_df)
    y_train_log = np.log(train_df["posted_rate"].to_numpy())
    y_valid = valid_df["posted_rate"].to_numpy()
    y_valid_log = np.log(y_valid)

    val_model = fit_model(X_train, y_train_log, X_valid, y_valid_log)
    best_iter = val_model.best_iteration_ or LGB_PARAMS["n_estimators"]
    valid_pred = np.exp(val_model.predict(X_valid, num_iteration=best_iter))
    metrics = _metrics(y_valid, valid_pred)
    metrics["best_iteration"] = int(best_iter)
    print("Temporal holdout metrics:")
    for k, v in metrics.items():
        print(f"  {k:>14}: {v:,.4f}")

    # 2) Refit on ALL training data (using the tuned iteration count) so the
    #    model that scores the real validation set has seen every available row,
    #    including the most recent dates closest to November/December.
    X_full = build_features(frame)
    y_full_log = np.log(frame["posted_rate"].to_numpy())
    final_params = dict(LGB_PARAMS)
    final_params["n_estimators"] = int(best_iter)
    final_model = lgb.LGBMRegressor(**final_params)
    final_model.fit(X_full, y_full_log, categorical_feature=["equipment", "lane"])

    model_path = ARTIFACT_DIR / "model.txt"
    final_model.booster_.save_model(str(model_path))

    importances = dict(
        sorted(
            zip(FEATURE_COLUMNS, final_model.feature_importances_.tolist()),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )
    report = {
        "n_training_rows": int(len(frame)),
        "validation_fraction": VALIDATION_FRACTION,
        "temporal_cutoff": str(train_df["date"].max().date()),
        "holdout_metrics": metrics,
        "feature_importance": importances,
        "lgb_params": LGB_PARAMS,
    }
    (ARTIFACT_DIR / "training_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved model to {model_path}")
    print(f"Saved metrics to {ARTIFACT_DIR / 'training_report.json'}")


if __name__ == "__main__":
    main()
