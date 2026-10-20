"""Generate the two deliverable prediction files from the trained model.

Outputs
-------
* ``validation_predictions.csv`` at the repo root -- exactly the two columns
  ``load_id,predicted_rate`` for all 12,000 validation loads, in template order.
* ``data/december_chart_inputs.csv`` -- the same seven columns it ships with,
  with the ``predicted_rate`` column filled for all 31 December days.

Both files are written in the precise shape ``score.py`` validates.

Run
---
    python -m src.predict
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FeatureReference, build_features

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "model.txt"
REFERENCE_PATH = ARTIFACT_DIR / "feature_reference.json"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"
VALIDATION_OUTPUT = ROOT / "validation_predictions.csv"


def load_booster() -> lgb.Booster:
    if not MODEL_PATH.is_file():
        raise SystemExit(
            f"Model not found at {MODEL_PATH}. Run `python -m src.train` first."
        )
    return lgb.Booster(model_file=str(MODEL_PATH))


def _predict_rates(
    booster: lgb.Booster, raw: pd.DataFrame, reference: FeatureReference
) -> np.ndarray:
    features = build_features(raw, reference)
    log_pred = booster.predict(features)
    rates = np.exp(log_pred)
    # The model trained on log-rate so output is already > 0; this is a belt-and
    # -braces guard against any pathological row the scorer would reject.
    return np.maximum(rates, 1.0).round(2)


def predict_validation(booster: lgb.Booster, reference: FeatureReference) -> None:
    raw = pd.read_csv(VALIDATION_PATH, parse_dates=["date"])
    template = pd.read_csv(TEMPLATE_PATH)

    raw = raw.assign(predicted_rate=_predict_rates(booster, raw, reference))
    rate_by_id = dict(zip(raw["load_id"], raw["predicted_rate"]))

    # Emit rows in the template's own order so the file lines up 1:1 with it.
    out = template[["load_id"]].copy()
    out["predicted_rate"] = out["load_id"].map(rate_by_id)
    if out["predicted_rate"].isna().any():
        missing = int(out["predicted_rate"].isna().sum())
        raise SystemExit(f"{missing} template load_ids had no prediction.")

    out.to_csv(VALIDATION_OUTPUT, index=False)
    print(f"Wrote {len(out):,} validation predictions -> {VALIDATION_OUTPUT}")


def predict_december(booster: lgb.Booster, reference: FeatureReference) -> None:
    december = pd.read_csv(DECEMBER_PATH)
    raw = december.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    december["predicted_rate"] = _predict_rates(booster, raw, reference)
    # Preserve the original column order the scorer checks for.
    december.to_csv(DECEMBER_PATH, index=False)
    print(f"Filled {len(december)} December rows -> {DECEMBER_PATH}")


def main() -> None:
    booster = load_booster()
    if not REFERENCE_PATH.is_file():
        raise SystemExit(
            f"Feature reference not found at {REFERENCE_PATH}. Run `python -m src.train` first."
        )
    reference = FeatureReference.load(REFERENCE_PATH)
    predict_validation(booster, reference)
    predict_december(booster, reference)


if __name__ == "__main__":
    main()
