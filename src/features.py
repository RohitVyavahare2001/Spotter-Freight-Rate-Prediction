"""Feature engineering for the freight-rate model.

Every transformation here is deterministic and depends only on a single row's
own values, so the exact same code path runs for the training rows, the 12,000
validation loads, and the 31 fixed December rows. Keeping the logic in one place
is what guarantees the model never sees a feature at predict time that it did
not see at train time.

Design notes
------------
* The raw ``distance`` column already encodes real road mileage (it runs ~1.17x
  the straight-line haversine distance, i.e. it carries route circuity), so we
  trust it as the primary driver and only *add* a haversine feature as a cheap
  sanity signal rather than replacing it.
* ``weight`` carries a handful of impossible values (negatives, zeros) and some
  missing entries. We clip it to a plausible freight range and let the model's
  own missing-value handling deal with the gaps, so no row is ever dropped at
  predict time.
* Dates are turned into smooth cyclical features plus a linear "day index" so
  the model can pick up both the mild month-to-month trend and any weekly
  rhythm without us hard-coding a calendar.
* The December chart file only carries seven columns (``pickup``, ``delivery``,
  ``distance``, ``equipment``, ``weight``, ``date``, ``predicted_rate``) -- it
  has no coordinates or market signals. To run the identical feature pipeline
  on it, ``build_features`` accepts a small ``reference`` table (learned from
  the training data and saved alongside the model) that supplies per-city
  coordinates and median market signals for any column the input is missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# A freight load realistically weighs between an empty-ish partial and the
# ~48k lb legal payload ceiling. Anything outside this band is a data error, so
# we clip rather than trust it.
WEIGHT_FLOOR = 1_000.0
WEIGHT_CEIL = 48_000.0

# Reference date used to turn calendar dates into a single linear index. Using a
# fixed anchor (not the min of whatever frame we happen to pass) keeps the
# feature identical across train, validation and December.
EPOCH = pd.Timestamp("2025-01-01")

EARTH_RADIUS_MILES = 3958.8

# Columns the model consumes. Order is fixed so the trained booster always
# receives its features in the layout it learned on.
CATEGORICAL_FEATURES = ["equipment", "lane"]
NUMERIC_FEATURES = [
    "distance",
    "haversine_miles",
    "circuity",
    "weight",
    "weight_per_mile",
    "market_index",
    "quote_signal",
    "day_index",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class FeatureReference:
    """Lookups learned from training data to backfill missing input columns.

    This is what lets the exact same feature code run on the December file,
    which lacks coordinates and market signals.
    """

    pickup_coords: dict = field(default_factory=dict)  # city -> (lat, lon)
    delivery_coords: dict = field(default_factory=dict)  # city -> (lat, lon)
    market_index_median: float = 1.0
    quote_signal_median: float = 2.0

    @classmethod
    def from_training(cls, frame: pd.DataFrame) -> "FeatureReference":
        pickup = (
            frame.dropna(subset=["pickup_lat", "pickup_lon"])
            .groupby("pickup")[["pickup_lat", "pickup_lon"]]
            .median()
        )
        delivery = (
            frame.dropna(subset=["delivery_lat", "delivery_lon"])
            .groupby("delivery")[["delivery_lat", "delivery_lon"]]
            .median()
        )
        return cls(
            pickup_coords={c: (r.pickup_lat, r.pickup_lon) for c, r in pickup.iterrows()},
            delivery_coords={c: (r.delivery_lat, r.delivery_lon) for c, r in delivery.iterrows()},
            market_index_median=float(frame["market_index"].median()),
            quote_signal_median=float(frame["quote_signal"].median()),
        )

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "pickup_coords": {k: list(v) for k, v in self.pickup_coords.items()},
                    "delivery_coords": {k: list(v) for k, v in self.delivery_coords.items()},
                    "market_index_median": self.market_index_median,
                    "quote_signal_median": self.quote_signal_median,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> "FeatureReference":
        blob = json.loads(Path(path).read_text())
        return cls(
            pickup_coords={k: tuple(v) for k, v in blob["pickup_coords"].items()},
            delivery_coords={k: tuple(v) for k, v in blob["delivery_coords"].items()},
            market_index_median=blob["market_index_median"],
            quote_signal_median=blob["quote_signal_median"],
        )


def _ensure_columns(df: pd.DataFrame, reference: "FeatureReference | None") -> pd.DataFrame:
    """Backfill any feature-source column the raw input does not carry."""
    df = df.copy()
    ref = reference or FeatureReference()

    if "pickup_lat" not in df.columns or "pickup_lon" not in df.columns:
        df["pickup_lat"] = df["pickup"].map(lambda c: ref.pickup_coords.get(c, (np.nan, np.nan))[0])
        df["pickup_lon"] = df["pickup"].map(lambda c: ref.pickup_coords.get(c, (np.nan, np.nan))[1])
    if "delivery_lat" not in df.columns or "delivery_lon" not in df.columns:
        df["delivery_lat"] = df["delivery"].map(
            lambda c: ref.delivery_coords.get(c, (np.nan, np.nan))[0]
        )
        df["delivery_lon"] = df["delivery"].map(
            lambda c: ref.delivery_coords.get(c, (np.nan, np.nan))[1]
        )
    if "market_index" not in df.columns:
        df["market_index"] = ref.market_index_median
    if "quote_signal" not in df.columns:
        df["quote_signal"] = ref.quote_signal_median
    return df


def _haversine_miles(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    """Great-circle distance in miles between two lat/lon points."""
    lat1r, lon1r, lat2r, lon2r = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    inner = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_MILES * np.arcsin(np.clip(np.sqrt(inner), 0.0, 1.0))


def build_features(
    frame: pd.DataFrame, reference: "FeatureReference | None" = None
) -> pd.DataFrame:
    """Return a new frame of model-ready features built from a raw load frame.

    The input may be the training data, the validation set, or the December
    inputs. For inputs that lack coordinates or market signals (December), the
    ``reference`` table backfills them so the feature layout is always identical.
    """
    df = _ensure_columns(frame, reference)

    # --- dates -------------------------------------------------------------
    date = pd.to_datetime(df["date"], errors="coerce")
    df["day_index"] = (date - EPOCH).dt.days.astype("float64")
    month = date.dt.month
    dow = date.dt.dayofweek
    # Cyclical encodings so December is "next to" January and Sunday next to
    # Monday, instead of being treated as far-apart integers.
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # --- geography ---------------------------------------------------------
    df["haversine_miles"] = _haversine_miles(
        df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"]
    )
    # Circuity = road miles / straight-line miles. Guards against divide-by-zero
    # on the rare zero-distance lane.
    df["circuity"] = df["distance"] / df["haversine_miles"].replace(0.0, np.nan)
    df["circuity"] = df["circuity"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    # --- weight cleaning ---------------------------------------------------
    weight = pd.to_numeric(df["weight"], errors="coerce")
    weight = weight.where((weight >= WEIGHT_FLOOR) & (weight <= WEIGHT_CEIL), other=np.nan)
    df["weight"] = weight  # NaNs kept on purpose; the model handles them.
    df["weight_per_mile"] = df["weight"] / df["distance"].replace(0.0, np.nan)

    # --- passthrough numerics ---------------------------------------------
    df["market_index"] = pd.to_numeric(df["market_index"], errors="coerce")
    df["quote_signal"] = pd.to_numeric(df["quote_signal"], errors="coerce")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")

    # --- lane --------------------------------------------------------------
    # The origin-destination pair is a strong proxy for regional pricing. We
    # treat it as a high-cardinality category and let the gradient booster find
    # the lanes that matter.
    df["lane"] = df["pickup"].astype(str) + " -> " + df["delivery"].astype(str)

    out = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype("category")
    return out
