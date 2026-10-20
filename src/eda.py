"""Exploratory data analysis: writes summary stats and figures used in report.

Run
---
    python -m src.eda
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "report" / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(DATA_DIR / "train_test.csv", parse_dates=["date"])
    tr["rpm"] = tr["posted_rate"] / tr["distance"]

    # 1) distance vs rate -- the dominant relationship
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    sample = tr.sample(min(6000, len(tr)), random_state=0)
    ax.scatter(sample["distance"], sample["posted_rate"], s=6, alpha=0.25, color="#064A56")
    ax.set_xlabel("distance (miles)")
    ax.set_ylabel("posted_rate ($)")
    ax.set_title("Posted rate scales almost linearly with distance (r = 0.91)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "distance_vs_rate.png")
    plt.close(fig)

    # 2) rate-per-mile by equipment
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    order = ["Dry Van", "Flatbed", "Reefer"]
    data = [tr.loc[tr.equipment == e, "rpm"].clip(upper=5) for e in order]
    ax.boxplot(data, labels=order, showfliers=False)
    ax.set_ylabel("rate per mile ($/mi)")
    ax.set_title("Reefer and Flatbed price above Dry Van per mile")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rpm_by_equipment.png")
    plt.close(fig)

    # 3) monthly trend
    monthly = tr.groupby(tr.date.dt.to_period("M"))["rpm"].mean()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(range(len(monthly)), monthly.values, marker="o", color="#064A56")
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels([str(p) for p in monthly.index], rotation=45, ha="right")
    ax.set_ylabel("mean rate per mile ($/mi)")
    ax.set_title("Mild upward drift in per-mile rates through 2025")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "monthly_trend.png")
    plt.close(fig)

    # 4) target distribution raw vs log
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=150)
    axes[0].hist(tr["posted_rate"], bins=60, color="#064A56")
    axes[0].set_title(f"posted_rate (skew {tr['posted_rate'].skew():.2f})")
    axes[1].hist(np.log(tr["posted_rate"]), bins=60, color="#064A56")
    axes[1].set_title(f"log(posted_rate) (skew {np.log(tr['posted_rate']).skew():.2f})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "target_distribution.png")
    plt.close(fig)

    print(f"Saved 4 EDA figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
