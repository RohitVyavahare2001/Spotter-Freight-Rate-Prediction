"""Render the technical report as a submittable DOCX with embedded figures.

Run
---
    python -m src.build_report
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
OUT = ROOT / "report" / "Freight_Rate_Prediction_Report.docx"

ACCENT = RGBColor(0x06, 0x4A, 0x56)


def h(doc: Document, text: str, level: int) -> None:
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.color.rgb = ACCENT


def fig(doc: Document, name: str, caption: str, width: float = 6.0) -> None:
    path = FIG / name
    if path.is_file():
        doc.add_picture(str(path), width=Inches(width))
        cap = doc.add_paragraph(caption)
        cap.runs[0].italic = True
        cap.runs[0].font.size = Pt(9)


def table(doc: Document, header: list[str], rows: list[list[str]]) -> None:
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for i, head in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = head
        cell.paragraphs[0].runs[0].bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = val


def main() -> None:
    doc = Document()
    doc.add_heading("Freight Rate Prediction — Technical Report", level=0)
    sub = doc.add_paragraph("Machine Learning Engineer Assessment")
    sub.runs[0].italic = True

    h(doc, "1. Data overview", 1)
    doc.add_paragraph(
        "The labeled development set (data/train_test.csv) contains 48,000 loads "
        "spanning 1 Jan 2025 to 31 Oct 2025. The validation set holds 12,000 loads "
        "from 1 Nov 2025 to 31 Dec 2025 — strictly after the training window. That "
        "single fact drives the entire validation strategy."
    )
    table(
        doc,
        ["Column", "Meaning"],
        [
            ["pickup / delivery", "origin / destination city (64 each)"],
            ["*_lat / *_lon", "coordinates of each city"],
            ["distance", "billed road miles"],
            ["equipment", "Dry Van 57%, Reefer 25%, Flatbed 18%"],
            ["weight", "shipment weight (lb)"],
            ["date", "pickup date"],
            ["market_index, quote_signal", "market / pricing signals"],
            ["posted_rate", "TARGET — rate paid for the load (USD)"],
        ],
    )

    h(doc, "2. Key findings from exploration", 1)
    for line in [
        "Distance is the dominant driver: correlation with posted_rate is 0.91 and "
        "the relationship is near-linear.",
        "Equipment carries a clear premium: mean rate-per-mile is $2.12 Dry Van < "
        "$2.29 Flatbed < $2.38 Reefer.",
        "Lane (origin to destination) is a strong regional-pricing proxy and ends up "
        "the second most important feature.",
        "Mild upward time drift through 2025 plus a small mid-week bump, captured with "
        "a linear day index and cyclical month / day-of-week features.",
        "The target is right-skewed (skew ~1.9); a log transform nearly symmetrises it "
        "(skew ~-0.5), so the model trains on log(posted_rate).",
    ]:
        doc.add_paragraph(line, style="List Bullet")
    fig(doc, "distance_vs_rate.png", "Figure 1. Posted rate scales almost linearly with distance.")
    fig(doc, "rpm_by_equipment.png", "Figure 2. Reefer and Flatbed price above Dry Van per mile.")

    h(doc, "3. Data-quality issues and handling", 1)
    table(
        doc,
        ["Issue", "Evidence", "Handling"],
        [
            ["Impossible weights", "292 rows weight <= 0, min -47,500 lb",
             "Clipped to 1,000-48,000 lb; out-of-band set to missing."],
            ["Missing weights", "300 train / 165 val NaN",
             "Left as NaN; LightGBM handles missing natively, no rows dropped."],
            ["Missing market index", "374 train / 249 val NaN",
             "Left NaN for model; median-backfilled for the December file."],
            ["Road vs straight-line distance", "distance ~1.17x haversine",
             "Kept billed distance; added haversine + circuity as secondary signals."],
            ["December file has only 7 columns", "no coords / signals",
             "Reference table of per-city coords + median signals backfills them."],
        ],
    )
    doc.add_paragraph(
        "All cleaning lives in src/features.py and runs identically for training, "
        "validation and December — there is no train/serve skew."
    )
    fig(doc, "target_distribution.png", "Figure 3. Target distribution: raw (skewed) vs log.", 6.5)

    h(doc, "4. Train / validation split", 1)
    doc.add_paragraph(
        "Because the real validation period is the future relative to training, a "
        "random shuffle split would leak future information and overstate performance. "
        "Instead a temporal forward split is used:"
    )
    for line in [
        "Sort the 48,000 rows by date.",
        "Hold out the most recent 20% by date (Sep-Oct 2025); train on the earlier 80% "
        "(Jan-Aug 2025).",
        "This mirrors the task: fit on the past, score on a later unseen window. Early "
        "stopping chose 172 boosting rounds.",
        "After measuring, the model is refit on all 48,000 rows with that round count, "
        "so it has seen every date closest to the Nov-Dec target period.",
    ]:
        doc.add_paragraph(line, style="List Bullet")
    doc.add_paragraph("Holdout performance (Sep-Oct 2025):")
    table(
        doc,
        ["Metric", "Value", "Reading"],
        [
            ["MAPE", "5.7%", "within ~6% of actual on average"],
            ["MAE", "$133", "typical dollar error"],
            ["RMSE", "$641", "penalises rare high-rate long hauls"],
            ["R2", "0.82", "explains 82% of variance on unseen future dates"],
        ],
    )

    h(doc, "5. Model choice and reasoning", 1)
    doc.add_paragraph(
        "LightGBM gradient-boosted regression on log(posted_rate). The strongest "
        "relationship (distance to rate) is near-linear, but equipment, lane and market "
        "signals introduce non-linear interactions a booster captures without manual "
        "terms. It handles mixed numeric + high-cardinality categorical features "
        "(64x64 lanes) and missing values natively. Training on the log makes the loss "
        "relative — appropriate for a multiplicative price — and guarantees strictly "
        "positive predictions, which the scorer requires. Feature importance ranks "
        "distance > lane > quote_signal > weight > equipment > date features."
    )

    h(doc, "6. The fixed December chart", 1)
    doc.add_paragraph(
        "Running the provided score.py on the filled December inputs produces the chart "
        "below. Inputs are fixed (Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb) "
        "with only the date changing, isolating the model's date/day-of-week response. "
        "Predictions sit around $848 (~$2.36/mile, consistent with training), with a "
        "small weekly oscillation from the day-of-week features."
    )
    fig(doc, "candidate_december.png", "Figure 4. Candidate December 2025 predicted load rate.", 6.5)

    h(doc, "7. Reproducing", 1)
    code = doc.add_paragraph(
        "python -m pip install -r requirements.txt\n"
        "python -m src.train\n"
        "python -m src.predict\n"
        "python score.py --predictions validation_predictions.csv \\\n"
        "    --december-predictions data/december_chart_inputs.csv"
    )
    code.runs[0].font.name = "Consolas"
    code.runs[0].font.size = Pt(9)

    doc.save(OUT)
    print(f"Saved report -> {OUT}")


if __name__ == "__main__":
    main()
