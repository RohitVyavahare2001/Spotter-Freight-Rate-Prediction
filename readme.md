# Freight Rate Prediction Challenge

A model that predicts the posted rate (in USD) for a truckload freight shipment
from its lane, distance, equipment type, weight, date and market signals.

See `freight-rate-ml-assessment-1-.pdf` for the original assessment instructions.

## Results (temporal holdout)

The model is evaluated on the most recent 20% of the training window by date
(Sep–Oct 2025), which imitates the real task of predicting the future Nov–Dec
validation period.

| Metric | Value |
| ------ | ----- |
| MAPE   | 5.7%  |
| MAE    | $133  |
| RMSE   | $641  |
| R²     | 0.82  |

Full numbers and feature importances are written to
`artifacts/training_report.json` by the training run.

## Project layout

```
.
├── data/                         # datasets (copied into a data/ folder for the scorer)
│   ├── train_test.csv            # 48,000 labeled development rows
│   ├── validation.csv            # 12,000 loads needing final predictions
│   ├── validation_predictions_template.csv
│   └── december_chart_inputs.csv # 31 fixed December rows (filled by src.predict)
├── notebooks/
│   └── eda.ipynb                 # exploratory analysis walkthrough (run, with charts)
├── src/
│   ├── features.py               # shared feature engineering + December backfill
│   ├── train.py                  # temporal split, train, refit, save model
│   ├── predict.py                # write validation_predictions.csv + December rates
│   └── eda.py                    # exploratory figures used in the report
├── artifacts/                    # model.txt, feature_reference.json, training_report.json
├── report/                       # written report + figures
├── score.py                      # provided scorer (unmodified)
├── validation_predictions.csv    # <- deliverable, 12,000 rows
└── requirements.txt
```

## Setup

```bash
python -m pip install -r requirements.txt
```

Tested with Python 3.11+.

## Reproduce the full pipeline

```bash
# 1. (optional) regenerate EDA figures
python -m src.eda

# 2. train + validate the model (writes artifacts/model.txt)
python -m src.train

# 3. write validation_predictions.csv and fill the December chart inputs
python -m src.predict

# 4. run the provided scorer
python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs.csv
```

Step 4 validates both files and writes `scorer_results/candidate_december.png`.

## Exploratory analysis

`notebooks/eda.ipynb` is the full EDA walkthrough — target distribution,
rate drivers, equipment and lane effects, time trends, the data-quality audit
and the reasoning behind the temporal validation split. It is committed with its
outputs and charts already rendered, so it can be read top-to-bottom on GitHub
without rerunning. To re-execute:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb
# or just open it: jupyter lab notebooks/eda.ipynb
```

## Approach in one paragraph

Rates are modelled as `log(posted_rate)` with a LightGBM gradient-boosted tree.
Distance is the dominant driver (correlation 0.91); the origin→destination lane,
equipment type, weight, market signals and smooth cyclical date features round
out the model. Data-quality issues — impossible negative/zero weights, missing
weights and missing market indices — are cleaned in `src/features.py` so the
same deterministic pipeline runs on training, validation and the December rows.
The December file ships with only seven columns, so a small reference table of
per-city coordinates and median market signals (learned at train time and saved
to `artifacts/feature_reference.json`) backfills the columns it lacks. See
`report/report.md` for the full write-up.
