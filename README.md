# Freight Rate Prediction

Predicts `posted_rate` for freight loads. Trained on `data/train_test.csv`
(48,000 labelled loads, Jan–Oct 2025) and scored on `data/validation.csv`
(12,000 loads, Nov–Dec 2025).

## Setup

```bash
python -m pip install -r requirements.txt
```

Python 3.10+.

## Run

```bash
python run.py
```

Trains on the full development set and writes:

| file | contents |
| --- | --- |
| `validation_predictions.csv` | `load_id,predicted_rate` for all 12,000 validation loads |
| `data/december_chart_inputs.csv` | the provided template with `predicted_rate` filled |

Then validate both files and draw the chart with the provided scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

This writes `scorer_results/candidate_december.png`.

To reproduce the validation study (forward holdouts, random CV, ablations):

```bash
python evaluate.py
```

Its output is saved in `report/validation_report.txt`.

## Layout

```
run.py                  train on everything, write both prediction files
evaluate.py             validation study: time-based holdouts, CV, ablations
score.py                provided scorer (unmodified)
src/pipeline.py         cleaning, features, model
data/                   inputs
report/                 written report + raw validation output
scorer_results/         chart produced by score.py
```

## Approach

**Target.** Rates are multiplicative in their drivers, so the model predicts
`log(posted_rate)` and converts back with a Duan smearing factor (the naive
`exp()` of a log-space prediction is biased low for the mean). The factor comes
out at 1.00003 here — the residual variance is small enough that it changes
nothing — but it is kept because it is the correct retransformation.

**Data quality.** Three defects were found and repaired, plus one on the label
side:

| issue | rows (train) | handling |
| --- | --- | --- |
| `weight` stored with a flipped sign | 292 | take the absolute value — magnitudes match the healthy distribution exactly |
| `weight` missing | 300 | equipment-level median + a missingness flag |
| `market_index` missing | 374 | that day's mean — the field is a day-level series observed with row noise |
| `posted_rate` corrupted | 677 (1.41%) | excluded from training only |

The corrupted labels are unambiguous. A robust regression of `log(rate)` on the
structural drivers leaves a residual distribution with a tight core (`|r| < 0.16`)
and two detached wings starting past `|r| ≈ 0.8` — loads priced 2–6× too high or
too low with healthy neighbours on the same lane, day and equipment type. They
are excluded from training and from holdout scoring (they are unpredictable by
construction), never from the submitted predictions.

**Time.** Everything that has to be predicted sits after the last training date,
so the two time effects are handled explicitly rather than left to the trees:

* a **drift** of about +2% per quarter (+7.4%/year), estimated by OLS with full
  structural controls, subtracted from the target before fitting and added back
  at prediction time — boosted trees flat-line past their last split point and
  cannot extrapolate a trend;
* a **within-quarter ramp** — net of the drift, rates are flat for two months
  and then climb 4–5% through the third month, resetting at the quarter
  boundary, with the same shape in Q1, Q2 and Q3. This enters as a
  `day_of_quarter` feature, learnable from three complete quarters and
  applicable to December (day-of-quarter 61–91).

**Geography.** 12% of validation loads touch a city that never appears in
training, so lanes are given to the model as coordinates as well as labels.
The origin premium is spatially smooth — Texas origins price high, Great Lakes
origins low — which lets latitude/longitude generalise to unseen cities.

**Model.** `HistGradientBoostingRegressor` on the detrended log target, with
native categorical support for equipment, origin and destination.

**Validation.** Forward (time-based) holdouts — train before a cut-off, score
the months after it — because that is the shape of the real task. Random 5-fold
CV is reported alongside only as an optimistic reference; it leaks same-day
market conditions into training and flatters the model.

Held-out accuracy on September–October 2025 (trained on January–August):

| | MAE | RMSE | MAPE | R² |
| --- | --- | --- | --- | --- |
| forward holdout, Sep–Oct | $30.97 | $44.54 | 1.394% | 0.99895 |
| random 5-fold CV (reference) | $27.51 | $40.22 | 1.236% | 0.99912 |

The small gap between the two is the whole cost of predicting forward in time.
Both sit close to the irreducible noise floor of the data (residual sd ≈ 1.8%
in log space once lane, day and equipment effects are removed).

Full numbers, including ablations for each design decision, are in
`report/validation_report.txt`.
