# Loom talking points (2–3 minutes)

Rough script for the walkthrough video. Timings are a guide, not a target.

---

### 0:00 — What the data is (15s)

48,000 labelled loads, January to October 2025. I have to price 12,000 loads in
November and December — so every prediction is a forecast forward in time. That
one fact drove most of the decisions.

### 0:15 — Key findings from exploring the data (40s)

Rate is basically multiplicative, so I work in log space. Distance elasticity is
about 0.87 — rate per mile falls from $2.80 on short hauls to $1.87 on long ones.

Two time effects, and this is the interesting part. First, a steady drift: rates
ratchet up about 2% per quarter net of everything else. Second — and I only found
this by plotting the rate level after stripping out market_index and structure —
there's a **repeating within-quarter ramp**. Rates are flat for two months of a
quarter, then climb 4–5% through the third month, and reset at the quarter
boundary. The same shape in Q1, Q2 and Q3.

That matters because December is the third month of Q4. It's the biggest single
driver of the December chart.

And `market_index` is a day-level series, not a row-level one: within-day sd is
0.025 against a between-day sd of 0.17. So I average it per day — which also
lets me score the December chart inputs, which have a date but no market columns.

### 0:55 — Data quality issues (35s)

Four, three on the input side and one on the label side.

`weight` had 292 sign-flipped negatives — the magnitudes match the healthy
distribution exactly, so I take the absolute value rather than dropping the rows.
300 missing weights get an equipment-level median plus a flag. 374 missing
`market_index` values get that day's mean.

The label one is the expensive one. A robust regression leaves a residual
distribution with a tight core and two completely detached wings — 677 loads,
1.4%, priced 2 to 6× off, with healthy neighbours on the same lane and day.
There's a clean gap between the core and the wings, so the threshold isn't a
judgement call. I exclude them from training. Leaving them in nearly triples
holdout error — 1.4% to 4% MAPE.

### 1:30 — Model choice and how time is handled (35s)

HistGradientBoosting on the log target. But the important part isn't the
estimator — it's that **trees can't extrapolate a trend**. Past the last split
point they return a constant, so a plain GBM would flat-line through November and
December.

So I do two things. The drift comes out analytically: OLS with day-of-quarter
dummies and full structural controls, subtracted from the target before fitting
and added back at prediction time. And I give the model `day_of_quarter` instead
of the absolute date — a feature that repeats, learnable from three complete
quarters and applicable to December's day-of-quarter 61 to 91.

Ablation: feeding the raw date to the trees instead costs 1.394% → 1.874% MAPE.

### 2:05 — Validation and split (25s)

Forward holdouts, not random splits. A random split puts loads from the same day
on both sides, which leaks that day's market conditions — a leak the real task
doesn't have.

Trained on January–August, scored on September–October: 1.394% MAPE, R² 0.99895.
I ran random 5-fold CV alongside to price the leak — 1.236%. The gap is the
entire cost of forecasting forward, which is small, which is the evidence the
time handling works.

1.4% is essentially the noise floor: residual sd is about 1.8% in log space once
lane, day and equipment effects are removed.

### 2:30 — Code walkthrough (25s)

`src/pipeline.py` is the whole model — three functions worth looking at.
`flag_corrupted_labels` is the Huber-residual filter. `fit_drift` is the OLS that
estimates the trend. `train` puts it together: clean, drop corrupted labels,
detrend, fit, and store the smearing factor for the transform back.

`run.py` trains on everything and writes both output files. `evaluate.py`
reproduces every number I just quoted.

One detail worth calling out: the December chart inputs have no market columns
and no coordinates, and they go through **exactly the same** `clean` → `features`
path as everything else. The daily market series and the city coordinate lookup
fill them in, so the only thing varying across the 31 rows is the date. That's
what makes the chart honest.
