"""Validation study for the freight rate model.

Two complementary designs are reported:

* **Forward (time-based) holdouts** - train on everything before a cut-off,
  score the months after it. This is the design that matters, because the real
  task is to price November and December 2025 from data that stops on
  31 October 2025. A random split would leak same-day market conditions into
  the training set and flatter the model.
* **Random 5-fold CV** - reported only as an upper bound, to show how much of
  the forward-holdout error is the cost of extrapolating in time rather than
  model error.

Ablations quantify the two decisions that are easy to get wrong: excluding the
corrupted `posted_rate` labels, and modelling the time drift explicitly instead
of letting the boosted trees flat-line past the last training date.

Usage:  python evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pipeline import (  # noqa: E402
    ORIGIN,
    Context,
    align_categories,
    as_categorical,
    build_features,
    clean,
    fit_drift,
    flag_corrupted_labels,
    load_csv,
    metrics,
    train,
)

DATA = Path(__file__).parent / "data"
CUTS = ["2025-07-01", "2025-08-01", "2025-09-01", "2025-10-01"]


def row(label: str, m: dict[str, float], n: int) -> str:
    return (f"{label:<34}{n:>8,}  {m['MAE']:>8.2f}  {m['RMSE']:>8.2f}"
            f"  {m['MAPE_%']:>7.3f}  {m['MedAPE_%']:>8.3f}  {m['R2']:>8.5f}")


HEADER = (f"{'':<34}{'n':>8}  {'MAE':>8}  {'RMSE':>8}  {'MAPE%':>7}  {'MedAPE%':>8}  {'R2':>8}")


def main() -> None:
    train_test = load_csv(DATA / "train_test.csv")
    validation = load_csv(DATA / "validation.csv")
    ctx = Context.build([train_test.drop(columns=["posted_rate"]), validation])

    # ---------------- data quality summary ----------------
    print("=" * 96)
    print("DATA QUALITY")
    print("=" * 96)
    n = len(train_test)
    print(f"rows                                 {n:,}")
    print(f"date range                           {train_test.date.min():%Y-%m-%d} -> "
          f"{train_test.date.max():%Y-%m-%d}")
    print(f"negative weight (sign-flipped)       {(train_test.weight < 0).sum():,}")
    print(f"missing weight                       {train_test.weight.isna().sum():,}")
    print(f"missing market_index                 {train_test.market_index.isna().sum():,}")
    print(f"duplicate load_id                    {train_test.load_id.duplicated().sum():,}")
    cleaned = clean(train_test, ctx)
    keep = flag_corrupted_labels(cleaned)
    print(f"corrupted posted_rate labels         {(~keep).sum():,} ({(~keep).mean():.2%})")
    unseen = ~validation.pickup.isin(train_test.pickup) | ~validation.delivery.isin(train_test.delivery)
    print(f"validation rows w/ unseen city       {unseen.sum():,} ({unseen.mean():.1%})")
    drift = fit_drift(cleaned[keep])
    print(f"estimated log-rate drift             {drift:+.6f}/day  "
          f"({np.exp(drift * 365) - 1:+.1%} per year)")

    # ---------------- forward holdouts ----------------
    print()
    print("=" * 96)
    print("FORWARD (TIME-BASED) HOLDOUTS  -  train < cut, test >= cut")
    print("=" * 96)
    print(HEADER)
    for cut in CUTS:
        cut_ts = pd.Timestamp(cut)
        tr = train_test[train_test.date < cut_ts]
        te = train_test[train_test.date >= cut_ts]
        model = train(tr, ctx)
        te_clean = clean(te, ctx)
        honest = flag_corrupted_labels(te_clean)  # corrupted labels are unpredictable
        pred = model.predict(te)
        m = metrics(te.posted_rate.to_numpy()[honest], pred[honest])
        print(row(f"holdout >= {cut}", m, int(honest.sum())))

    # ---------------- random CV (upper bound) ----------------
    print()
    print("=" * 96)
    print("RANDOM 5-FOLD CV  (optimistic reference - leaks same-day market state)")
    print("=" * 96)
    print(HEADER)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for i, (a, b) in enumerate(kf.split(train_test), 1):
        model = train(train_test.iloc[a], ctx)
        te = train_test.iloc[b]
        honest = flag_corrupted_labels(clean(te, ctx))
        pred = model.predict(te)
        m = metrics(te.posted_rate.to_numpy()[honest], pred[honest])
        scores.append(m)
        print(row(f"fold {i}", m, int(honest.sum())))
    mean = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
    print(row("mean", mean, len(train_test)))

    # ---------------- ablations on the Sep-Oct holdout ----------------
    print()
    print("=" * 96)
    print("ABLATIONS  -  holdout = September + October 2025")
    print("=" * 96)
    print(HEADER)
    cut_ts = pd.Timestamp("2025-09-01")
    tr = train_test[train_test.date < cut_ts]
    te = train_test[train_test.date >= cut_ts]
    honest = flag_corrupted_labels(clean(te, ctx))
    actual = te.posted_rate.to_numpy()[honest]

    def report(label, pred):
        print(row(label, metrics(actual, pred[honest]), int(honest.sum())))

    report("full model", train(tr, ctx).predict(te))
    report("keep corrupted labels", train(tr, ctx, drop_corrupted=False).predict(te))

    # drift removed: trees see `t` directly and cannot extrapolate it
    tr_clean = clean(tr, ctx)
    tr_clean = tr_clean[flag_corrupted_labels(tr_clean)]
    Xtr = as_categorical(build_features(tr_clean, ctx))
    Xtr["t"] = (tr_clean.date - ORIGIN).dt.days.to_numpy()
    ytr = np.log(tr_clean.posted_rate.to_numpy())
    from pipeline import GBM_PARAMS, CATEGORICALS  # noqa: E402
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

    g = HistGradientBoostingRegressor(**{**GBM_PARAMS, "categorical_features": CATEGORICALS})
    g.fit(Xtr, ytr)
    sm_ = float(np.mean(np.exp(ytr - g.predict(Xtr))))
    te_clean = clean(te, ctx)
    Xte = align_categories(Xtr, build_features(te_clean, ctx))
    Xte["t"] = (te_clean.date - ORIGIN).dt.days.to_numpy()
    report("date as a tree feature (no drift)", np.exp(g.predict(Xte)) * sm_)

    # no smearing correction
    model = train(tr, ctx)
    report("no smearing correction", model.predict(te) / model.smearing)

    print()
    print("Chosen configuration: full model (drift removed analytically, corrupted")
    print("labels excluded, log target, Duan smearing on retransformation).")


if __name__ == "__main__":
    main()
