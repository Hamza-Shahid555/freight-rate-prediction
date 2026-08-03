"""Train on the full development set and produce both deliverables.

Outputs
-------
validation_predictions.csv          load_id,predicted_rate for all 12,000 loads
data/december_chart_inputs.csv      the provided template with predicted_rate filled

Then run the provided scorer to validate both files and draw the chart:

    python score.py --predictions validation_predictions.csv \
                    --december-predictions data/december_chart_inputs.csv

Usage:  python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from pipeline import Context, clean, flag_corrupted_labels, load_csv, train  # noqa: E402

DATA = ROOT / "data"
DECEMBER_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]


def main() -> None:
    train_test = load_csv(DATA / "train_test.csv")
    validation = load_csv(DATA / "validation.csv")
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    december = pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])

    # The context (daily market series, city coordinates, weight medians) is
    # built from features only - `posted_rate` is dropped so no label from the
    # development set can leak into it.
    ctx = Context.build([train_test.drop(columns=["posted_rate"]), validation])

    dropped = int((~flag_corrupted_labels(clean(train_test, ctx))).sum())
    print(f"development rows            {len(train_test):,}")
    print(f"corrupted labels excluded   {dropped:,} ({dropped / len(train_test):.2%})")

    model = train(train_test, ctx)
    print(f"log-rate drift              {model.drift:+.6f}/day "
          f"({np.exp(model.drift * 365) - 1:+.1%} per year)")
    print(f"smearing factor             {model.smearing:.5f}")

    # ---- 1. the 12,000 validation loads -----------------------------------
    predictions = pd.DataFrame({
        "load_id": validation["load_id"],
        "predicted_rate": np.round(model.predict(validation), 2),
    })
    # Preserve the template's row order exactly.
    predictions = (template[["load_id"]]
                   .merge(predictions, on="load_id", how="left")
                   .loc[:, ["load_id", "predicted_rate"]])
    assert predictions["predicted_rate"].notna().all(), "missing predictions"
    assert (predictions["predicted_rate"] > 0).all(), "non-positive predictions"
    out = ROOT / "validation_predictions.csv"
    predictions.to_csv(out, index=False)
    print(f"\nwrote {out.name}  ({len(predictions):,} rows, "
          f"mean ${predictions.predicted_rate.mean():,.0f}, "
          f"median ${predictions.predicted_rate.median():,.0f})")

    # ---- 2. the fixed-input December chart --------------------------------
    # These 31 rows carry no market_index / quote_signal / coordinates. The
    # pipeline fills them from the day-level market series and the city
    # lookup, so only the date varies - which is the point of the chart.
    december["predicted_rate"] = np.round(model.predict(december), 2)
    december["date"] = december["date"].dt.strftime("%Y-%m-%d")
    december[DECEMBER_COLUMNS].to_csv(DATA / "december_chart_inputs.csv", index=False)
    print(f"wrote data/december_chart_inputs.csv  "
          f"(${december.predicted_rate.min():,.0f} - ${december.predicted_rate.max():,.0f})")


if __name__ == "__main__":
    main()
