"""Freight rate prediction pipeline.

Core idea
---------
`posted_rate` is close to multiplicative in its drivers (distance, equipment,
weight, market conditions, lane), so everything is modelled in **log space**:

    log(rate) = beta * t  +  g(structural features, day-of-quarter, ...) + noise

Two properties of the time dimension drive the design, because everything we
have to predict (November and December 2025) sits *after* the last training
date (31 October 2025):

* **A persistent drift.** Net of every other driver, rates ratchet up about
  +2% per quarter. Boosted trees cannot extrapolate a trend - they flat-line
  at the last split point - so `beta * t` is estimated analytically by OLS,
  subtracted from the target before fitting, and added back at prediction
  time.
* **A within-quarter ramp.** Net of the drift, the rate level is roughly flat
  for the first two months of a quarter and then climbs ~4-5% through the
  third month, resetting at the quarter boundary. The shape is the same in
  Q1, Q2 and Q3, so `day_of_quarter` is a feature the model can learn from
  complete quarters and apply to December (day-of-quarter 61-91 of Q4) -
  which is exactly the window the December chart covers.

Data quality handling and feature construction live here too so that the
training run and the two prediction runs share exactly the same code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor

ORIGIN = pd.Timestamp("2025-01-01")
CATEGORICALS = ["equipment", "pickup", "delivery"]
RANDOM_STATE = 42

# |residual| in log space beyond which a posted_rate is treated as corrupted.
# The clean core of the residual distribution ends around 0.16 and the
# contaminated wings start around 0.78, so anything in the gap works.
OUTLIER_LOG_THRESHOLD = 0.4

GBM_PARAMS = dict(
    loss="squared_error",
    max_iter=1500,
    learning_rate=0.04,
    max_leaf_nodes=63,
    min_samples_leaf=20,
    l2_regularization=1.0,
    max_bins=255,
    early_stopping=False,
    random_state=RANDOM_STATE,
)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


@dataclass
class Context:
    """Everything the feature builder needs that is derived from the corpus.

    `market_index` and `quote_signal` are day-level series observed with
    row-level noise (within-day sd is ~0.025 for market_index against a
    between-day sd of ~0.17). Averaging them per calendar day gives a much
    cleaner signal, lets us impute the missing values, and - critically -
    lets us score `december_chart_inputs.csv`, which carries a date but no
    market columns at all.
    """

    daily_market_index: pd.Series
    daily_quote_signal: pd.Series
    city_coords: pd.DataFrame
    equipment_weight_median: pd.Series
    global_weight_median: float

    @classmethod
    def build(cls, frames: list[pd.DataFrame]) -> "Context":
        pool = pd.concat(frames, ignore_index=True)
        daily_mi = pool.groupby("date")["market_index"].mean().sort_index()
        daily_qs = pool.groupby("date")["quote_signal"].mean().sort_index()

        coords = pd.concat(
            [
                pool[["pickup", "pickup_lat", "pickup_lon"]].rename(
                    columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
                ),
                pool[["delivery", "delivery_lat", "delivery_lon"]].rename(
                    columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
                ),
            ]
        ).drop_duplicates("city").set_index("city")

        weight = pool["weight"].abs()
        return cls(
            daily_market_index=daily_mi,
            daily_quote_signal=daily_qs,
            city_coords=coords,
            equipment_weight_median=weight.groupby(pool["equipment"]).median(),
            global_weight_median=float(weight.median()),
        )

    def market_index_for(self, dates: pd.Series) -> pd.Series:
        """Daily market index, extended by weekday-aware nearest-neighbour fill."""
        mapped = dates.map(self.daily_market_index)
        if mapped.isna().any():
            mapped = mapped.fillna(_weekday_fallback(dates, self.daily_market_index))
        return mapped.astype(float)

    def quote_signal_for(self, dates: pd.Series) -> pd.Series:
        mapped = dates.map(self.daily_quote_signal)
        if mapped.isna().any():
            mapped = mapped.fillna(_weekday_fallback(dates, self.daily_quote_signal))
        return mapped.astype(float)


def _weekday_fallback(dates: pd.Series, series: pd.Series) -> pd.Series:
    """Fill unknown dates with the mean of the same weekday in the last 8 weeks."""
    tail = series[series.index >= series.index.max() - pd.Timedelta(days=56)]
    by_dow = tail.groupby(tail.index.dayofweek).mean()
    return dates.dt.dayofweek.map(by_dow)


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean(df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    """Repair the three input-side data-quality defects found during EDA.

    1. `weight` carries sign-flipped values (292 of 48,000 in train are
       negative, down to -47,500 lb). Magnitudes match the healthy
       distribution exactly, so the sign is corruption, not information:
       take the absolute value rather than discarding the row.
    2. `weight` is missing on ~0.6% of rows -> equipment-level median, with a
       flag so the model can price the uncertainty.
    3. `market_index` is missing on ~0.8% of rows -> filled from that day's
       mean, which is the right imputation because the field is a day-level
       series observed with noise.

    Geography is also back-filled from the city lookup so that rows which only
    carry city names (the December chart inputs) can be scored.
    """
    d = df.copy()

    d["weight_missing"] = d["weight"].isna().astype("int8")
    d["weight_sign_flipped"] = (d["weight"] < 0).astype("int8")
    d["weight"] = d["weight"].abs()
    d["weight"] = d["weight"].fillna(d["equipment"].map(ctx.equipment_weight_median))
    d["weight"] = d["weight"].fillna(ctx.global_weight_median)

    if "market_index" not in d:
        d["market_index"] = np.nan
    d["market_index_missing"] = d["market_index"].isna().astype("int8")
    d["market_index"] = d["market_index"].fillna(ctx.market_index_for(d["date"]))

    if "quote_signal" not in d:
        d["quote_signal"] = np.nan
    d["quote_signal"] = d["quote_signal"].fillna(ctx.quote_signal_for(d["date"]))

    for side, prefix in (("pickup", "pickup"), ("delivery", "delivery")):
        for axis in ("lat", "lon"):
            col = f"{prefix}_{axis}"
            filled = d[side].map(ctx.city_coords[axis])
            d[col] = filled if col not in d else d[col].fillna(filled)

    return d


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def day_of_quarter(dates: pd.Series) -> pd.Series:
    return (dates - dates.dt.to_period("Q").dt.start_time).dt.days.astype("int16")


def build_features(d: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    f = pd.DataFrame(index=d.index)

    # Rate scales with distance at an elasticity near 0.87, i.e. long hauls
    # earn a lower rate per mile. Log distance makes that relationship linear.
    f["log_distance"] = np.log(d["distance"])
    f["distance"] = d["distance"]

    f["weight"] = d["weight"]
    f["weight_missing"] = d["weight_missing"]
    f["weight_per_mile"] = d["weight"] / d["distance"]

    # Market state: the row-level reading plus the (much less noisy) daily mean.
    f["market_index"] = d["market_index"]
    f["market_index_daily"] = ctx.market_index_for(d["date"]).values
    f["quote_signal"] = d["quote_signal"]
    f["quote_signal_daily"] = ctx.quote_signal_for(d["date"]).values

    # Geography as coordinates, not just as city labels: 12% of the validation
    # loads touch a city that never appears in training, and the lane premium
    # is spatially smooth (Texas origins price high, Great Lakes origins low),
    # so a model that reads latitude/longitude generalises to unseen cities.
    f["pickup_lat"] = d["pickup_lat"]
    f["pickup_lon"] = d["pickup_lon"]
    f["delivery_lat"] = d["delivery_lat"]
    f["delivery_lon"] = d["delivery_lon"]
    f["mid_lat"] = (d["pickup_lat"] + d["delivery_lat"]) / 2
    f["mid_lon"] = (d["pickup_lon"] + d["delivery_lon"]) / 2
    f["delta_lat"] = d["delivery_lat"] - d["pickup_lat"]
    f["delta_lon"] = d["delivery_lon"] - d["pickup_lon"]

    # Calendar. The absolute date `t` is deliberately absent: the drift is
    # handled analytically (see fit_drift) so that predictions past the last
    # training date keep moving instead of flat-lining at the boundary.
    # `day_of_quarter` carries the repeating within-quarter ramp, which is the
    # only calendar shape that is both large and safely learnable - three
    # complete quarters of training data cover all 92 positions.
    f["day_of_quarter"] = day_of_quarter(d["date"])
    f["day_of_week"] = d["date"].dt.dayofweek.astype("int16")
    f["is_weekend"] = d["date"].dt.dayofweek.isin([5, 6]).astype("int8")

    for col in CATEGORICALS:
        f[col] = d[col]
    return f


def align_categories(train: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    out = other.copy()
    for col in CATEGORICALS:
        out[col] = pd.Categorical(out[col], categories=train[col].cat.categories)
    return out


def as_categorical(f: pd.DataFrame) -> pd.DataFrame:
    out = f.copy()
    for col in CATEGORICALS:
        out[col] = out[col].astype("category")
    return out


# --------------------------------------------------------------------------
# Label quality
# --------------------------------------------------------------------------
def flag_corrupted_labels(d: pd.DataFrame) -> np.ndarray:
    """Return a boolean mask of rows whose `posted_rate` looks trustworthy.

    A Huber regression of log(rate) on the structural drivers leaves a residual
    distribution with a tight core (|r| < 0.16) and two detached wings starting
    beyond |r| ~ 0.8 - roughly 1.4% of rows priced 2-6x too high or too low.
    They are label noise, not a lane effect: the same lane/day/equipment
    combination has healthy neighbours. Fitting on them drags the model, so
    they are excluded from training (never from prediction).
    """
    y = np.log(d["posted_rate"])
    X = pd.get_dummies(d[["equipment"]], drop_first=True).astype(float)
    X["log_distance"] = np.log(d["distance"])
    X["weight"] = d["weight"]
    X["market_index"] = d["market_index"]
    X["quote_signal"] = d["quote_signal"]
    X["t"] = (d["date"] - ORIGIN).dt.days
    X = sm.add_constant(X)
    fit = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    return ((y - fit.predict(X)).abs() < OUTLIER_LOG_THRESHOLD).to_numpy()


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def fit_drift(d: pd.DataFrame) -> float:
    """OLS estimate of the daily log-rate drift, net of every other driver.

    The within-quarter ramp is absorbed by day-of-quarter dummies so that the
    `t` coefficient measures only the quarter-over-quarter ratchet; leaving
    them out would blend the two and over-state the drift. Controlling as well
    for distance, weight, equipment, market index, quote signal and
    origin/destination fixed effects gives ~+2.1e-4 per day (~+8%/year), which
    is what gets extrapolated into November and December.
    """
    y = np.log(d["posted_rate"])
    frame = d[["equipment", "pickup", "delivery"]].copy()
    frame["doq"] = day_of_quarter(d["date"])
    X = pd.get_dummies(frame, columns=list(frame.columns), drop_first=True).astype(float)
    X["log_distance"] = np.log(d["distance"])
    X["log_distance_sq"] = X["log_distance"] ** 2
    X["weight"] = d["weight"]
    X["market_index"] = d["market_index"]
    X["quote_signal"] = d["quote_signal"]
    X["t"] = (d["date"] - ORIGIN).dt.days
    X = sm.add_constant(X)
    return float(sm.OLS(y, X).fit().params["t"])


@dataclass
class RateModel:
    ctx: Context
    drift: float
    gbm: HistGradientBoostingRegressor
    smearing: float
    train_columns: pd.DataFrame  # kept for category alignment

    def predict(self, raw: pd.DataFrame) -> np.ndarray:
        d = clean(raw, self.ctx)
        X = align_categories(self.train_columns, build_features(d, self.ctx))
        t = (d["date"] - ORIGIN).dt.days.to_numpy()
        log_rate = self.gbm.predict(X) + self.drift * t
        # Duan smearing: E[rate] != exp(E[log rate]); the factor is the mean
        # of exp(training residuals) and corrects the retransformation bias.
        return np.exp(log_rate) * self.smearing


def train(raw: pd.DataFrame, ctx: Context, drop_corrupted: bool = True, **gbm_kw) -> RateModel:
    d = clean(raw, ctx)
    if drop_corrupted:
        d = d[flag_corrupted_labels(d)].reset_index(drop=True)

    drift = fit_drift(d)
    t = (d["date"] - ORIGIN).dt.days.to_numpy()
    y = np.log(d["posted_rate"].to_numpy()) - drift * t

    X = as_categorical(build_features(d, ctx))
    params = {**GBM_PARAMS, **gbm_kw, "categorical_features": CATEGORICALS}
    gbm = HistGradientBoostingRegressor(**params).fit(X, y)

    smearing = float(np.mean(np.exp(y - gbm.predict(X))))
    return RateModel(ctx=ctx, drift=drift, gbm=gbm, smearing=smearing, train_columns=X)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    err = predicted - actual
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_%": float(np.mean(np.abs(err) / actual) * 100),
        "MedAPE_%": float(np.median(np.abs(err) / actual) * 100),
        "R2": 1 - ss_res / ss_tot,
    }
