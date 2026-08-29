"""
Module 02: Data Processing (generic — works on any date+value dataset)
------------------------------------------------------------------------
Unlike Module 1 (stock-specific), this module knows nothing about stocks.
It takes ANY uploaded CSV/Excel with a date column and one or more value
columns, and prepares it for forecasting:

  1. Ingest the uploaded file (CSV or Excel)
  2. Detect the schema (which column is the date, which columns are values)
  3. Infer the data's native/current frequency (hourly, daily, weekly, ...)
  4. Resample to a user-chosen target frequency, so every time step is
     evenly spaced:
       - Downsampling (e.g. daily -> weekly): aggregate multiple original
         rows into each new step (mean/sum/last/etc., user's choice).
       - Upsampling (e.g. weekly -> daily): create new time steps and
         fill their values via interpolation.
     Both directions are handled by the same resample+interpolate logic,
     so the same function works regardless of which way the user is going.

Follows the project's three-layer pattern:
  - detect_schema() / infer_native_frequency()  -> detection (pure functions)
  - (question of which column / which frequency is asked in app.py, and
     ONLY when detection is genuinely ambiguous)
  - resample_to_frequency()                      -> apply, returns
                                                     (df, actions_log)
"""

import logging
from io import BytesIO

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("data_processing")

FREQ_LABEL_TO_CODE = {"Hourly": "h", "Daily": "D", "Weekly": "W", "Monthly": "MS", "Yearly": "YS"}

# Aggregation functions offered for downsampling (e.g. daily -> weekly).
AGG_METHODS = ["mean", "sum", "last", "first", "min", "max"]

# Interpolation methods offered for upsampling / filling newly-created
# time steps (e.g. weekly -> daily). "pad" = forward-fill.
INTERP_METHODS = ["linear", "time", "nearest", "pad"]


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Read an uploaded CSV or Excel file (as given by Streamlit's
    file_uploader) into a DataFrame. Does not assume any particular
    schema yet -- that's detect_schema()'s job.
    """
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()
    if name.endswith(".csv"):
        return pd.read_csv(BytesIO(raw_bytes))
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(raw_bytes))
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}. Use CSV or Excel.")


def categorize_columns(df: pd.DataFrame) -> dict:
    """
    Categorize each column as one of:
      - "step_index": the integer time-step column produced by
        build_regular_grid() (exact name match, case-insensitive) --
        this is an ordering key, not a forecastable series or a date
      - "datetime": parses as a date/timestamp with high confidence
      - "numeric_series": numeric dtype, or convertible to numeric with
        the majority of entries valid
      - "other": text/categorical -- not clearly a date or a numeric series

    Numeric columns are never re-checked against date parsing (see the
    bug note in detect_schema below) -- this function supersedes
    detect_schema with a fuller categorization, but keeps the same fix.
    """
    categories = {}
    for col in df.columns:
        if str(col).strip().lower() == "time_step":
            categories[col] = "step_index"
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            categories[col] = "datetime"
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            categories[col] = "numeric_series"
            continue
        parsed_dt = pd.to_datetime(df[col], errors="coerce")
        if parsed_dt.notna().mean() > 0.9:
            categories[col] = "datetime"
            continue
        parsed_num = pd.to_numeric(df[col], errors="coerce")
        if parsed_num.notna().mean() > 0.5:
            categories[col] = "numeric_series"
        else:
            categories[col] = "other"
    return categories


def coerce_numeric_with_missing(df: pd.DataFrame, value_cols: list) -> tuple:
    """
    Force each named value column to numeric. Any entry that isn't a
    valid number (stray text, e.g. someone typed "N/A" or "pending")
    becomes NaN -- i.e. is treated as a missing value, per the project's
    rule that a time series variable containing text is missing data,
    not a data type to preserve.
    Returns (df_with_numeric_cols, actions_log).
    """
    work = df.copy()
    actions_log = []
    for col in value_cols:
        numeric = pd.to_numeric(work[col], errors="coerce")
        newly_missing = int((work[col].notna() & numeric.isna()).sum())
        work[col] = numeric
        if newly_missing:
            actions_log.append(
                f"'{col}': {newly_missing} non-numeric value(s) treated as missing."
            )
    return work, actions_log


def detect_schema(df: pd.DataFrame) -> dict:
    """
    Pure detection function -- identifies which column(s) look like a date
    and which look like numeric values. Does not ask the user anything;
    app.py decides whether the result is unambiguous enough to skip
    asking, based on how many candidates come back.
    """
    date_candidates = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_candidates.append(col)
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            # Numeric columns are never treated as date candidates here.
            # pd.to_datetime() will happily (mis)interpret plain integers
            # as nanosecond-since-epoch timestamps, which would otherwise
            # falsely flag ordinary numeric value columns as dates.
            continue
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.9:
                date_candidates.append(col)
        except Exception:
            pass

    value_candidates = [
        c for c in df.columns
        if c not in date_candidates and pd.api.types.is_numeric_dtype(df[c])
    ]

    return {"date_candidates": date_candidates, "value_candidates": value_candidates}


def infer_native_frequency(dates: pd.Series) -> dict:
    """
    Infer the rough native frequency of a date column, for display to the
    user (e.g. "this looks like daily data"). Uses the median gap between
    consecutive unique timestamps rather than pandas' infer_freq() alone,
    since infer_freq() gives up on real-world data with any irregularity
    (weekends, holidays, missing rows) -- which is the normal case.
    """
    dates = pd.to_datetime(dates, errors="coerce").dropna().sort_values().drop_duplicates()
    if len(dates) < 2:
        return {"label": "Unknown (not enough distinct dates)", "median_gap": None}

    median_gap = dates.diff().dropna().median()

    if median_gap <= pd.Timedelta(hours=2):
        label = "Hourly (or finer)"
    elif median_gap <= pd.Timedelta(days=2):
        label = "Daily"
    elif median_gap <= pd.Timedelta(days=9):
        label = "Weekly"
    elif median_gap <= pd.Timedelta(days=45):
        label = "Monthly"
    elif median_gap <= pd.Timedelta(days=400):
        label = "Yearly"
    else:
        years = round(median_gap.days / 365.25, 1)
        label = f"Multi-year (~{years} years) — not yet supported for automatic frequency confirmation"

    return {"label": label, "median_gap": str(median_gap)}


def resample_to_frequency(df: pd.DataFrame, date_col: str, value_cols: list,
                           target_freq_label: str, agg_method: str = "mean",
                           interp_method: str = "linear") -> tuple[pd.DataFrame, list]:
    """
    Convert the dataset to an evenly-spaced target frequency, replacing
    the actual calendar timestamps with regular time steps.

    - Downsampling (native finer than target, e.g. daily -> weekly):
      each new step aggregates the original rows that fall inside it,
      using agg_method.
    - Upsampling (native coarser than target, e.g. weekly -> daily):
      resample() creates new, empty time steps; those are then filled
      via interp_method.
    - Both directions go through the same call, so the function doesn't
      need to be told which direction it's going -- it just works from
      whatever the data actually needs.

    Returns (resampled_df, actions_log) -- actions_log is a human-readable
    audit trail of exactly what was done, per the project's dev methodology.
    """
    actions_log = []

    if target_freq_label not in FREQ_LABEL_TO_CODE:
        raise ValueError(f"Unknown target frequency: {target_freq_label}")
    freq_code = FREQ_LABEL_TO_CODE[target_freq_label]

    work = df[[date_col] + value_cols].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    n_dropped = work[date_col].isna().sum()
    if n_dropped:
        work = work.dropna(subset=[date_col])
        actions_log.append(f"Dropped {n_dropped} row(s) with an unparseable date.")

    work = work.sort_values(date_col).set_index(date_col)

    if work.index.duplicated().any():
        dup_count = int(work.index.duplicated().sum())
        work = work.groupby(work.index)[value_cols].mean()
        actions_log.append(
            f"Found {dup_count} duplicate timestamp(s) — averaged before resampling."
        )
    else:
        work = work[value_cols]

    n_before = len(work)
    resampled = work.resample(freq_code).agg(agg_method)
    n_after = len(resampled)
    actions_log.append(
        f"Resampled from {n_before} original rows to {n_after} rows at "
        f"{target_freq_label} frequency, using '{agg_method}' aggregation."
    )

    n_missing = int(resampled.isna().any(axis=1).sum())
    if n_missing:
        interp_kwargs = {"limit_direction": "both"}
        resampled = resampled.interpolate(method=interp_method, **interp_kwargs)
        actions_log.append(
            f"Filled {n_missing} time step(s) that had no data at the new "
            f"frequency, using '{interp_method}' interpolation."
        )

    resampled = resampled.reset_index().rename(columns={resampled.index.name or "index": date_col})
    return resampled, actions_log


# ---------------------------------------------------------------------------
# Time-step conversion, interpolation ranking, cleaning, step-aggregation
#
# Design note on trend removal: this module removes trend via a fitted
# LINEAR REGRESSION line (trend-stationary approach), not differencing,
# specifically for filling gaps -- a smooth fitted trend can be cleanly
# added back at any point, but differencing cannot (a single missing
# value creates two undefined difference values, one on each side of the
# gap, which can't be cleanly inverted back to a level). Differencing
# is still the right tool for stationarizing the COMPLETE series before
# ARIMA/GARCH modeling (per the project's master plan, Phase 2/3) -- it's
# just applied downstream, after this module has already produced a
# complete, gap-free series, not during the gap-filling step itself.
# A log transform is applied first when valid (see _log_transform_if_valid)
# to stabilize variance, since price volatility scales with price level.
# ---------------------------------------------------------------------------

def build_regular_grid(df: pd.DataFrame, date_col: str, value_cols: list,
                        freq_label: str) -> tuple:
    """
    Replace calendar timestamps with a regular time-step grid at the
    confirmed frequency. Any grid point with no matching source row (a
    genuine missing time step) or a value that was already missing
    becomes NaN -- both are treated identically as "missing" to be
    interpolated in the next step.

    The original date is kept as a reference column alongside the new
    'time_step' column (1, 2, 3, ...), rather than discarded -- this
    keeps the result auditable (you can always trace step N back to a
    real calendar date) while `time_step` is what forecasting/modeling
    steps downstream will actually use as the ordering key.

    Returns (grid_df, actions_log). grid_df columns: time_step, date_col,
    then each value column.
    """
    if freq_label not in FREQ_LABEL_TO_CODE:
        raise ValueError(
            f"'{freq_label}' isn't supported for grid construction yet "
            f"(supported: {list(FREQ_LABEL_TO_CODE.keys())})."
        )
    freq_code = FREQ_LABEL_TO_CODE[freq_label]
    actions_log = []

    work = df[[date_col] + value_cols].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    n_dropped = int(work[date_col].isna().sum())
    if n_dropped:
        work = work.dropna(subset=[date_col])
        actions_log.append(f"Dropped {n_dropped} row(s) with an unparseable date.")

    work = work.sort_values(date_col).set_index(date_col)
    if work.index.duplicated().any():
        dup_count = int(work.index.duplicated().sum())
        work = work.groupby(work.index)[value_cols].mean()
        actions_log.append(f"Found {dup_count} duplicate timestamp(s) — averaged before building the grid.")
    else:
        work = work[value_cols]

    grid_index = pd.date_range(start=work.index.min(), end=work.index.max(), freq=freq_code)
    n_before = len(work)
    grid = work.reindex(grid_index)
    grid.index.name = date_col

    n_missing = int(grid.isna().any(axis=1).sum())
    actions_log.append(
        f"Built a regular {freq_label.lower()} grid: {n_before} original rows -> "
        f"{len(grid)} evenly-spaced time steps ({grid_index[0].date()} to {grid_index[-1].date()})."
    )
    if n_missing:
        pct = 100 * n_missing / len(grid)
        actions_log.append(
            f"{n_missing} time step(s) ({pct:.1f}%) are missing at least one value — "
            "either a genuinely missing time step, or a value already flagged as "
            "missing earlier — and need interpolation before forecasting."
        )

    grid = grid.reset_index()
    grid.insert(0, "time_step", range(1, len(grid) + 1))
    return grid, actions_log


# Candidate interpolation methods: (display label, pandas method, extra kwargs).
# Spline/polynomial/akima/pchip require scipy. This set covers the interpolation
# methods most commonly used on time series in practice.
INTERPOLATION_CANDIDATES = [
    ("Linear", "linear", {}),
    ("Nearest", "nearest", {}),
    ("Forward fill", "pad", {}),
    ("Backward fill", "bfill", {}),
    ("Quadratic spline", "spline", {"order": 2}),
    ("Cubic spline", "spline", {"order": 3}),
    ("Polynomial (degree 2)", "polynomial", {"order": 2}),
    ("Akima", "akima", {}),
    ("PCHIP (shape-preserving cubic)", "pchip", {}),
]
_METHOD_LABEL_TO_PARAMS = {label: (method, kwargs) for label, method, kwargs in INTERPOLATION_CANDIDATES}
_DIRECT_METHODS = {"pad": "ffill", "bfill": "bfill"}  # methods pandas 3.x no longer accepts inside .interpolate()


def _interpolate_series(series: pd.Series, method: str, kwargs: dict) -> pd.Series:
    """
    Apply one interpolation method to a series, routing around pandas 3.x
    quirks: 'pad' (forward-fill) and 'bfill' (backward-fill) are no longer
    accepted inside .interpolate() at all and must go through .ffill()/
    .bfill() directly instead.
    """
    if method in _DIRECT_METHODS:
        return getattr(series, _DIRECT_METHODS[method])()
    return series.interpolate(method=method, limit_direction="both", **kwargs)


def _log_transform_if_valid(series: pd.Series) -> tuple:
    """
    Log-transform the series if ALL known values are strictly positive
    (log is undefined at zero/negative). This stabilizes variance for
    data like stock prices, where the size of typical fluctuations scales
    with the price level itself (a $5 move means very different things
    for a $10 stock vs. a $500 stock) -- standard practice before
    comparing/applying interpolation on financial time series.

    Returns (transformed_series, was_logged). If any known value is <= 0
    (e.g. a profit/loss or returns column that can go negative), the
    series is returned unchanged and was_logged=False.
    """
    known = series.dropna()
    if len(known) > 0 and (known > 0).all():
        return np.log(series), True
    return series, False


def fit_linear_trend(values: pd.Series, time_index: pd.Series) -> tuple:
    """
    Fit a simple linear trend (ordinary least squares) to the known
    (non-missing) points of `values` against `time_index`, then predict
    that trend across the FULL time_index range -- including at points
    where `values` is missing.

    This trend is what gets subtracted before interpolation is evaluated
    or applied, and added back afterward. The reason: comparing
    interpolation methods directly on raw trending data (e.g. a rising
    stock price) lets a strong long-term trend dominate the comparison
    and makes the methods look more similar than they really are at
    capturing local/short-term structure. Removing the trend first is
    standard time-series practice and gives a fairer comparison.

    Returns (trend_series, slope, intercept). If there are fewer than 2
    known points, returns a flat trend (no meaningful line can be fit).
    """
    known_mask = values.notna()
    if known_mask.sum() < 2:
        flat_value = float(values[known_mask].mean()) if known_mask.any() else 0.0
        return pd.Series(flat_value, index=values.index), 0.0, flat_value

    x = time_index[known_mask].astype(float)
    y = values[known_mask].astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    trend = intercept + slope * time_index.astype(float)
    trend.index = values.index
    return trend, float(slope), float(intercept)


def evaluate_interpolation_methods(series: pd.Series, time_index: pd.Series = None,
                                    mask_fraction: float = 0.15, n_trials: int = 5,
                                    random_state: int = 42) -> pd.DataFrame:
    """
    Rank candidate interpolation methods for ONE column by how accurately
    they reconstruct values that are already known -- not by any
    assumption about which method "should" work best.

    The comparison is run on the series after two transforms, both
    reversed before scoring is reported in the original scale where
    relevant: (1) a log transform, if every known value is strictly
    positive (stabilizes variance -- typical price fluctuations scale
    with the price level itself), then (2) a long-term linear trend
    removed via fit_linear_trend. This gives a fairer comparison of each
    method's ability to reconstruct local structure, rather than one
    dominated by overall drift or scale.

    Procedure: repeatedly take a random subset of the column's genuinely
    known (non-missing) values, hide them, interpolate, and measure the
    error against what the real (detrended) value actually was. This
    never touches real missing data during evaluation -- only already-
    known points are masked, so the resulting ranking is an honest,
    measured comparison.

    Returns a DataFrame with columns: method, avg_rmse, rank (best first).
    Empty DataFrame if there isn't enough known data to evaluate reliably.
    """
    if time_index is None:
        time_index = pd.Series(range(len(series)), index=series.index)

    transformed, was_logged = _log_transform_if_valid(series)
    trend, _, _ = fit_linear_trend(transformed, time_index)
    residual = transformed - trend

    known_idx = residual[residual.notna()].index
    if len(known_idx) < 10:
        return pd.DataFrame(columns=["method", "avg_rmse", "rank"])

    rng = np.random.default_rng(random_state)
    n_mask = max(1, int(len(known_idx) * mask_fraction))
    scores = {label: [] for label, _, _ in INTERPOLATION_CANDIDATES}

    for _ in range(n_trials):
        mask_positions = rng.choice(known_idx, size=n_mask, replace=False)
        masked = residual.copy()
        masked.loc[mask_positions] = np.nan

        for label, method, kwargs in INTERPOLATION_CANDIDATES:
            try:
                filled = _interpolate_series(masked, method, kwargs)
                errors = filled.loc[mask_positions] - residual.loc[mask_positions]
                rmse = float(np.sqrt(np.mean(np.square(errors))))
                if np.isfinite(rmse):
                    scores[label].append(rmse)
            except Exception:
                continue  # this method failed on this trial (e.g. too few points for the spline order) -- skip

    rows = []
    for label, _, _ in INTERPOLATION_CANDIDATES:
        if scores[label]:
            rows.append({"method": label, "avg_rmse": float(np.mean(scores[label])),
                         "n_trials_succeeded": len(scores[label])})

    result = pd.DataFrame(rows).sort_values("avg_rmse").reset_index(drop=True)
    result["rank"] = result.index + 1
    return result


def rank_interpolation_methods_multi(df: pd.DataFrame, value_cols: list, time_col: str = "time_step",
                                      mask_fraction: float = 0.15, n_trials: int = 5,
                                      random_state: int = 42) -> pd.DataFrame:
    """
    Combine per-column interpolation rankings (each computed on the
    detrended series -- see evaluate_interpolation_methods) into one
    overall ranking across all selected value columns, by averaging each
    method's RANK per column (not raw RMSE -- columns can be on very
    different scales, e.g. a $100 stock vs a $10 stock, so comparing raw
    error directly would unfairly favor whichever column happens to have
    smaller numbers). Returns one recommended default method for the
    whole dataset, plus the full ranking table to show the user.
    """
    time_index = df[time_col]
    per_col_ranks = []
    for col in value_cols:
        col_result = evaluate_interpolation_methods(df[col], time_index, mask_fraction, n_trials, random_state)
        if not col_result.empty:
            per_col_ranks.append(col_result.set_index("method")["rank"].rename(col))

    if not per_col_ranks:
        return pd.DataFrame(columns=["method", "avg_rank"])

    combined = pd.concat(per_col_ranks, axis=1)
    combined["avg_rank"] = combined.mean(axis=1)
    overall = combined.sort_values("avg_rank").reset_index().rename(columns={"index": "method"})
    overall["overall_rank"] = range(1, len(overall) + 1)
    return overall


def apply_interpolation(df: pd.DataFrame, value_cols: list, method_label: str,
                         time_col: str = "time_step") -> tuple:
    """
    Apply the chosen interpolation method to fill missing values/time
    steps in the given columns. Each column is log-transformed (if all
    known values are strictly positive) and has its long-term linear
    trend removed first, the RESIDUAL is interpolated, then the trend is
    added back and the log undone -- matching how the method was ranked
    in evaluate_interpolation_methods.
    Returns (cleaned_df, actions_log).
    """
    if method_label not in _METHOD_LABEL_TO_PARAMS:
        raise ValueError(f"Unknown interpolation method: {method_label}")
    method, kwargs = _METHOD_LABEL_TO_PARAMS[method_label]
    time_index = df[time_col]

    work = df.copy()
    actions_log = []
    for col in value_cols:
        n_missing_before = int(work[col].isna().sum())
        if n_missing_before == 0:
            continue
        transformed, was_logged = _log_transform_if_valid(work[col])
        trend, slope, _ = fit_linear_trend(transformed, time_index)
        residual = transformed - trend
        residual_filled = _interpolate_series(residual, method, kwargs)
        filled_transformed = residual_filled + trend
        work[col] = np.exp(filled_transformed) if was_logged else filled_transformed

        n_missing_after = int(work[col].isna().sum())
        filled = n_missing_before - n_missing_after
        scale_note = "log-scale " if was_logged else ""
        actions_log.append(
            f"'{col}': {scale_note}linear trend removed (slope={slope:.4g} per time step) "
            f"before interpolating, filled {filled} missing value(s)/time step(s) using "
            f"{method_label}, then added the trend back"
            + (" and reversed the log transform." if was_logged else ".")
        )
        if n_missing_after:
            actions_log.append(
                f"'{col}': {n_missing_after} value(s) still missing after interpolation "
                "(not enough surrounding data — e.g. right at the start/end of the series)."
            )
    return work, actions_log


def aggregate_by_step_count(df: pd.DataFrame, date_col: str, value_cols: list,
                             step_size: int, agg_method: str = "mean") -> tuple:
    """
    Combine every `step_size` consecutive time steps into one, by ROW
    COUNT -- not by a calendar unit. E.g. step_size=5 on daily data
    collapses every 5 consecutive time steps into 1, regardless of what
    calendar period that happens to span. This is a distinct operation
    from resample_to_frequency() above, which converts between named
    calendar frequencies (daily/weekly/monthly/...); this instead lets
    the user pick an arbitrary block size directly.

    Returns (aggregated_df, actions_log). Re-numbers time_step 1..N on
    the aggregated result.
    """
    actions_log = []
    if step_size < 2:
        return df.copy(), ["Step size is 1 — no aggregation applied."]

    work = df.reset_index(drop=True)
    group_id = work.index // step_size

    grouped_dates = work.groupby(group_id)[date_col].first()
    grouped_values = work.groupby(group_id)[value_cols].agg(agg_method)
    result = pd.concat([grouped_dates, grouped_values], axis=1).reset_index(drop=True)
    result.insert(0, "time_step", range(1, len(result) + 1))

    actions_log.append(
        f"Aggregated every {step_size} consecutive time steps into 1, using "
        f"'{agg_method}' ({len(work)} steps -> {len(result)} steps)."
    )
    remainder = len(work) % step_size
    if remainder:
        actions_log.append(
            f"The last aggregated step combines only {remainder} original time "
            f"step(s), since {len(work)} isn't evenly divisible by {step_size}."
        )
    return result, actions_log


if __name__ == "__main__":
    # Standalone test block -- run this file directly to sanity-check the
    # module against real (small) test data before wiring it into app.py.
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    test_df = pd.DataFrame({
        "Date": dates,
        "value_a": range(12),
        "value_b": [x * 2 for x in range(12)],
    })

    schema = detect_schema(test_df)
    print("Detected schema:", schema)

    freq_info = infer_native_frequency(test_df["Date"])
    print("Inferred native frequency:", freq_info)

    resampled_df, log = resample_to_frequency(
        test_df, "Date", ["value_a", "value_b"], "Weekly", agg_method="mean"
    )
    print("\nResampled (Daily -> Weekly):")
    print(resampled_df)
    print("\nActions log:")
    for entry in log:
        print(" -", entry)
