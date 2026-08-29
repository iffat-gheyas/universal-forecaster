"""
Module 06: SARIMA-GARCH Forecasting
--------------------------------------
Implements the SARIMA-GARCH method agreed for the reduced 3-method scope
(univariate_forecasting_methodology_agreed.md, Section 12/13/17).

*** DESIGN DECISION: only Section 17.1 (transform), NOT 17.2 (differencing) ***
Section 17 was agreed as the general detrending procedure for every
method in this project. SARIMA-GARCH is a deliberate exception to HALF
of it: applying Section 17.2's external, t-test-chosen differencing on
top of SARIMA would double-difference the series, since SARIMA already
has its OWN principled, AIC-searched way of handling differencing built
directly into the model (the "I" in ARIMA -- its own d/D order). So this
module applies:
  - Section 17.1 ONLY: F-test-chosen variance-stabilizing transform
    (none/log/sqrt), same procedure as every other method.
  - Differencing is left entirely to SARIMA's own (p,d,q)(P,D,Q,s) order
    search (auto AIC grid search, reusing `03_forecasting.py`'s existing
    `_arima_order_search`) or direct manual specification -- unchanged
    from before this module was updated for Section 17.
- GARCH (via the `arch` package) models the conditional VARIANCE of the
  residuals left over from the SARIMA fit -- this is what turns a
  single-number forecast into a realistic, time-varying interval, rather
  than a constant-width one.
- Reverse-transform: SARIMAX's own `get_forecast()` already returns
  forecasts back in its own (undifferenced) working scale automatically
  -- statsmodels handles undoing its internal d/D differencing itself.
  So the ONLY reverse step needed on our end is undoing the Section 17.1
  transform (exp for log, square for sqrt, no-op for none), via the
  shared `reverse_transform_only()` in `10_statistical_detrending.py`.

Both SARIMA orders and GARCH orders can be found via automatic
trial-and-error (AIC-based grid search) or specified manually by the
user -- same two-mode pattern as Module 05 (LSTM), for the same reason:
auto search is thorough but can take a while, manual is fast when the
user already knows what they want (or wants to save compute).

Per the user's explicit preference: SARIMA's seasonal order is NEVER
found by manually inspecting ACF/PACF plots -- only by automatic search
(auto mode) or direct user specification (manual mode).
"""

import warnings

import numpy as np
import pandas as pd

import importlib.util
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("forecasting_core", _MODULE_DIR / "03_forecasting.py")
forecasting_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(forecasting_core)

_spec_dt = importlib.util.spec_from_file_location("statistical_detrending", _MODULE_DIR / "10_statistical_detrending.py")
statistical_detrending = importlib.util.module_from_spec(_spec_dt)
_spec_dt.loader.exec_module(statistical_detrending)

choose_variance_stabilizing_transform = statistical_detrending.choose_variance_stabilizing_transform
reverse_transform_only = statistical_detrending.reverse_transform_only


# ---------------------------------------------------------------------------
# SARIMA order search / fitting (reuses the project's existing AIC search)
# ---------------------------------------------------------------------------

def search_sarima_order(log_series: np.ndarray, seasonal: bool, season_length: int = None,
                         max_p: int = 3, max_d: int = 2, max_q: int = 3) -> tuple:
    """
    Thin wrapper around the project's existing AIC grid search
    (03_forecasting.py's _arima_order_search) -- reused rather than
    reimplemented, so both the (now-inactive) baseline ARIMA method and
    this SARIMA-GARCH method stay consistent.
    Returns (order, seasonal_order).
    """
    return forecasting_core._arima_order_search(
        log_series, seasonal=seasonal, season_length=season_length,
        max_p=max_p, max_d=max_d, max_q=max_q,
    )


def fit_sarima(log_series: np.ndarray, order: tuple, seasonal_order: tuple = (0, 0, 0, 0),
                exog: np.ndarray = None):
    """Fit a SARIMAX model on the (log) series with a given order, optionally
    with exogenous regressors (multivariate mode). Returns the fitted
    statsmodels results object."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            log_series, exog=exog, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
    return model


# ---------------------------------------------------------------------------
# GARCH order search / fitting
# ---------------------------------------------------------------------------

def search_garch_order(residuals: np.ndarray, max_p: int = 2, max_q: int = 2) -> tuple:
    """
    Small AIC-based grid search over GARCH(p, q) orders -- the GARCH
    equivalent of the SARIMA order search above: automatic trial-and-error
    rather than a fixed guess.
    Returns (best_p, best_q).
    """
    from arch import arch_model
    best_aic = np.inf
    best_order = (1, 1)
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = arch_model(residuals, vol="GARCH", p=p, q=q, rescale=False).fit(disp="off")
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, q)
            except Exception:
                continue
    return best_order


def fit_garch(residuals: np.ndarray, p: int = 1, q: int = 1):
    """Fit a GARCH(p, q) model on the SARIMA residuals. Returns the fitted
    `arch` results object."""
    from arch import arch_model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = arch_model(residuals, vol="GARCH", p=p, q=q, rescale=False).fit(disp="off")
    return model


# ---------------------------------------------------------------------------
# Rolling-origin cross-validation accuracy
#
# IMPORTANT: this is deliberately NOT shuffled K-fold CV. SARIMA is a
# sequential state-space model -- it needs a contiguous, time-ordered
# series to fit at all; randomly shuffling rows the way LSTM/DFS's
# tabular lag-window samples can be shuffled would break the
# autoregressive structure the model depends on. The correct, standard
# time-series-CV substitute is ROLLING-ORIGIN validation: pick several
# points along the training data, fit on everything up to each point,
# score the next step, and average -- each fold still respects "you
# can't see the future," just without shuffling. Labeled distinctly in
# the UI for this reason, not just renamed for variety.
# ---------------------------------------------------------------------------

def rolling_origin_cv_accuracy(transformed_series: np.ndarray, order: tuple, seasonal_order: tuple,
                                n_folds: int = 3) -> float:
    """
    Picks n_folds origins spread across transformed_series (already
    through Section 17.1's variance-stabilizing transform, NOT
    differenced -- SARIMA handles differencing itself per-fit via its
    own order), fits SARIMA on the data up to each origin, forecasts 1
    step ahead, and pools all origins' (actual, forecast) pairs into one
    MAAPE-based accuracy score. Returns None if there isn't enough data.
    """
    n = len(transformed_series)
    min_train = max(10, len(transformed_series) // 4)
    if n - min_train < n_folds:
        return None

    origins = np.linspace(min_train, n - 2, num=n_folds, dtype=int)
    origins = sorted(set(origins.tolist()))

    all_actual, all_pred = [], []
    for origin in origins:
        history = transformed_series[:origin + 1]
        actual_next = transformed_series[origin + 1]
        try:
            fitted = fit_sarima(history, order, seasonal_order)
            pred_next = float(fitted.get_forecast(steps=1).predicted_mean[0])
            all_actual.append(actual_next)
            all_pred.append(pred_next)
        except Exception:
            continue

    if not all_actual:
        return None
    return forecasting_core.maape_accuracy_pct(np.array(all_actual), np.array(all_pred))


def rolling_origin_cv_accuracy_multivariate(aligned_target: np.ndarray, aligned_predictors: dict,
                                             predictor_cols: list, order: tuple, seasonal_order: tuple,
                                             lag_window: int = 3, season_length: int = None,
                                             n_folds: int = 3) -> float:
    """
    Multivariate counterpart of rolling_origin_cv_accuracy -- same
    rolling-origin (not shuffled) reasoning as the univariate version,
    for the same reason (SARIMA needs contiguous, time-ordered data).
    At each origin, builds the lagged-exog training set (Step 4's
    construction) TRUNCATED to that origin's own history, fits, and
    forecasts exactly 1 step ahead -- reuses
    build_multivariate_sarimax_data rather than duplicating the lag
    construction logic.
    """
    n = len(aligned_target)
    min_train = max(15, n // 4)
    if n - min_train < n_folds:
        return None

    origins = np.linspace(min_train, n - 2, num=n_folds, dtype=int)
    origins = sorted(set(origins.tolist()))

    all_actual, all_pred = [], []
    for origin in origins:
        target_hist = aligned_target[:origin + 1]
        predictors_hist = {col: aligned_predictors[col][:origin + 1] for col in predictor_cols}
        try:
            data = build_multivariate_sarimax_data(
                target_hist, predictors_hist, predictor_cols, h=1,
                lag_window=lag_window, season_length=season_length,
            )
            fitted = fit_sarima(data["endog"], order, seasonal_order, exog=data["exog_train"])
            forecast_result = fitted.get_forecast(steps=1, exog=data["exog_future"])
            pred_next = float(np.asarray(forecast_result.predicted_mean)[-1])
            actual_next = aligned_target[origin + 1]
            all_actual.append(actual_next)
            all_pred.append(pred_next)
        except Exception:
            continue

    if not all_actual:
        return None
    return forecasting_core.maape_accuracy_pct(np.array(all_actual), np.array(all_pred))


# ---------------------------------------------------------------------------
# Full orchestration
# ---------------------------------------------------------------------------

def sarima_garch_forecast(series: np.ndarray, horizon: int, mode: str = "auto",
                           order: tuple = None, seasonal_order: tuple = None,
                           garch_p: int = None, garch_q: int = None,
                           seasonal: bool = False, season_length: int = None,
                           confidence: str = "95%",
                           max_p: int = 3, max_d: int = 2, max_q: int = 3,
                           max_garch_p: int = 2, max_garch_q: int = 2,
                           cv_folds: int = 3) -> dict:
    """
    Full pipeline: Section 17.1 variance-stabilizing transform (F-test-
    chosen, NOT differenced -- see module docstring) -> fit/search SARIMA
    for the mean (its own d/D order handles differencing) -> fit/search
    GARCH on SARIMA's residuals for the variance -> combine into point +
    interval forecasts -> reverse ONLY the Section 17.1 transform
    (SARIMAX already un-differences its own forecasts internally).

    mode="auto": both SARIMA order and GARCH order are found via AIC
        grid search (trial-and-error) -- thorough, can take a while.
    mode="manual": order, seasonal_order, garch_p, garch_q must all be
        supplied directly -- no search, much faster.

    Returns a dict: {
        "point": array[H], "lower": array[H], "upper": array[H],
        "order": tuple, "seasonal_order": tuple, "garch_order": (p, q),
        "rolling_cv_accuracy": float or None,
        "detrend_transform": str,
    }
    """
    transform_result = choose_variance_stabilizing_transform(series)
    transformed_series = transform_result["transformed_series"]
    transform_name = transform_result["transform"]

    if mode == "auto":
        order, seasonal_order = search_sarima_order(
            transformed_series, seasonal=seasonal, season_length=season_length,
            max_p=max_p, max_d=max_d, max_q=max_q,
        )
    elif mode == "manual":
        if order is None:
            raise ValueError("mode='manual' requires `order` (p,d,q) to be specified.")
        if seasonal_order is None:
            seasonal_order = (0, 0, 0, 0)
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'auto' or 'manual')")

    rolling_cv_accuracy = rolling_origin_cv_accuracy(transformed_series, order, seasonal_order, n_folds=cv_folds)

    sarima_result = fit_sarima(transformed_series, order, seasonal_order)
    residuals = np.asarray(sarima_result.resid)
    # Drop any leading zeros/NaNs from differencing warm-up
    residuals = residuals[np.isfinite(residuals)]

    if mode == "auto":
        garch_p, garch_q = search_garch_order(residuals, max_p=max_garch_p, max_q=max_garch_q)
    else:
        if garch_p is None or garch_q is None:
            raise ValueError("mode='manual' requires garch_p and garch_q to be specified.")

    garch_result = fit_garch(residuals, p=garch_p, q=garch_q)

    mean_forecast = sarima_result.get_forecast(steps=horizon).predicted_mean
    mean_forecast = np.asarray(mean_forecast)

    garch_forecast = garch_result.forecast(horizon=horizon, reindex=False)
    variance_forecast = np.asarray(garch_forecast.variance.values[-1])
    std_forecast = np.sqrt(variance_forecast)

    z = forecasting_core.Z_SCORES_BY_CONFIDENCE[confidence]
    transformed_lower = mean_forecast - z * std_forecast
    transformed_upper = mean_forecast + z * std_forecast

    # Only the Section 17.1 transform needs reversing here -- SARIMAX's
    # own get_forecast() already returns forecasts back in its working
    # (undifferenced) scale, since differencing was handled internally
    # by the model's own d/D order, not externally by this module.
    point = reverse_transform_only(mean_forecast, transform_name)
    lower = reverse_transform_only(transformed_lower, transform_name)
    upper = reverse_transform_only(transformed_upper, transform_name)

    return {
        "point": point, "lower": lower, "upper": upper,
        "order": order, "seasonal_order": seasonal_order,
        "garch_order": (garch_p, garch_q),
        "rolling_cv_accuracy": rolling_cv_accuracy,
        "detrend_transform": transform_name,
    }


# ---------------------------------------------------------------------------
# Multivariate SARIMA-GARCH (per multivariate_sarima_garch_plan.md)
#
# Two things make this different from DFS/LSTM's multivariate versions,
# both explained in the plan file:
#   1. Target detrending stays TRANSFORM-ONLY (Section 19's existing
#      exception) -- SARIMA's own d/D order still handles differencing
#      for the target internally. Predictors get the FULL Section 17
#      treatment (transform + differencing), since they have no internal
#      differencing mechanism of their own.
#   2. Predictors must be LAGGED (never used at their own future time
#      position) -- a separate SARIMAX fit per horizon h, using predictor
#      lags strictly further back than h, so real forecasting never
#      needs an unknown future predictor value. Per the user's
#      instruction, this uses a WINDOW of consecutive lags per predictor
#      (not just one), plus a seasonal lag if seasonality is enabled --
#      SARIMAX's own fitted coefficients decide which lags matter.
# Variance uses Option B (Constant Conditional Correlation style, per
# the plan file's recommendation, not objected to): univariate GARCH
# fit separately on the target's residuals and on each predictor's own
# detrended series, blended using their correlation -- a named,
# deliberate simplification of full dynamic multivariate GARCH (DCC),
# not a hidden one; see the plan file for the honest trade-off.
# ---------------------------------------------------------------------------

def _predictor_lag_offsets(h: int, lag_window: int, season_length: int = None) -> list:
    """
    Builds the list of lag offsets (all >= h, i.e. all strictly at or
    before the point h steps back from the forecast target -- never more
    recent) used for each predictor's exog columns at horizon h:
    [h, h+1, ..., h+lag_window-1], plus a seasonal lag (h + season_length)
    if seasonality is enabled and not already included.
    """
    offsets = list(range(h, h + lag_window))
    if season_length:
        seasonal_offset = h + season_length
        if seasonal_offset not in offsets:
            offsets.append(seasonal_offset)
    return sorted(set(offsets))


def build_multivariate_sarimax_data(aligned_target: np.ndarray, aligned_predictors: dict,
                                     predictor_cols: list, h: int, lag_window: int = 3,
                                     season_length: int = None) -> dict:
    """
    Builds the training (endog, exog) pair AND the h-step-ahead future
    exog block needed to forecast horizon h, using ONLY predictor lags
    that are strictly h-or-more steps back (Step 4 of the plan -- never
    needs an unknown future predictor value).

    Training: for each valid time t (t >= max_offset), endog[t] =
    aligned_target[t], exog[t] = every predictor's value at each lag
    offset (t - offset for offset in the lag-offset list).

    Forecasting: SARIMAX's own multi-step mechanism is used correctly by
    requesting exactly `h` steps from the model fit on this reduced
    series (whose own last training point corresponds to the real series'
    last observed point) and keeping only the LAST of those h forecasted
    points -- the h future exog rows needed for this are built entirely
    from ALREADY-OBSERVED predictor values (verified in the plan file:
    every lag offset is >= h, so even the h-th future step's exog need
    never reaches beyond the last real observation).

    Returns {"endog": array, "exog_train": array, "exog_future": array
    (shape (h, n_cols))}.
    """
    offsets = _predictor_lag_offsets(h, lag_window, season_length)
    max_offset = max(offsets)
    T = len(aligned_target)

    n_train = T - max_offset
    endog = aligned_target[max_offset:]  # length n_train, real positions max_offset..T-1

    n_cols = len(predictor_cols) * len(offsets)
    exog_train = np.zeros((n_train, n_cols))
    col_idx = 0
    for col in predictor_cols:
        pred_series = aligned_predictors[col]
        for offset in offsets:
            # row i (real position max_offset+i) needs predictor[(max_offset+i)-offset]
            exog_train[:, col_idx] = pred_series[max_offset - offset: T - offset]
            col_idx += 1

    # Future exog: h rows, one per forecasted step m=1..h (real target
    # position T-1+m), each needing predictor[(T-1+m)-offset] for every
    # offset -- all of which are <= T-1 (already observed), since offset >= h >= m.
    exog_future = np.zeros((h, n_cols))
    for m in range(1, h + 1):
        col_idx = 0
        for col in predictor_cols:
            pred_series = aligned_predictors[col]
            for offset in offsets:
                idx = (T - 1 + m) - offset
                exog_future[m - 1, col_idx] = pred_series[idx]
                col_idx += 1

    return {"endog": endog, "exog_train": exog_train, "exog_future": exog_future, "offsets": offsets}


def _blend_variance_ccc(target_resid: np.ndarray, target_variance_forecast: float,
                         aligned_predictors: dict, predictor_cols: list,
                         garch_p: int, garch_q: int) -> float:
    """
    Option B from the plan file: Constant Conditional Correlation style
    variance blend -- a NAMED SIMPLIFICATION of full dynamic multivariate
    GARCH (DCC), not a hidden one.

    IMPORTANT (found via testing, corrected here): the target's residuals
    and each predictor's own series can be on COMPLETELY DIFFERENT SCALES
    -- the target always uses Section 19's transform-only detrending
    (often log, since that's typical for price data), but each predictor
    gets its own independently-chosen Section 17 transform, which could
    easily be "none" (e.g. if its own F-test didn't detect
    heteroskedasticity) while raw price DIFFERENCES sit at a much larger
    absolute scale than the target's log-return residuals. Directly
    averaging raw variance magnitudes across these mismatched scales is
    mathematically invalid and was confirmed, via direct testing, to
    inflate the blended variance by over 200x in one case.

    Fixed by using a DIMENSIONLESS ratio per predictor -- its own current
    conditional variance forecast relative to its own historical
    (unconditional) variance -- rather than ever mixing raw variance
    magnitudes across series. This ratio is always "is this predictor
    currently more or less volatile than its own normal," regardless of
    what scale that predictor happens to be measured in:

        blend_factor = average |correlation| across predictors (0 to 1)
        vol_ratio[col] = predictor's own GARCH variance forecast
                          / predictor's own historical variance
        weighted_vol_ratio = correlation-weighted average of vol_ratio
        adjustment_factor = 1 + blend_factor * (weighted_vol_ratio - 1)
        adjusted_variance = target_variance_forecast * adjustment_factor

    If predictors are uncorrelated with the target (blend_factor ~ 0),
    adjustment_factor ~ 1 -- i.e. this reduces to the target's own
    univariate GARCH variance, so multivariate mode never makes the
    interval worse than univariate would when predictors carry no real
    information. A floor is applied to adjustment_factor so a pathological
    predictor can't collapse the variance toward zero.
    """
    if not predictor_cols:
        return target_variance_forecast

    correlations = []
    vol_ratios = []
    n_overlap_target = len(target_resid)

    for col in predictor_cols:
        pred_series = aligned_predictors[col]
        n_overlap = min(n_overlap_target, len(pred_series))
        if n_overlap < 10:
            continue
        try:
            corr = float(np.corrcoef(target_resid[-n_overlap:], pred_series[-n_overlap:])[0, 1])
            if not np.isfinite(corr):
                continue
            pred_garch = fit_garch(pred_series, p=garch_p, q=garch_q)
            pred_var_forecast = float(pred_garch.forecast(horizon=1, reindex=False).variance.values[-1][0])
            pred_own_historical_variance = float(np.var(pred_series))
            if pred_own_historical_variance <= 0:
                continue
            vol_ratio = pred_var_forecast / pred_own_historical_variance  # dimensionless, ~1 = "normal"
            correlations.append(abs(corr))
            vol_ratios.append(vol_ratio)
        except Exception:
            continue

    if not correlations:
        return target_variance_forecast

    correlations = np.array(correlations)
    vol_ratios = np.array(vol_ratios)
    blend_factor = float(np.mean(correlations))
    weights = correlations / correlations.sum() if correlations.sum() > 0 else np.ones_like(correlations) / len(correlations)
    weighted_vol_ratio = float(np.sum(weights * vol_ratios))

    adjustment_factor = 1 + blend_factor * (weighted_vol_ratio - 1)
    adjustment_factor = max(adjustment_factor, 0.1)  # floor -- never collapse variance toward zero
    return target_variance_forecast * adjustment_factor


def sarima_garch_forecast_multivariate(series_dict: dict, target_col: str, predictor_cols: list,
                                        horizon: int, mode: str = "auto",
                                        order: tuple = None, seasonal_order: tuple = None,
                                        garch_p: int = None, garch_q: int = None,
                                        seasonal: bool = False, season_length: int = None,
                                        confidence: str = "95%",
                                        max_p: int = 3, max_d: int = 2, max_q: int = 3,
                                        max_garch_p: int = 2, max_garch_q: int = 2,
                                        predictor_lag_window: int = 3, cv_folds: int = 3) -> dict:
    """
    Full multivariate SARIMA-GARCH pipeline (Steps 1-9 of
    multivariate_sarima_garch_plan.md).

    series_dict: {column_name: raw_series_array} -- must contain
        target_col and every name in predictor_cols.

    Returns {
        "point": array[H], "lower": array[H], "upper": array[H],
        "order": tuple, "seasonal_order": tuple, "garch_order": (p, q),
        "target_detrend_transform": str,
        "predictor_detrend_info": {col: {"transform": str, "diff_order": int}, ...},
        "predictor_lag_offsets_by_horizon": {h: [offsets...], ...},
        "rolling_cv_accuracy": float or None,
    }
    """
    import importlib.util as _ilu
    _spec_dfs = _ilu.spec_from_file_location("dfs_core_for_sg", _MODULE_DIR / "07_dfs_forecast.py")
    _dfs_core = _ilu.module_from_spec(_spec_dfs)
    _spec_dfs.loader.exec_module(_dfs_core)
    align_detrended_series = _dfs_core.align_detrended_series
    full_detrend = statistical_detrending.detrend

    # Step 2: target gets transform-only (Section 19's existing exception);
    # predictors get the FULL Section 17 treatment (transform + differencing).
    target_transform_result = choose_variance_stabilizing_transform(series_dict[target_col])
    target_detrend_wrapped = {"detrended_series": target_transform_result["transformed_series"]}
    predictor_detrend_infos = {col: full_detrend(series_dict[col]) for col in predictor_cols}

    all_detrend_infos = {target_col: target_detrend_wrapped}
    all_detrend_infos.update(predictor_detrend_infos)

    # Step 3: align by trimming from the earliest end only (reused from DFS)
    aligned = align_detrended_series(all_detrend_infos)
    aligned_target = aligned[target_col]
    aligned_predictors = {col: aligned[col] for col in predictor_cols}

    # Rolling-origin CV accuracy check, using a single h=1 reference order
    # (same cost-scoping pattern used elsewhere in this project -- one
    # representative check, not repeated per horizon).
    cv_data_h1 = build_multivariate_sarimax_data(
        aligned_target, aligned_predictors, predictor_cols, h=1,
        lag_window=predictor_lag_window, season_length=season_length if seasonal else None,
    )
    if mode == "auto":
        cv_order, cv_seasonal_order = search_sarima_order(
            cv_data_h1["endog"], seasonal=seasonal, season_length=season_length,
            max_p=max_p, max_d=max_d, max_q=max_q,
        )
    else:
        cv_order = order if order is not None else (1, 0, 0)
        cv_seasonal_order = seasonal_order if seasonal_order is not None else (0, 0, 0, 0)
    rolling_cv_accuracy = rolling_origin_cv_accuracy_multivariate(
        aligned_target, aligned_predictors, predictor_cols, cv_order, cv_seasonal_order,
        lag_window=predictor_lag_window, season_length=season_length if seasonal else None,
        n_folds=cv_folds,
    )

    z = forecasting_core.Z_SCORES_BY_CONFIDENCE[confidence]
    point = np.zeros(horizon)
    lower = np.zeros(horizon)
    upper = np.zeros(horizon)
    offsets_by_horizon = {}
    final_order, final_seasonal_order, final_garch_order = order, seasonal_order, (garch_p, garch_q)

    for h in range(1, horizon + 1):
        data = build_multivariate_sarimax_data(
            aligned_target, aligned_predictors, predictor_cols, h,
            lag_window=predictor_lag_window, season_length=season_length if seasonal else None,
        )
        offsets_by_horizon[h] = data["offsets"]

        if mode == "auto":
            h_order, h_seasonal_order = search_sarima_order(
                data["endog"], seasonal=seasonal, season_length=season_length,
                max_p=max_p, max_d=max_d, max_q=max_q,
            )
        elif mode == "manual":
            if order is None:
                raise ValueError("mode='manual' requires `order` (p,d,q) to be specified.")
            h_order = order
            h_seasonal_order = seasonal_order if seasonal_order is not None else (0, 0, 0, 0)
        else:
            raise ValueError(f"Unknown mode: {mode!r} (expected 'auto' or 'manual')")

        sarima_result = fit_sarima(data["endog"], h_order, h_seasonal_order, exog=data["exog_train"])
        residuals = np.asarray(sarima_result.resid)
        residuals = residuals[np.isfinite(residuals)]

        if mode == "auto":
            h_garch_p, h_garch_q = search_garch_order(residuals, max_p=max_garch_p, max_q=max_garch_q)
        else:
            if garch_p is None or garch_q is None:
                raise ValueError("mode='manual' requires garch_p and garch_q to be specified.")
            h_garch_p, h_garch_q = garch_p, garch_q

        garch_result = fit_garch(residuals, p=h_garch_p, q=h_garch_q)
        target_var_forecast = float(garch_result.forecast(horizon=1, reindex=False).variance.values[-1][0])

        # Option B: CCC-style variance blend with predictors
        blended_variance = _blend_variance_ccc(
            residuals, target_var_forecast, aligned_predictors, predictor_cols, h_garch_p, h_garch_q,
        )
        std_forecast = np.sqrt(max(blended_variance, 0.0))

        forecast_result = sarima_result.get_forecast(steps=h, exog=data["exog_future"])
        mean_forecast_h = float(np.asarray(forecast_result.predicted_mean)[-1])

        transformed_lower = mean_forecast_h - z * std_forecast
        transformed_upper = mean_forecast_h + z * std_forecast

        # Step 9: reverse ONLY the target's own transform (Section 19's rule)
        point[h - 1] = reverse_transform_only(np.array([mean_forecast_h]), target_transform_result["transform"])[0]
        lower[h - 1] = reverse_transform_only(np.array([transformed_lower]), target_transform_result["transform"])[0]
        upper[h - 1] = reverse_transform_only(np.array([transformed_upper]), target_transform_result["transform"])[0]

        final_order, final_seasonal_order, final_garch_order = h_order, h_seasonal_order, (h_garch_p, h_garch_q)

    return {
        "point": point, "lower": lower, "upper": upper,
        "order": final_order, "seasonal_order": final_seasonal_order, "garch_order": final_garch_order,
        "target_detrend_transform": target_transform_result["transform"],
        "predictor_detrend_info": {
            col: {"transform": predictor_detrend_infos[col]["transform"],
                  "diff_order": predictor_detrend_infos[col]["diff_order"]}
            for col in predictor_cols
        },
        "predictor_lag_offsets_by_horizon": offsets_by_horizon,
        "rolling_cv_accuracy": rolling_cv_accuracy,
    }


if __name__ == "__main__":
    # Standalone test block -- small synthetic series, quick settings.
    np.random.seed(0)
    n = 80
    trend = np.linspace(100, 130, n)
    series = trend * np.exp(np.cumsum(np.random.randn(n) * 0.01))

    print("--- auto mode ---")
    result = sarima_garch_forecast(series, horizon=5, mode="auto", max_p=2, max_d=1, max_q=2)
    print("order:", result["order"], "seasonal_order:", result["seasonal_order"],
          "garch_order:", result["garch_order"], "transform:", result["detrend_transform"])
    print("point:", result["point"])
    print("lower:", result["lower"])
    print("upper:", result["upper"])

    print("\n--- manual mode ---")
    result2 = sarima_garch_forecast(
        series, horizon=5, mode="manual",
        order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), garch_p=1, garch_q=1,
    )
    print("point:", result2["point"])
