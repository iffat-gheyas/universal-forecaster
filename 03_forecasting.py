"""
Module 03: Forecasting (initial slice — mode selection)
----------------------------------------------------------
This module is being built incrementally. This first slice only covers
deciding univariate vs. multivariate mode and resolving which columns
act as the forecast target vs. predictors -- pure logic, no model
fitting yet. Baseline methods (naive, moving average, ARIMA...), the
backtesting framework, and the auto-benchmarking leaderboard are coming
in later sessions.

Kept separate from Module 2 deliberately: Module 3 works on a
cleaned/processed file the user uploads independently (per the
project's design -- each mode/section works on its own uploaded file
rather than silently depending on another section's in-memory state).
"""

MODE_OPTIONS = ["Univariate time series analysis", "Multivariate time series analysis"]

MIN_MULTIVARIATE_SERIES = 2  # need at least a target + 1 predictor


def get_predictor_columns(mode: str, target_col: str, value_cols: list) -> list:
    """
    For multivariate mode, predictors are every selected value column
    except the target (e.g. Tesla forecast using Tesla + other automaker
    stocks). For univariate mode, there are no predictors -- only the
    target's own history is used.
    """
    if mode == "Univariate time series analysis":
        return []
    return [c for c in value_cols if c != target_col]


def can_run_multivariate(value_cols: list) -> bool:
    """Multivariate mode needs at least a target plus one predictor series."""
    return len(value_cols) >= MIN_MULTIVARIATE_SERIES


# ---------------------------------------------------------------------------
# Accuracy metrics (point and interval)
#
# MAAPE and the mean +/- 3*std interval convention below are used in
# Iffat Gheyas et al., "A Resilient Cyber-Physical Demand Forecasting
# System..." (Appl. Sci. 2022, 12, 10093) -- MAAPE is included because it
# is more robust than plain MAPE when actual values are near zero (MAPE
# blows up / is undefined there; MAAPE's arctan bounds the ratio).
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(actual - forecast))))


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Undefined/unstable near actual=0 --
    prefer MAAPE below when the series can be near zero."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    nonzero = actual != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((actual[nonzero] - forecast[nonzero]) / actual[nonzero])) * 100)


def maape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Mean Arctangent Absolute Percentage Error (Kim & Kim, 2016), as used
    in the referenced paper. Bounded and stable even when actual values
    are at or near zero, unlike plain MAPE.
    AAPE_t = arctan(|A_t - F_t| / |A_t|), with the zero-actual case
    handled by treating the ratio as infinite (AAPE -> pi/2), matching
    the metric's intent that a nonzero error against a zero actual is
    "maximally wrong" rather than undefined.
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(actual - forecast) / np.abs(actual)
    ratio = np.where(np.isfinite(ratio), ratio, np.inf)
    aape = np.arctan(ratio)
    return float(np.mean(aape))


def maape_accuracy_pct(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Accuracy = 100 - MAAPE*100, exactly as defined in the reference paper."""
    return 100.0 - maape(actual, forecast) * 100.0


def directional_accuracy(actual: np.ndarray, forecast: np.ndarray, last_actual: float) -> float:
    """
    Percentage of forecasted steps that got the DIRECTION of change right
    (up/down relative to the previous actual value), not just the magnitude.
    Often more business-relevant than point error for decisions like
    "should I expect this to rise or fall".
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    prev = np.concatenate([[last_actual], actual[:-1]])
    actual_dir = np.sign(actual - prev)
    forecast_dir = np.sign(forecast - prev)
    return float(np.mean(actual_dir == forecast_dir) * 100)


def interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """% of actual values that fall within [lower, upper] -- the interval
    equivalent of accuracy. A good interval should hit close to its
    target confidence level (e.g. ~95% of points inside a 95% interval)."""
    actual = np.asarray(actual, dtype=float)
    inside = (actual >= lower) & (actual <= upper)
    return float(np.mean(inside) * 100)


def interval_avg_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Average width of the forecast interval -- narrower is better AT a
    given coverage level (a method that just gives huge intervals can
    "win" on coverage alone, so width must be reported alongside it)."""
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


Z_SCORES_BY_CONFIDENCE = {"90%": 1.645, "95%": 1.96, "99.7%": 3.0}


# ---------------------------------------------------------------------------
# Baseline univariate forecasting methods
#
# Each *_forecast function takes a 1-D array of historical values and a
# horizon H, and returns (point_forecast, lower_bound, upper_bound), each
# an array of length H. This common signature is what lets the backtest
# framework below treat every method interchangeably.
# ---------------------------------------------------------------------------

def _residual_interval(point_forecast: np.ndarray, resid_std: float, confidence: str) -> tuple:
    z = Z_SCORES_BY_CONFIDENCE[confidence]
    lower = point_forecast - z * resid_std
    upper = point_forecast + z * resid_std
    return lower, upper


def naive_forecast(history: np.ndarray, horizon: int, confidence: str = "95%") -> tuple:
    """Every future step equals the last observed value."""
    last = history[-1]
    point = np.full(horizon, last, dtype=float)
    resid_std = float(np.std(np.diff(history))) if len(history) > 1 else 0.0
    # Interval widens with horizon under a random-walk assumption (variance
    # of a k-step-ahead random walk grows with k), same logic ARIMA(0,1,0)
    # would give.
    widen = np.sqrt(np.arange(1, horizon + 1))
    lower, upper = _residual_interval(point, resid_std, confidence)
    lower = point - (point - lower) * widen
    upper = point + (upper - point) * widen
    return point, lower, upper


def moving_average_forecast(history: np.ndarray, horizon: int, window: int = 5, confidence: str = "95%") -> tuple:
    """Every future step equals the average of the last `window` observed values."""
    window = min(window, len(history))
    avg = float(np.mean(history[-window:]))
    point = np.full(horizon, avg, dtype=float)
    resid_std = float(np.std(history[-window:])) if window > 1 else 0.0
    lower, upper = _residual_interval(point, resid_std, confidence)
    return point, lower, upper


def ses_forecast(history: np.ndarray, horizon: int, confidence: str = "95%") -> tuple:
    """Simple Exponential Smoothing -- flat forecast, but weights recent
    observations more than a plain moving average."""
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    model = SimpleExpSmoothing(history, initialization_method="estimated").fit()
    point = model.forecast(horizon)
    resid_std = float(np.std(model.resid))
    lower, upper = _residual_interval(point, resid_std, confidence)
    return np.asarray(point), lower, upper


def holt_forecast(history: np.ndarray, horizon: int, confidence: str = "95%") -> tuple:
    """Holt's linear trend method -- like SES but allows a trending forecast."""
    from statsmodels.tsa.holtwinters import Holt
    model = Holt(history, initialization_method="estimated").fit()
    point = model.forecast(horizon)
    resid_std = float(np.std(model.resid))
    lower, upper = _residual_interval(point, resid_std, confidence)
    return np.asarray(point), lower, upper


def _arima_order_search(history: np.ndarray, seasonal: bool, season_length: int = None,
                         max_p: int = 3, max_d: int = 2, max_q: int = 3) -> tuple:
    """
    Small grid search over (p,d,q) -- and (P,D,Q,s) if seasonal -- picking
    the combination with the lowest AIC. Deliberately a plain grid search
    rather than pulling in a heavier auto-ARIMA dependency, per the
    project's plan to use AIC/BIC search as the default lag-order method.
    Returns (best_order, best_seasonal_order) -- seasonal_order is
    (0,0,0,0) when seasonal=False.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import warnings

    best_aic = np.inf
    best_order = (1, 1, 1)
    best_seasonal = (0, 0, 0, 0)

    d_candidates = range(0, max_d + 1)
    p_candidates = range(0, max_p + 1)
    q_candidates = range(0, max_q + 1)
    seasonal_candidates = [(0, 0, 0, 0)]
    if seasonal and season_length:
        seasonal_candidates += [(1, 1, 0, season_length), (0, 1, 1, season_length)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for d in d_candidates:
            for p in p_candidates:
                for q in q_candidates:
                    if p == 0 and q == 0 and d == 0:
                        continue
                    for seasonal_order in seasonal_candidates:
                        try:
                            model = SARIMAX(
                                history, order=(p, d, q), seasonal_order=seasonal_order,
                                enforce_stationarity=False, enforce_invertibility=False,
                            ).fit(disp=False)
                            if model.aic < best_aic:
                                best_aic = model.aic
                                best_order = (p, d, q)
                                best_seasonal = seasonal_order
                        except Exception:
                            continue
    return best_order, best_seasonal


def arima_forecast(history: np.ndarray, horizon: int, seasonal: bool = False,
                    season_length: int = None, confidence: str = "95%",
                    order: tuple = None, seasonal_order: tuple = None) -> tuple:
    """
    ARIMA (or SARIMA if seasonal=True). Uses statsmodels' native multi-step
    forecast, which is inherently "recursive" in spirit -- each future
    step's distribution is propagated forward through the fitted state
    -space model rather than forecast independently. If order/seasonal_order
    aren't supplied, runs the AIC grid search above to find them.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import warnings

    if order is None:
        order, seasonal_order = _arima_order_search(history, seasonal, season_length)
    if seasonal_order is None:
        seasonal_order = (0, 0, 0, 0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            history, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        fc = model.get_forecast(steps=horizon)
        point = fc.predicted_mean
        alpha = 1 - _confidence_to_level(confidence)
        ci = fc.conf_int(alpha=alpha)
    lower, upper = np.asarray(ci[:, 0]), np.asarray(ci[:, 1])
    return np.asarray(point), lower, upper, order, seasonal_order


def _confidence_to_level(confidence: str) -> float:
    """Convert a confidence label (e.g. '95%') to a 0-1 level (0.95)."""
    return {"90%": 0.90, "95%": 0.95, "99.7%": 0.997}[confidence]


def direct_ar_forecast(history: np.ndarray, horizon: int, lookback: int = 10,
                        confidence: str = "95%") -> tuple:
    """
    "Direct" multi-step baseline: fits a SEPARATE linear regression per
    horizon step h (1..H), each one trained to predict y[t+h] directly
    from the `lookback` most recent values at time t -- rather than
    iterating a one-step model forward (which is what ARIMA/naive/etc.
    effectively do, i.e. the "recursive" approach). This is the classic
    direct-vs-recursive comparison for multi-step forecasting: direct
    avoids compounding one-step errors, at the cost of a separate model
    per horizon and no guarantee the per-horizon forecasts stay mutually
    coherent.
    """
    from sklearn.linear_model import LinearRegression

    n = len(history)
    lookback = min(lookback, n - horizon - 1)
    if lookback < 2:
        # Not enough data for a meaningful direct model -- fall back to naive
        return naive_forecast(history, horizon, confidence)

    point = np.zeros(horizon)
    resid_stds = np.zeros(horizon)
    for h in range(1, horizon + 1):
        X, y = [], []
        for t in range(lookback, n - h):
            X.append(history[t - lookback:t])
            y.append(history[t + h - 1])
        if len(X) < 5:
            point[h - 1] = history[-1]
            resid_stds[h - 1] = 0.0
            continue
        X, y = np.array(X), np.array(y)
        model = LinearRegression().fit(X, y)
        x_latest = history[-lookback:].reshape(1, -1)
        point[h - 1] = model.predict(x_latest)[0]
        resid_stds[h - 1] = float(np.std(y - model.predict(X)))

    z = Z_SCORES_BY_CONFIDENCE[confidence]
    lower = point - z * resid_stds
    upper = point + z * resid_stds
    return point, lower, upper


BASELINE_METHODS = {
    "Naive": lambda hist, h, conf: naive_forecast(hist, h, conf),
    "Moving Average": lambda hist, h, conf: moving_average_forecast(hist, h, confidence=conf),
    "Simple Exponential Smoothing": lambda hist, h, conf: ses_forecast(hist, h, conf),
    "Holt (trend)": lambda hist, h, conf: holt_forecast(hist, h, conf),
    "ARIMA": lambda hist, h, conf: arima_forecast(hist, h, confidence=conf)[:3],
    "Direct AR (direct multi-step)": lambda hist, h, conf: direct_ar_forecast(hist, h, confidence=conf),
}

# Which methods forecast the whole horizon from a single fit, propagated
# forward step by step (the "recursive" family), vs. which fit an
# independent model per horizon step (the "direct" family). Used to label
# results and to let the backtest highlight the recursive-vs-direct
# comparison specifically, not just an undifferentiated leaderboard.
METHOD_STRATEGY = {
    "Naive": "recursive", "Moving Average": "recursive",
    "Simple Exponential Smoothing": "recursive", "Holt (trend)": "recursive",
    "ARIMA": "recursive", "Direct AR (direct multi-step)": "direct",
}


def backtest_forecast_methods(series: np.ndarray, horizon: int, method_names: list = None,
                               confidence: str = "95%", n_origins: int = 5,
                               min_train_size: int = 30) -> pd.DataFrame:
    """
    Walk-forward backtest: repeatedly picks an "origin" point in the
    series, fits each method on everything up to that origin only, and
    scores its horizon-1..H forecast against the actual values that
    follow. This is the same walk-forward principle used everywhere else
    in this project -- a method is only ever judged on data it could not
    have seen.

    ARIMA's (p,d,q) order is searched ONCE on the full series up front
    (not re-searched at every origin) purely to keep backtest runtime
    reasonable; the same order is then refit at each origin.

    Returns a leaderboard DataFrame: one row per method, averaged across
    all origins and all horizon steps, sorted by RMSE (best first). Each
    row also reports which strategy family the method belongs to
    (recursive vs. direct), so the two families can be compared directly.
    """
    if method_names is None:
        method_names = list(BASELINE_METHODS.keys())

    series = np.asarray(series, dtype=float)
    n = len(series)
    max_origin = n - horizon
    if max_origin - min_train_size < n_origins:
        n_origins = max(1, max_origin - min_train_size)
    origins = np.linspace(min_train_size, max_origin - 1, num=n_origins, dtype=int)
    origins = sorted(set(origins.tolist()))

    # Precompute ARIMA's order once on the full series, reused at every origin.
    arima_order, arima_seasonal_order = None, None
    if "ARIMA" in method_names:
        arima_order, arima_seasonal_order = _arima_order_search(series[:max_origin], seasonal=False)

    rows_per_method = {name: [] for name in method_names}

    for origin in origins:
        history = series[:origin + 1]
        actual = series[origin + 1: origin + 1 + horizon]
        if len(actual) < horizon:
            continue

        for name in method_names:
            try:
                if name == "ARIMA":
                    point, lower, upper = arima_forecast(
                        history, horizon, confidence=confidence,
                        order=arima_order, seasonal_order=arima_seasonal_order,
                    )[:3]
                else:
                    point, lower, upper = BASELINE_METHODS[name](history, horizon, confidence)

                rows_per_method[name].append({
                    "mae": mae(actual, point),
                    "rmse": rmse(actual, point),
                    "mape": mape(actual, point),
                    "maape_accuracy": maape_accuracy_pct(actual, point),
                    "directional_accuracy": directional_accuracy(actual, point, last_actual=history[-1]),
                    "interval_coverage": interval_coverage(actual, lower, upper),
                    "interval_avg_width": interval_avg_width(lower, upper),
                })
            except Exception:
                continue  # this method failed at this origin (e.g. too little data) -- skip

    leaderboard_rows = []
    for name in method_names:
        trials = rows_per_method[name]
        if not trials:
            continue
        avg = {k: float(np.mean([t[k] for t in trials])) for k in trials[0]}
        avg["method"] = name
        avg["strategy"] = METHOD_STRATEGY.get(name, "recursive")
        avg["n_origins_succeeded"] = len(trials)
        leaderboard_rows.append(avg)

    leaderboard = pd.DataFrame(leaderboard_rows)
    if leaderboard.empty:
        return leaderboard
    leaderboard = leaderboard.sort_values("rmse").reset_index(drop=True)
    leaderboard["rank"] = leaderboard.index + 1
    cols = ["rank", "method", "strategy", "mae", "rmse", "mape", "maape_accuracy",
            "directional_accuracy", "interval_coverage", "interval_avg_width", "n_origins_succeeded"]
    return leaderboard[cols]
