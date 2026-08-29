"""
Module 08: N-BEATS Forecasting
----------------------------------
N-BEATS (Oreshkin et al., 2020) placed at/near the top of the M4
forecasting competition and is one of the most-cited modern architectures
in recent forecasting literature -- the method referenced in an earlier
conversation as "computationally expensive but influential." This is the
GENERIC variant (unconstrained fully-connected basis, not the
interpretable trend/seasonality-basis variant) -- a deliberate scope
choice to keep this tractable, same spirit as other simplifications made
elsewhere in this project (e.g. DFS's safety cap, LSTM's fixed layer count).

Core idea, and why it's a natural fit alongside DFS: a stack of small
feedforward "blocks" is chained together. Each block looks at the recent
window, predicts (a) a "backcast" -- its best reconstruction of that same
window, and (b) a "forecast" -- its guess at the future. The backcast is
subtracted from the input before the NEXT block sees it, so each
subsequent block only has to explain what earlier blocks got wrong --
the same boosting/residual-fitting spirit as DFS's CNN ensemble, just
built as one differentiable network trained end-to-end in a single pass,
rather than DFS's sequential fit-then-check-then-refit loop. All blocks'
forecasts are summed for the final prediction.

*** DESIGN NOTES ***

1. Multi-output, not per-step "direct": unlike LSTM (a separate model
   trained per horizon step) or DFS (same), N-BEATS natively predicts the
   ENTIRE horizon vector from ONE forward pass. This is a third strategy
   already named in the project's early planning ("multi-output"). It
   still avoids the recursive concern (no model call feeds its own past
   output back in as input for a later call) -- there's simply one call
   producing the whole trajectory at once, which is actually more
   efficient than LSTM/DFS's per-step approach.

2. Statistically-justified detrending (Section 17, via
   `10_statistical_detrending.py`'s `detrend()`): the network's target is
   a VECTOR of the next H individual detrended-series values (whatever
   Section 17 chose -- raw future levels if diff_order=0, one-step
   differences if diff_order=1, second differences if diff_order=2).
   Because N-BEATS predicts the WHOLE vector in a single forward pass
   (not per-horizon independent models like LSTM), reconstruction uses
   `reverse_detrend_cumulative()` -- the order-aware cumulative/double-
   cumulative-sum reconstruction already built and round-trip-tested in
   `10_statistical_detrending.py` -- rather than LSTM's per-horizon
   scalar `reconstruct_direct_forecast()`, which was designed for
   independently-trained per-horizon models, not a single vector output.

3. Interval via Deep Ensemble (same technique as LSTM, Section 13) --
   independently-initialized N-BEATS networks, each reconstructed into a
   full price trajectory, then mean/std TAKEN ACROSS the reconstructed
   trajectories (not the raw diff vectors) -- this correctly captures
   how much the members' full forecasted paths disagree, not just their
   immediate next-step disagreement.

4. All settings below default LOW (small lookback, few blocks, small
   hidden width, few epochs, small ensemble) for lightweight Colab-free
   use, with NO caps on any of them -- consistent with the project's
   "give a low default, let the user scale up with their own compute"
   convention. A paying client's run can push every one of these higher.
"""

import numpy as np
import pandas as pd

import importlib.util
from pathlib import Path

_MODULE_DIR = Path(__file__).parent

_spec_core = importlib.util.spec_from_file_location("forecasting_core", _MODULE_DIR / "03_forecasting.py")
forecasting_core = importlib.util.module_from_spec(_spec_core)
_spec_core.loader.exec_module(forecasting_core)

_spec_dt = importlib.util.spec_from_file_location("statistical_detrending", _MODULE_DIR / "10_statistical_detrending.py")
statistical_detrending = importlib.util.module_from_spec(_spec_dt)
_spec_dt.loader.exec_module(statistical_detrending)

_spec_dfs = importlib.util.spec_from_file_location("dfs_core_for_nbeats", _MODULE_DIR / "07_dfs_forecast.py")
dfs_core = importlib.util.module_from_spec(_spec_dfs)
_spec_dfs.loader.exec_module(dfs_core)

detrend = statistical_detrending.detrend
reverse_detrend_cumulative = statistical_detrending.reverse_detrend_cumulative
align_detrended_series = dfs_core.align_detrended_series  # reused from multivariate DFS, unchanged
_per_band_stats = dfs_core._per_band_stats  # reused: per-SERIES stats here, same generic last-axis logic
_apply_per_band_scaling = dfs_core._apply_per_band_scaling


# ---------------------------------------------------------------------------
# Windowed dataset: lookback window -> full horizon vector (multi-output)
# ---------------------------------------------------------------------------

def build_windowed_dataset(detrended_series: np.ndarray, lookback: int, horizon: int) -> tuple:
    """
    X[i] = detrended_series[i : i+lookback]              (the lookback window)
    y[i] = detrended_series[i+lookback : i+lookback+horizon]   (the next H
           individual detrended-series values, as a vector -- NOT
           cumulative; reverse_detrend_cumulative handles the order-aware
           cumulative reconstruction afterward, once all H values are
           available together from one forward pass)
    Returns (X, y): X shape (n_samples, lookback), y shape (n_samples, horizon).
    """
    n = len(detrended_series)
    n_samples = n - lookback - horizon + 1
    if n_samples <= 0:
        return np.empty((0, lookback)), np.empty((0, horizon))
    X = np.zeros((n_samples, lookback))
    y = np.zeros((n_samples, horizon))
    for i in range(n_samples):
        X[i] = detrended_series[i:i + lookback]
        y[i] = detrended_series[i + lookback:i + lookback + horizon]
    return X, y


# ---------------------------------------------------------------------------
# N-BEATS model (generic variant): stacked residual blocks
# ---------------------------------------------------------------------------

def build_nbeats_model(lookback: int, horizon: int, n_blocks: int = 2,
                        hidden_units: int = 16, n_hidden_layers: int = 2, n_features: int = 1):
    """
    Each block: a small MLP producing a backcast (reconstruction of the
    input window) and a forecast (the horizon vector). The backcast is
    subtracted from the residual passed to the next block; every block's
    forecast is summed for the final output.

    `n_features` defaults to 1 (univariate). For multivariate (n_features
    > 1), the (lookback, n_features) window is FLATTENED into one vector
    before entering the blocks -- N-BEATS' blocks are plain MLPs (Dense
    layers), so this is the natural way to widen the input: each block's
    backcast reconstructs the whole flattened multivariate window, and
    the network's own weights learn which series/lags matter for
    forecasting the target, the same way LSTM/DFS's widened inputs work.
    """
    from tensorflow import keras

    if n_features > 1:
        inputs = keras.Input(shape=(lookback, n_features))
        residual = keras.layers.Flatten()(inputs)
        input_dim = lookback * n_features
    else:
        inputs = keras.Input(shape=(lookback,))
        residual = inputs
        input_dim = lookback

    forecasts = []
    for i in range(n_blocks):
        x = residual
        for _ in range(n_hidden_layers):
            x = keras.layers.Dense(hidden_units, activation="relu")(x)
        backcast = keras.layers.Dense(input_dim, activation="linear", name=f"backcast_{i}")(x)
        block_forecast = keras.layers.Dense(horizon, activation="linear", name=f"forecast_{i}")(x)
        residual = keras.layers.Subtract()([residual, backcast])
        forecasts.append(block_forecast)

    total_forecast = keras.layers.Add()(forecasts) if len(forecasts) > 1 else forecasts[0]
    model = keras.Model(inputs=inputs, outputs=total_forecast)
    model.compile(optimizer="adam", loss="mse")
    return model


def train_one_model(X: np.ndarray, y: np.ndarray, lookback: int, horizon: int,
                     n_blocks: int, hidden_units: int, n_hidden_layers: int,
                     epochs: int, batch_size: int):
    """
    X may be 2-D (n_samples, lookback) for univariate, or 3-D
    (n_samples, lookback, n_series) for multivariate -- detected
    automatically. Univariate keeps the original single global mean/std
    (unchanged, already tested); multivariate uses PER-SERIES
    standardization (reusing DFS's already-tested per-band helpers) --
    a single global scale across series with very different price
    levels risks the same distortion found and fixed in DFS's Section
    16.3, so this is applied proactively here rather than repeating it.
    Returns (model, x_mean_or_means, x_std_or_stds, y_mean, y_std).
    """
    if X.ndim == 3:
        n_features = X.shape[2]
        x_stat1, x_stat2 = _per_band_stats(X)  # per-series means, stds
        X_scaled = _apply_per_band_scaling(X, x_stat1, x_stat2)
    else:
        n_features = 1
        x_stat1, x_stat2 = float(X.mean()), float(X.std()) or 1.0
        X_scaled = (X - x_stat1) / x_stat2

    y_mean, y_std = float(y.mean()), float(y.std()) or 1.0

    model = build_nbeats_model(lookback, horizon, n_blocks, hidden_units, n_hidden_layers, n_features=n_features)
    model.fit(X_scaled, (y - y_mean) / y_std, epochs=epochs, batch_size=batch_size, verbose=0)
    return model, x_stat1, x_stat2, y_mean, y_std


def predict_one_model(model, x_stat1, x_stat2, y_mean, y_std, x_input: np.ndarray) -> np.ndarray:
    """x_input: shape (lookback,) for univariate, or (lookback, n_series)
    for multivariate -- detected automatically. Returns the
    de-standardized forecast vector, shape (horizon,)."""
    if x_input.ndim == 2:
        x_scaled = _apply_per_band_scaling(x_input, x_stat1, x_stat2).reshape(1, *x_input.shape)
    else:
        x_scaled = ((x_input - x_stat1) / x_stat2).reshape(1, -1)
    pred_scaled = model.predict(x_scaled, verbose=0)[0]
    return pred_scaled * y_std + y_mean


# ---------------------------------------------------------------------------
# K-fold CV accuracy (shuffled -- valid here for the same reason as
# LSTM/DFS: once built into (X, y) samples, this is tabular data)
# ---------------------------------------------------------------------------

def kfold_cv_accuracy(X: np.ndarray, y: np.ndarray, lookback: int, horizon: int,
                       n_blocks: int, hidden_units: int, n_hidden_layers: int,
                       epochs: int, batch_size: int, n_folds: int):
    """Pools out-of-fold predictions across all folds (flattened across
    the horizon dimension) and scores with MAAPE-based accuracy, same
    convention as the other three methods. X may be 2-D (univariate) or
    3-D (multivariate) -- detected automatically inside train_one_model.
    Returns None if too little data."""
    n = len(X)
    if n < n_folds * 2:
        return None
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    fold_sizes = np.full(n_folds, n // n_folds)
    fold_sizes[: n % n_folds] += 1

    all_actual, all_pred = [], []
    start = 0
    for fold_size in fold_sizes:
        val_idx = indices[start:start + fold_size]
        train_idx = np.setdiff1d(indices, val_idx)
        start += fold_size
        if len(train_idx) < 5 or len(val_idx) < 1:
            continue
        model, x_stat1, x_stat2, y_mean, y_std = train_one_model(
            X[train_idx], y[train_idx], lookback, horizon,
            n_blocks, hidden_units, n_hidden_layers, epochs, batch_size,
        )
        if X.ndim == 3:
            X_val_scaled = _apply_per_band_scaling(X[val_idx], x_stat1, x_stat2)
        else:
            X_val_scaled = (X[val_idx] - x_stat1) / x_stat2
        preds_scaled = model.predict(X_val_scaled, verbose=0)
        preds = preds_scaled * y_std + y_mean
        all_actual.extend(y[val_idx].flatten().tolist())
        all_pred.extend(preds.flatten().tolist())

    if not all_actual:
        return None
    return forecasting_core.maape_accuracy_pct(np.array(all_actual), np.array(all_pred))


# ---------------------------------------------------------------------------
# Multivariate N-BEATS (per the same plan pattern as multivariate DFS/LSTM)
#
# Reuses: per-series detrending (detrend()), earliest-end alignment
# (align_detrended_series(), imported from 07_dfs_forecast.py), and the
# reconstruction function (reverse_detrend_cumulative, target's own
# detrend_info only). Genuinely new: build_windowed_dataset_multivariate(),
# which widens the INPUT lookback window into a (lookback, n_series)
# matrix -- the OUTPUT stays a plain vector of the target's own future H
# values, since N-BEATS predicts the whole horizon in one forward pass
# regardless of how many series feed into the input (multi-output
# structure is unaffected by widening the input side).
# ---------------------------------------------------------------------------

def build_windowed_dataset_multivariate(aligned_detrended: dict, target_col: str,
                                         predictor_cols: list, lookback: int, horizon: int) -> tuple:
    """
    X[i] shape (lookback, n_series): at each of the lookback positions,
    the target's own detrended value AND every predictor's detrended
    value at that SAME time step, side by side -- same widening
    principle as multivariate LSTM's dataset builder.
    y[i] shape (horizon,): the target series' own next H detrended
    values (a plain vector, NOT cumulative -- reverse_detrend_cumulative
    handles the order-aware reconstruction afterward, same as univariate
    N-BEATS) -- computed ONLY from the target's own array, never predictors'.

    aligned_detrended: {column_name: detrended_array}, all the SAME
        length (output of align_detrended_series()).
    Returns (X, y): X shape (n_samples, lookback, n_series), y shape (n_samples, horizon).
    """
    all_cols = [target_col] + list(predictor_cols)
    n_series = len(all_cols)
    T = len(aligned_detrended[target_col])  # same for every series after alignment
    n_samples = T - lookback - horizon + 1
    if n_samples <= 0:
        return np.empty((0, lookback, n_series)), np.empty((0, horizon))

    X = np.zeros((n_samples, lookback, n_series))
    y = np.zeros((n_samples, horizon))
    target_detrended = aligned_detrended[target_col]

    for i in range(n_samples):
        for s_idx, col in enumerate(all_cols):
            X[i, :, s_idx] = aligned_detrended[col][i:i + lookback]
        y[i] = target_detrended[i + lookback: i + lookback + horizon]

    return X, y


def nbeats_forecast_multivariate(series_dict: dict, target_col: str, predictor_cols: list,
                                  horizon: int, lookback: int = 10, n_blocks: int = 2,
                                  hidden_units: int = 16, n_hidden_layers: int = 2,
                                  ensemble_size: int = 3, epochs: int = 15, batch_size: int = 8,
                                  confidence: str = "95%", cv_folds: int = 2) -> dict:
    """
    Full multivariate N-BEATS pipeline.

    series_dict: {column_name: raw_series_array} -- must contain
        target_col and every name in predictor_cols.

    Returns the same shape as nbeats_forecast()'s output, plus a record
    of every series' own detrending choice for transparency:
    {
        "point": array[H], "lower": array[H], "upper": array[H],
        "kfold_accuracy": float or None,
        "target_detrend_transform": str, "target_detrend_diff_order": int,
        "predictor_detrend_info": {col: {"transform": str, "diff_order": int}, ...},
    }
    """
    all_cols = [target_col] + list(predictor_cols)

    # Detrend every series independently, then align by trimming from
    # the earliest end only (reused from multivariate DFS)
    detrend_infos = {col: detrend(series_dict[col]) for col in all_cols}
    aligned_detrended = align_detrended_series(detrend_infos)
    target_detrend_info = detrend_infos[target_col]

    X, y = build_windowed_dataset_multivariate(aligned_detrended, target_col, predictor_cols, lookback, horizon)
    if len(X) == 0:
        raise ValueError(
            f"Not enough data: need at least {lookback + horizon} points after "
            f"detrending and alignment, got {len(aligned_detrended[target_col])}."
        )

    kfold_accuracy = kfold_cv_accuracy(
        X, y, lookback, horizon, n_blocks, hidden_units, n_hidden_layers,
        epochs, batch_size, cv_folds,
    )

    latest_window = np.zeros((lookback, len(all_cols)))
    for s_idx, col in enumerate(all_cols):
        latest_window[:, s_idx] = aligned_detrended[col][-lookback:]

    trajectories = []
    for _ in range(ensemble_size):
        model, x_stat1, x_stat2, y_mean, y_std = train_one_model(
            X, y, lookback, horizon, n_blocks, hidden_units, n_hidden_layers, epochs, batch_size,
        )
        pred_vector = predict_one_model(model, x_stat1, x_stat2, y_mean, y_std, latest_window)
        # Step 9 (per the multivariate plan pattern): reconstruct using
        # ONLY the target series' own detrend_info -- predictors are
        # never reconstructed, since they're never forecast themselves.
        trajectories.append(reverse_detrend_cumulative(pred_vector, target_detrend_info))

    trajectories = np.array(trajectories)
    point = trajectories.mean(axis=0)
    std = trajectories.std(axis=0)

    z = forecasting_core.Z_SCORES_BY_CONFIDENCE[confidence]
    lower = point - z * std
    upper = point + z * std

    return {
        "point": point, "lower": lower, "upper": upper, "kfold_accuracy": kfold_accuracy,
        "target_detrend_transform": target_detrend_info["transform"],
        "target_detrend_diff_order": target_detrend_info["diff_order"],
        "predictor_detrend_info": {
            col: {"transform": detrend_infos[col]["transform"], "diff_order": detrend_infos[col]["diff_order"]}
            for col in predictor_cols
        },
    }


# ---------------------------------------------------------------------------
# Full orchestration
# ---------------------------------------------------------------------------

def nbeats_forecast(series: np.ndarray, horizon: int, lookback: int = 10,
                     n_blocks: int = 2, hidden_units: int = 16, n_hidden_layers: int = 2,
                     ensemble_size: int = 3, epochs: int = 15, batch_size: int = 8,
                     confidence: str = "95%", cv_folds: int = 2) -> dict:
    """
    Full pipeline: statistically-justified detrend (Section 17) -> build
    the windowed multi-output dataset -> train a deep ensemble of
    N-BEATS models -> each member forecasts the WHOLE horizon vector in
    one pass -> reconstruct each member's full price trajectory via
    reverse_detrend_cumulative (order-aware) -> combine across members
    (mean = point, std = interval width).

    Returns {
        "point": array[H], "lower": array[H], "upper": array[H],
        "kfold_accuracy": float or None,
        "detrend_transform": str, "detrend_diff_order": int,
    }
    """
    detrend_info = detrend(series)
    detrended_series = detrend_info["detrended_series"]

    X, y = build_windowed_dataset(detrended_series, lookback, horizon)
    if len(X) == 0:
        raise ValueError(
            f"Not enough data: need at least {lookback + horizon} points after "
            f"detrending, got {len(detrended_series)}."
        )

    kfold_accuracy = kfold_cv_accuracy(
        X, y, lookback, horizon, n_blocks, hidden_units, n_hidden_layers,
        epochs, batch_size, cv_folds,
    )

    latest_window = detrended_series[-lookback:]
    trajectories = []
    for _ in range(ensemble_size):
        model, x_mean, x_std, y_mean, y_std = train_one_model(
            X, y, lookback, horizon, n_blocks, hidden_units, n_hidden_layers, epochs, batch_size,
        )
        pred_vector = predict_one_model(model, x_mean, x_std, y_mean, y_std, latest_window)
        trajectories.append(reverse_detrend_cumulative(pred_vector, detrend_info))

    trajectories = np.array(trajectories)  # shape (ensemble_size, horizon)
    point = trajectories.mean(axis=0)
    std = trajectories.std(axis=0)

    z = forecasting_core.Z_SCORES_BY_CONFIDENCE[confidence]
    lower = point - z * std
    upper = point + z * std

    return {
        "point": point, "lower": lower, "upper": upper, "kfold_accuracy": kfold_accuracy,
        "detrend_transform": detrend_info["transform"], "detrend_diff_order": detrend_info["diff_order"],
    }


if __name__ == "__main__":
    # Standalone test block -- small synthetic series, minimal settings.
    np.random.seed(0)
    n = 60
    trend = np.linspace(100, 130, n)
    series = trend * np.exp(np.random.randn(n) * 0.01)

    result = nbeats_forecast(
        series, horizon=3, lookback=8, n_blocks=2, hidden_units=8,
        n_hidden_layers=1, ensemble_size=2, epochs=8, batch_size=4, cv_folds=2,
    )
    print("point:", result["point"])
    print("lower:", result["lower"])
    print("upper:", result["upper"])
    print("kfold_accuracy:", result["kfold_accuracy"])
