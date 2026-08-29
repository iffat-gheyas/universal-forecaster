"""
Module 09: Transformer Forecasting
--------------------------------------
A Transformer encoder for time series forecasting -- the architecture
family behind models like the Temporal Fusion Transformer, referenced
alongside N-BEATS as one of the most-cited modern approaches, and
genuinely the most computationally expensive of the methods in this
project at full scale (multi-head attention costs grow faster than
LSTM/N-BEATS as sequence length and model width increase).

Given the user's explicit direction: defaults here are deliberately
LIGHT (few heads, small model width, one encoder block, short lookback),
with NO caps on any setting -- same "low default, no ceiling" convention
used for N-BEATS/LSTM/DFS/SARIMA-GARCH. If a client later wants full-
scale accuracy, the same settings can be pushed up (and Colab Pro or a
paid compute tier used), consistent with the project's compute-tier /
cost-passed-to-client design established earlier.

Structure, and why it's built this way:
  - A lookback window of log-differenced values is treated as a short
    SEQUENCE (not a flat vector) -- this is what lets self-attention do
    anything useful; it looks at how every time step in the window
    relates to every other one, not just the most recent value.
  - Since attention has no inherent sense of order, a standard sinusoidal
    positional encoding is added so the model can tell "3 steps ago"
    from "yesterday."
  - One (by default) Transformer encoder block: multi-head self-attention
    + a small feedforward layer, each wrapped in a residual connection
    and layer normalization -- the standard Transformer encoder recipe.
  - Like N-BEATS, this predicts the WHOLE horizon in one forward pass per
    ensemble member (multi-output), reusing N-BEATS' windowed-dataset
    builder directly rather than reimplementing the same logic.
  - Same Deep Ensemble interval approach and same log+diff reverse-
    transform convention as every other method in this project.
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

_spec_nbeats = importlib.util.spec_from_file_location("nbeats_core", _MODULE_DIR / "08_nbeats_forecast.py")
nbeats_core = importlib.util.module_from_spec(_spec_nbeats)
_spec_nbeats.loader.exec_module(nbeats_core)

_spec_dfs = importlib.util.spec_from_file_location("dfs_core_for_transformer", _MODULE_DIR / "07_dfs_forecast.py")
dfs_core = importlib.util.module_from_spec(_spec_dfs)
_spec_dfs.loader.exec_module(dfs_core)

detrend = statistical_detrending.detrend
reverse_detrend_cumulative = statistical_detrending.reverse_detrend_cumulative
build_windowed_dataset = nbeats_core.build_windowed_dataset  # identical (lookback -> horizon vector) shape, reused as-is
build_windowed_dataset_multivariate = nbeats_core.build_windowed_dataset_multivariate
align_detrended_series = dfs_core.align_detrended_series  # reused from multivariate DFS, unchanged
_per_band_stats = dfs_core._per_band_stats  # reused: per-SERIES stats here, same generic last-axis logic
_apply_per_band_scaling = dfs_core._apply_per_band_scaling


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding (fixed, not trainable -- standard recipe)
# ---------------------------------------------------------------------------

def _sinusoidal_positional_encoding(length: int, d_model: int) -> np.ndarray:
    if d_model % 2 != 0:
        d_model += 1  # keep it even so sin/cos pairs line up cleanly
    position = np.arange(length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((length, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


# ---------------------------------------------------------------------------
# Transformer encoder model (generic multi-output, like N-BEATS)
# ---------------------------------------------------------------------------

def build_transformer_model(lookback: int, horizon: int, num_heads: int = 2,
                             key_dim: int = 8, ff_dim: int = 16, num_blocks: int = 1,
                             n_features: int = 1):
    """
    Projects the per-timestep input (1 value for univariate, n_features
    values for multivariate) up to a small model width (d_model), adds
    positional encoding, runs it through `num_blocks` encoder blocks
    (self-attention + feedforward, each with a residual connection and
    layer norm), pools across time, and predicts the whole horizon
    vector in one Dense layer.

    `n_features` defaults to 1 (univariate). For multivariate, this
    needs NO architectural change beyond the input shape and the initial
    Dense projection -- Dense layers project from whatever the last
    input dimension is (1 or n_features) up to d_model automatically,
    and self-attention operates on the resulting d_model-wide sequence
    exactly the same way regardless of how many original series
    contributed to each time step's input vector.
    """
    from tensorflow import keras
    import tensorflow as tf

    d_model = key_dim if key_dim % 2 == 0 else key_dim + 1
    pos_enc = _sinusoidal_positional_encoding(lookback, d_model)
    pos_enc_tensor = tf.constant(pos_enc[:, :d_model], dtype=tf.float32)

    inputs = keras.Input(shape=(lookback, n_features))
    x = keras.layers.Dense(d_model)(inputs)
    x = x + pos_enc_tensor

    for _ in range(num_blocks):
        attn_output = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
        x = keras.layers.LayerNormalization()(x + attn_output)
        ff = keras.layers.Dense(ff_dim, activation="relu")(x)
        ff = keras.layers.Dense(d_model)(ff)
        x = keras.layers.LayerNormalization()(x + ff)

    x = keras.layers.GlobalAveragePooling1D()(x)
    outputs = keras.layers.Dense(horizon)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def train_one_model(X: np.ndarray, y: np.ndarray, lookback: int, horizon: int,
                     num_heads: int, key_dim: int, ff_dim: int, num_blocks: int,
                     epochs: int, batch_size: int):
    """
    X may be 2-D (n_samples, lookback) for univariate, or 3-D
    (n_samples, lookback, n_series) for multivariate -- detected
    automatically. Univariate keeps the original single global mean/std
    (unchanged, already tested); multivariate uses PER-SERIES
    standardization (reusing DFS's already-tested per-band helpers,
    same lesson already applied to multivariate N-BEATS) rather than
    risking the scale-mismatch distortion found and fixed in DFS's
    Section 16.3/20. Returns (model, x_stat1, x_stat2, y_mean, y_std).
    """
    if X.ndim == 3:
        n_features = X.shape[2]
        x_stat1, x_stat2 = _per_band_stats(X)
        X_scaled = _apply_per_band_scaling(X, x_stat1, x_stat2)
    else:
        n_features = 1
        x_stat1, x_stat2 = float(X.mean()), float(X.std()) or 1.0
        X_scaled = ((X - x_stat1) / x_stat2).reshape(-1, lookback, 1)

    y_mean, y_std = float(y.mean()), float(y.std()) or 1.0

    model = build_transformer_model(lookback, horizon, num_heads, key_dim, ff_dim, num_blocks, n_features=n_features)
    model.fit(X_scaled, (y - y_mean) / y_std, epochs=epochs, batch_size=batch_size, verbose=0)
    return model, x_stat1, x_stat2, y_mean, y_std


def predict_one_model(model, x_stat1, x_stat2, y_mean, y_std, x_input: np.ndarray) -> np.ndarray:
    """x_input: shape (lookback,) for univariate, or (lookback, n_series)
    for multivariate -- detected automatically. Returns the
    de-standardized forecast vector, shape (horizon,)."""
    lookback = x_input.shape[0]
    if x_input.ndim == 2:
        x_scaled = _apply_per_band_scaling(x_input, x_stat1, x_stat2).reshape(1, lookback, -1)
    else:
        x_scaled = ((x_input - x_stat1) / x_stat2).reshape(1, lookback, 1)
    pred_scaled = model.predict(x_scaled, verbose=0)[0]
    return pred_scaled * y_std + y_mean


# ---------------------------------------------------------------------------
# K-fold CV accuracy (shuffled -- same reasoning as LSTM/DFS/N-BEATS:
# once built into (X, y) samples, this is tabular data)
# ---------------------------------------------------------------------------

def kfold_cv_accuracy(X: np.ndarray, y: np.ndarray, lookback: int, horizon: int,
                       num_heads: int, key_dim: int, ff_dim: int, num_blocks: int,
                       epochs: int, batch_size: int, n_folds: int):
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
            num_heads, key_dim, ff_dim, num_blocks, epochs, batch_size,
        )
        if X.ndim == 3:
            x_val_scaled = _apply_per_band_scaling(X[val_idx], x_stat1, x_stat2)
        else:
            x_val_scaled = ((X[val_idx] - x_stat1) / x_stat2).reshape(-1, lookback, 1)
        preds_scaled = model.predict(x_val_scaled, verbose=0)
        preds = preds_scaled * y_std + y_mean
        all_actual.extend(y[val_idx].flatten().tolist())
        all_pred.extend(preds.flatten().tolist())

    if not all_actual:
        return None
    return forecasting_core.maape_accuracy_pct(np.array(all_actual), np.array(all_pred))


# ---------------------------------------------------------------------------
# Multivariate Transformer (per the same plan pattern as multivariate
# DFS/LSTM/N-BEATS)
#
# Reuses: per-series detrending (detrend()), earliest-end alignment
# (align_detrended_series(), from 07_dfs_forecast.py), the widened
# windowed-dataset builder (build_windowed_dataset_multivariate, from
# 08_nbeats_forecast.py -- identical shape needed here, reused directly
# rather than reimplemented), and the reconstruction function
# (reverse_detrend_cumulative, target's own detrend_info only).
# Genuinely new: nothing beyond what build_transformer_model/
# train_one_model/predict_one_model above already generalized -- unlike
# N-BEATS, no flattening step was needed, since attention/Dense layers
# handle a wider per-timestep feature vector natively.
# ---------------------------------------------------------------------------

def transformer_forecast_multivariate(series_dict: dict, target_col: str, predictor_cols: list,
                                       horizon: int, lookback: int = 10, num_heads: int = 2,
                                       key_dim: int = 8, ff_dim: int = 16, num_blocks: int = 1,
                                       ensemble_size: int = 3, epochs: int = 15, batch_size: int = 8,
                                       confidence: str = "95%", cv_folds: int = 2) -> dict:
    """
    Full multivariate Transformer pipeline.

    series_dict: {column_name: raw_series_array} -- must contain
        target_col and every name in predictor_cols.

    Returns the same shape as transformer_forecast()'s output, plus a
    record of every series' own detrending choice for transparency:
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
        X, y, lookback, horizon, num_heads, key_dim, ff_dim, num_blocks,
        epochs, batch_size, cv_folds,
    )

    latest_window = np.zeros((lookback, len(all_cols)))
    for s_idx, col in enumerate(all_cols):
        latest_window[:, s_idx] = aligned_detrended[col][-lookback:]

    trajectories = []
    for _ in range(ensemble_size):
        model, x_stat1, x_stat2, y_mean, y_std = train_one_model(
            X, y, lookback, horizon, num_heads, key_dim, ff_dim, num_blocks, epochs, batch_size,
        )
        pred_vector = predict_one_model(model, x_stat1, x_stat2, y_mean, y_std, latest_window)
        # Reconstruct using ONLY the target series' own detrend_info --
        # predictors are never reconstructed, since they're never
        # forecast themselves (same rule as every other multivariate method).
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

def transformer_forecast(series: np.ndarray, horizon: int, lookback: int = 10,
                          num_heads: int = 2, key_dim: int = 8, ff_dim: int = 16,
                          num_blocks: int = 1, ensemble_size: int = 3, epochs: int = 15,
                          batch_size: int = 8, confidence: str = "95%", cv_folds: int = 2) -> dict:
    """
    Full pipeline: statistically-justified detrend (Section 17) -> build
    the windowed multi-output dataset -> train a deep ensemble of
    Transformer encoders -> each member forecasts the WHOLE horizon
    vector in one pass -> reconstruct each member's full price
    trajectory via reverse_detrend_cumulative (order-aware) -> combine
    across members (mean = point, std = interval width).

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
        X, y, lookback, horizon, num_heads, key_dim, ff_dim, num_blocks,
        epochs, batch_size, cv_folds,
    )

    latest_window = detrended_series[-lookback:]
    trajectories = []
    for _ in range(ensemble_size):
        model, x_mean, x_std, y_mean, y_std = train_one_model(
            X, y, lookback, horizon, num_heads, key_dim, ff_dim, num_blocks, epochs, batch_size,
        )
        pred_vector = predict_one_model(model, x_mean, x_std, y_mean, y_std, latest_window)
        trajectories.append(reverse_detrend_cumulative(pred_vector, detrend_info))

    trajectories = np.array(trajectories)
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

    result = transformer_forecast(
        series, horizon=3, lookback=8, num_heads=2, key_dim=4, ff_dim=8,
        num_blocks=1, ensemble_size=2, epochs=8, batch_size=4, cv_folds=2,
    )
    print("point:", result["point"])
    print("lower:", result["lower"])
    print("upper:", result["upper"])
    print("kfold_accuracy:", result["kfold_accuracy"])
