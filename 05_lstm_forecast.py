"""
Module 05: LSTM Forecasting (direct multi-step, deep ensemble intervals)
---------------------------------------------------------------------------
Implements the LSTM method agreed for the reduced 3-method scope
(univariate_forecasting_methodology_agreed.md, Sections 3-6, 12-14, 17-18):
  - Statistically-justified detrending (Section 17, via
    `10_statistical_detrending.py`'s `detrend()`): F-test-chosen
    variance-stabilizing transform (none/log/sqrt), then t-test-chosen
    differencing order (0, 1, or 2) -- REPLACES the earlier fixed
    "always log, always first-difference" rule.
  - last-k consecutive lags as inputs, k chosen via forward selection
    (Section 5) using shuffled K-fold CV (Section 4) on the training set
  - DIRECT strategy only (Section 14): a separate model per horizon step,
    not one model iterated forward
  - Deep Ensemble for interval estimation (Section 13): 5-10 independently
    initialized LSTMs per horizon step, spread across their predictions
    gives the interval
  - reduced epochs, small ensemble size -- lightweight for Colab free
    tier; both are function parameters so a paying client's run can use
    larger values (ties into the project's compute-tier design)

*** DIRECT-STRATEGY TARGET FORMULA (Section 18) ***
Since the differencing order is no longer fixed at 1, the direct-strategy
target (each horizon-h model predicts the WHOLE h-step-ahead change in
one shot, no recursion) needs a formula that works for order 0, 1, or 2 --
this is handled by `build_direct_cumulative_targets()` /
`reconstruct_direct_forecast()` in `10_statistical_detrending.py`
(shared across all method modules, not reimplemented here). See that
module's docstring for the order-2 weighted-sum derivation.
"""

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

_spec_dfs = importlib.util.spec_from_file_location("dfs_core", _MODULE_DIR / "07_dfs_forecast.py")
dfs_core = importlib.util.module_from_spec(_spec_dfs)
_spec_dfs.loader.exec_module(dfs_core)

detrend = statistical_detrending.detrend
build_direct_cumulative_targets = statistical_detrending.build_direct_cumulative_targets
compute_direct_target = statistical_detrending.compute_direct_target
reconstruct_direct_forecast = statistical_detrending.reconstruct_direct_forecast
align_detrended_series = dfs_core.align_detrended_series  # reused from multivariate DFS, unchanged


# ---------------------------------------------------------------------------
# LSTM model builder and deep ensemble training
# ---------------------------------------------------------------------------

def _build_lstm_model(k: int, units: int = 16, num_layers: int = 1, n_features: int = 1):
    """
    A small, configurable LSTM: `num_layers` stacked LSTM layers of
    `units` hidden nodes each, followed by a Dense output. Both are
    exposed as user-facing settings (not hardcoded) so accuracy can be
    traded off against training time/compute cost.

    `n_features` defaults to 1 (univariate -- one value per time step);
    multivariate mode passes n_features > 1 (one value per SELECTED
    SERIES per time step). LSTM layers handle multi-feature sequences
    natively, so this is the only change needed to the architecture
    itself for multivariate support.
    """
    from tensorflow import keras
    layers = [keras.layers.Input(shape=(k, n_features))]
    for i in range(num_layers):
        return_sequences = i < num_layers - 1  # only the last LSTM layer feeds Dense directly
        layers.append(keras.layers.LSTM(units, return_sequences=return_sequences))
    layers.append(keras.layers.Dense(1))
    model = keras.Sequential(layers)
    model.compile(optimizer="adam", loss="mse")
    return model


def train_deep_ensemble(X: np.ndarray, y: np.ndarray, ensemble_size: int = 5,
                         epochs: int = 20, units: int = 16, num_layers: int = 1,
                         batch_size: int = 16, verbose: int = 0) -> list:
    """
    Train `ensemble_size` independently-initialized LSTMs on the same
    (X, y) -- each gets a different random weight initialization (and a
    different random train/validation split via shuffling), so their
    disagreement reflects genuine model uncertainty, not just noise from
    a single fit. This is the Deep Ensemble approach confirmed in
    Section 13 (not literal MCMC).

    X may be 2-D (n_samples, k) for univariate, or 3-D
    (n_samples, k, n_features) for multivariate -- detected automatically.
    """
    n_k = X.shape[1]
    n_features = X.shape[2] if X.ndim == 3 else 1
    X_reshaped = X.reshape(-1, n_k, n_features)
    models = []
    for member in range(ensemble_size):
        model = _build_lstm_model(n_k, units=units, num_layers=num_layers, n_features=n_features)
        model.fit(X_reshaped, y, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True)
        models.append(model)
    return models


def predict_ensemble(models: list, x_input: np.ndarray) -> tuple:
    """
    x_input: shape (k,) for univariate, or (k, n_features) for
    multivariate -- detected automatically.
    Returns (mean_prediction, std_prediction) across ensemble members.
    """
    if x_input.ndim == 2:
        k, n_features = x_input.shape
    else:
        k, n_features = len(x_input), 1
    x_reshaped = x_input.reshape(1, k, n_features)
    preds = np.array([float(m.predict(x_reshaped, verbose=0)[0, 0]) for m in models])
    return float(np.mean(preds)), float(np.std(preds))


# ---------------------------------------------------------------------------
# Forward selection of k via shuffled K-fold CV (Sections 4-5)
# ---------------------------------------------------------------------------

def _kfold_cv_score(X: np.ndarray, y: np.ndarray, n_folds: int, epochs: int,
                     units: int = 16, num_layers: int = 1, batch_size: int = 16) -> float:
    """Shuffled K-fold CV (Section 4). Operates on already-built (X, y) --
    doesn't need to know diff_order itself, since that only affects HOW
    y was constructed (build_direct_cumulative_targets), not CV mechanics.
    X may be 2-D (univariate) or 3-D (multivariate, n_features > 1) --
    detected automatically. Trains ONE quick model per fold (not a full
    ensemble). Returns average validation RMSE across folds."""
    n = len(X)
    if n < n_folds * 2:
        return np.inf
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    fold_sizes = np.full(n_folds, n // n_folds)
    fold_sizes[: n % n_folds] += 1
    scores = []
    start = 0
    k_lags = X.shape[1]
    n_features = X.shape[2] if X.ndim == 3 else 1
    for fold_size in fold_sizes:
        val_idx = indices[start:start + fold_size]
        train_idx = np.setdiff1d(indices, val_idx)
        start += fold_size
        if len(train_idx) < 5 or len(val_idx) < 1:
            continue
        model = _build_lstm_model(k_lags, units=units, num_layers=num_layers, n_features=n_features)
        model.fit(X[train_idx].reshape(-1, k_lags, n_features), y[train_idx],
                   epochs=epochs, batch_size=batch_size, verbose=0, shuffle=True)
        preds = model.predict(X[val_idx].reshape(-1, k_lags, n_features), verbose=0).flatten()
        scores.append(forecasting_core.rmse(y[val_idx], preds))
    return float(np.mean(scores)) if scores else np.inf


def kfold_cv_accuracy(X: np.ndarray, y: np.ndarray, n_folds: int, epochs: int,
                       units: int = 16, num_layers: int = 1, batch_size: int = 16):
    """
    Same shuffled K-fold split as _kfold_cv_score, but pools every fold's
    out-of-fold predictions together and scores them with MAAPE-based
    accuracy (the project's primary point-accuracy metric, per the user's
    paper) instead of RMSE -- this is what gets shown to the user as
    "average K-fold CV accuracy" after a run. X may be 2-D (univariate)
    or 3-D (multivariate) -- detected automatically. Returns None if
    there isn't enough data to run CV at all.
    """
    n = len(X)
    if n < n_folds * 2:
        return None
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    fold_sizes = np.full(n_folds, n // n_folds)
    fold_sizes[: n % n_folds] += 1
    all_actual, all_pred = [], []
    start = 0
    k_lags = X.shape[1]
    n_features = X.shape[2] if X.ndim == 3 else 1
    for fold_size in fold_sizes:
        val_idx = indices[start:start + fold_size]
        train_idx = np.setdiff1d(indices, val_idx)
        start += fold_size
        if len(train_idx) < 5 or len(val_idx) < 1:
            continue
        model = _build_lstm_model(k_lags, units=units, num_layers=num_layers, n_features=n_features)
        model.fit(X[train_idx].reshape(-1, k_lags, n_features), y[train_idx],
                   epochs=epochs, batch_size=batch_size, verbose=0, shuffle=True)
        preds = model.predict(X[val_idx].reshape(-1, k_lags, n_features), verbose=0).flatten()
        all_actual.extend(y[val_idx].tolist())
        all_pred.extend(preds.tolist())
    if not all_actual:
        return None
    return forecasting_core.maape_accuracy_pct(np.array(all_actual), np.array(all_pred))


def forward_select_k_and_units(detrended_series: np.ndarray, h: int, diff_order: int, max_k: int = 8,
                                unit_candidates: list = None, n_folds: int = 2,
                                epochs: int = 15, num_layers: int = 1,
                                batch_size: int = 16) -> tuple:
    """
    Forward selection (Section 5), extended to also trial-and-error the
    LSTM's hidden unit count -- consistent with the project's established
    philosophy (the DFS paper's own CNN hyperparameters are "determined
    purely on a trial-and-error basis"), rather than leaving architecture
    size silently fixed while only k gets this treatment.

    At each candidate k, every unit_candidate is tried via K-fold CV; the
    best (k, units) pair seen so far is tracked, and k stops growing once
    a new k fails to beat the best score found at any smaller k -- same
    stopping rule as before, just also picking units along the way.

    Number of layers is deliberately NOT searched -- kept fixed at one
    LSTM layer, standard practice for this problem size (confirmed with
    the user).

    Returns (best_k, best_units, history) -- history is a list of dicts
    {k, units, rmse} for transparency/debugging.
    """
    if unit_candidates is None:
        unit_candidates = [8, 16, 32]

    best_k, best_units, best_score = 1, unit_candidates[0], np.inf
    history = []

    for k in range(1, max_k + 1):
        X, y = build_direct_cumulative_targets(detrended_series, k, h, diff_order)
        if len(X) < n_folds * 2:
            break

        best_score_this_k = np.inf
        best_units_this_k = unit_candidates[0]
        for units in unit_candidates:
            score = _kfold_cv_score(X, y, n_folds=n_folds, epochs=epochs,
                                     units=units, num_layers=num_layers, batch_size=batch_size)
            history.append({"k": k, "units": units, "rmse": score})
            if score < best_score_this_k:
                best_score_this_k = score
                best_units_this_k = units

        if best_score_this_k < best_score:
            best_score = best_score_this_k
            best_k = k
            best_units = best_units_this_k
        else:
            break  # stop growing k once it stops improving on the best found so far

    return best_k, best_units, history


# ---------------------------------------------------------------------------
# Multivariate LSTM (per multivariate_lstm_plan.md, confirmed with the user)
#
# Reuses: per-series detrending (detrend()), earliest-end alignment
# (align_detrended_series(), imported from 07_dfs_forecast.py -- same
# fix, same reasoning, no need to reimplement), the deep ensemble
# training/prediction functions above (now n_features-aware), and
# reconstruct_direct_forecast (target's own detrend_info only).
#
# Genuinely new: build_direct_cumulative_targets_multivariate(), which
# widens each of the k lag positions into a feature vector across every
# selected series instead of a single scalar -- the model input becomes
# (k, n_series) instead of (k,). Every ensemble member sees this same
# widened input (unlike DFS's boosting ensemble, LSTM's deep ensemble
# members are independent, not sequential -- confirmed in the plan file
# as the natural, unambiguous design here).
# ---------------------------------------------------------------------------

def build_direct_cumulative_targets_multivariate(aligned_detrended: dict, target_col: str,
                                                   predictor_cols: list, k: int, h: int,
                                                   target_diff_order: int) -> tuple:
    """
    Step 4 of multivariate_lstm_plan.md: X[i] is a (k, n_series) matrix --
    at each of the k lag positions, the target's own detrended value AND
    every predictor's detrended value at that SAME time step, side by
    side. y[i] is computed ONLY from the target series' own detrended
    array, via the same order-aware direct-strategy formula used
    everywhere else (compute_direct_target) -- never touches predictors.

    aligned_detrended: {column_name: detrended_array}, all the SAME
        length (output of align_detrended_series()).
    Returns (X, y): X shape (n_samples, k, n_series), y shape (n_samples,).
    """
    all_cols = [target_col] + list(predictor_cols)
    n_series = len(all_cols)
    T = len(aligned_detrended[target_col])  # same for every series after alignment
    n_samples = T - k - h + 1
    if n_samples <= 0:
        return np.empty((0, k, n_series)), np.empty((0,))

    X = np.zeros((n_samples, k, n_series))
    y = np.zeros(n_samples)
    target_detrended = aligned_detrended[target_col]

    for i in range(n_samples):
        for s_idx, col in enumerate(all_cols):
            X[i, :, s_idx] = aligned_detrended[col][i:i + k]
        window = target_detrended[i + k: i + k + h]
        y[i] = compute_direct_target(window, h, target_diff_order)

    return X, y


def forward_select_k_and_units_multivariate(aligned_detrended: dict, target_col: str, predictor_cols: list,
                                              h: int, target_diff_order: int, max_k: int = 8,
                                              unit_candidates: list = None, n_folds: int = 2,
                                              epochs: int = 15, num_layers: int = 1,
                                              batch_size: int = 16) -> tuple:
    """
    Multivariate counterpart of forward_select_k_and_units -- IDENTICAL
    search logic (grow k from 1, try unit candidates at each k via
    K-fold CV, stop once performance stops improving on the best found
    so far), just built on the widened multivariate dataset instead of
    the plain 1-D one. Still searches for a single shared k and a single
    shared hidden-unit count (Section 5/16-style, confirmed in the plan).
    """
    if unit_candidates is None:
        unit_candidates = [8, 16, 32]

    best_k, best_units, best_score = 1, unit_candidates[0], np.inf
    history = []

    for k in range(1, max_k + 1):
        X, y = build_direct_cumulative_targets_multivariate(
            aligned_detrended, target_col, predictor_cols, k, h, target_diff_order
        )
        if len(X) < n_folds * 2:
            break

        best_score_this_k = np.inf
        best_units_this_k = unit_candidates[0]
        for units in unit_candidates:
            score = _kfold_cv_score(X, y, n_folds=n_folds, epochs=epochs,
                                     units=units, num_layers=num_layers, batch_size=batch_size)
            history.append({"k": k, "units": units, "rmse": score})
            if score < best_score_this_k:
                best_score_this_k = score
                best_units_this_k = units

        if best_score_this_k < best_score:
            best_score = best_score_this_k
            best_k = k
            best_units = best_units_this_k
        else:
            break

    return best_k, best_units, history


def lstm_direct_forecast_multivariate(series_dict: dict, target_col: str, predictor_cols: list,
                                       horizon: int, mode: str = "auto", k: int = None, units: int = None,
                                       num_layers: int = 1, ensemble_size: int = 5, epochs: int = 20,
                                       confidence: str = "95%", max_k_search: int = 8,
                                       unit_candidates: list = None, cv_folds: int = 2,
                                       batch_size: int = 16) -> dict:
    """
    Full multivariate LSTM pipeline (Steps 1-9 of multivariate_lstm_plan.md).

    series_dict: {column_name: raw_series_array} -- must contain target_col
        and every name in predictor_cols, all the same length and time-
        aligned (as they naturally are, coming from the same uploaded file).

    Returns the same shape as lstm_direct_forecast()'s output, plus a
    record of every series' own detrending choice for transparency:
    {
        "point": array[H], "lower": array[H], "upper": array[H],
        "k_used": int, "units_used": int, "search_history": list or None,
        "kfold_accuracy": float or None,
        "target_detrend_transform": str, "target_detrend_diff_order": int,
        "predictor_detrend_info": {col: {"transform": str, "diff_order": int}, ...},
    }
    """
    all_cols = [target_col] + list(predictor_cols)

    # Step 2: detrend every series independently
    detrend_infos = {col: detrend(series_dict[col]) for col in all_cols}

    # Step 3: align by trimming from the earliest end only (reused from DFS)
    aligned_detrended = align_detrended_series(detrend_infos)

    target_detrend_info = detrend_infos[target_col]
    target_diff_order = target_detrend_info["diff_order"]

    if mode == "auto":
        k_used, units_used, search_history = forward_select_k_and_units_multivariate(
            aligned_detrended, target_col, predictor_cols, h=1, target_diff_order=target_diff_order,
            max_k=max_k_search, unit_candidates=unit_candidates, n_folds=cv_folds, epochs=epochs,
            num_layers=num_layers, batch_size=batch_size,
        )
    elif mode == "manual":
        if k is None or units is None:
            raise ValueError("mode='manual' requires both k and units to be specified.")
        k_used, units_used, search_history = k, units, None
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'auto' or 'manual')")

    X_h1, y_h1 = build_direct_cumulative_targets_multivariate(
        aligned_detrended, target_col, predictor_cols, k_used, 1, target_diff_order
    )
    kfold_accuracy = kfold_cv_accuracy(
        X_h1, y_h1, n_folds=cv_folds, epochs=epochs,
        units=units_used, num_layers=num_layers, batch_size=batch_size,
    )

    z = forecasting_core.Z_SCORES_BY_CONFIDENCE[confidence]
    point = np.zeros(horizon)
    lower = np.zeros(horizon)
    upper = np.zeros(horizon)

    # Latest k rows of EVERY series, side by side (Step 8's prediction input)
    latest_k_multivariate = np.zeros((k_used, len(all_cols)))
    for s_idx, col in enumerate(all_cols):
        latest_k_multivariate[:, s_idx] = aligned_detrended[col][-k_used:]

    for h in range(1, horizon + 1):
        X, y = build_direct_cumulative_targets_multivariate(
            aligned_detrended, target_col, predictor_cols, k_used, h, target_diff_order
        )
        models = train_deep_ensemble(
            X, y, ensemble_size=ensemble_size, epochs=epochs,
            units=units_used, num_layers=num_layers, batch_size=batch_size,
        )
        mean_pred, std_pred = predict_ensemble(models, latest_k_multivariate)

        # Step 9: reconstruct using ONLY the target series' own detrend_info
        point[h - 1] = reconstruct_direct_forecast(mean_pred, target_detrend_info, h)
        lower[h - 1] = reconstruct_direct_forecast(mean_pred - z * std_pred, target_detrend_info, h)
        upper[h - 1] = reconstruct_direct_forecast(mean_pred + z * std_pred, target_detrend_info, h)

    return {
        "point": point, "lower": lower, "upper": upper,
        "k_used": k_used, "units_used": units_used, "search_history": search_history,
        "kfold_accuracy": kfold_accuracy,
        "target_detrend_transform": target_detrend_info["transform"],
        "target_detrend_diff_order": target_diff_order,
        "predictor_detrend_info": {
            col: {"transform": detrend_infos[col]["transform"], "diff_order": detrend_infos[col]["diff_order"]}
            for col in predictor_cols
        },
    }


# ---------------------------------------------------------------------------
# Full orchestration: forecast H steps ahead using direct LSTM ensembles
# ---------------------------------------------------------------------------

def lstm_direct_forecast(series: np.ndarray, horizon: int, mode: str = "auto",
                          k: int = None, units: int = None, num_layers: int = 1,
                          ensemble_size: int = 5, epochs: int = 20,
                          confidence: str = "95%", max_k_search: int = 8,
                          unit_candidates: list = None, cv_folds: int = 2,
                          batch_size: int = 16) -> dict:
    """
    Full pipeline: statistically-justified detrend (Section 17) ->
    resolve k and units (auto trial-and-error search, or user-specified
    manual values) -> for each horizon step h=1..H, build the direct
    cumulative-target dataset (Section 18, order-aware), train a deep
    ensemble, and forecast from the latest k known detrended values ->
    reconstruct each horizon's forecast back to the original scale via
    reconstruct_direct_forecast (handles whichever transform/diff_order
    Section 17 chose).

    mode="auto": k and units are found via forward_select_k_and_units
        (trial-and-error, can take a while -- runs a full CV cycle per
        candidate).
    mode="manual": k and units must both be supplied directly; no search
        happens at all, so this is much faster -- the right choice when
        compute time matters more than searching for the best setting.

    Returns a dict: {
        "point": array[H], "lower": array[H], "upper": array[H],
        "k_used": int, "units_used": int, "search_history": list or None,
        "kfold_accuracy": float or None,
        "detrend_transform": str, "detrend_diff_order": int,
    }
    """
    detrend_info = detrend(series)
    detrended_series = detrend_info["detrended_series"]
    diff_order = detrend_info["diff_order"]

    if mode == "auto":
        k_used, units_used, search_history = forward_select_k_and_units(
            detrended_series, h=1, diff_order=diff_order, max_k=max_k_search,
            unit_candidates=unit_candidates, n_folds=cv_folds, epochs=epochs,
            num_layers=num_layers, batch_size=batch_size,
        )
    elif mode == "manual":
        if k is None or units is None:
            raise ValueError("mode='manual' requires both k and units to be specified.")
        k_used, units_used, search_history = k, units, None
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'auto' or 'manual')")

    # Average K-fold CV accuracy for the FINAL (k, units), reported
    # regardless of mode -- auto mode already ran CV to pick these values,
    # but this is a fresh, dedicated pass on the chosen combo specifically
    # (not just whichever fold scores happened to win during search), and
    # manual mode gets this evaluation for the first time since it skips
    # search entirely otherwise.
    X_h1, y_h1 = build_direct_cumulative_targets(detrended_series, k_used, 1, diff_order)
    kfold_accuracy = kfold_cv_accuracy(
        X_h1, y_h1, n_folds=cv_folds, epochs=epochs,
        units=units_used, num_layers=num_layers, batch_size=batch_size,
    )

    z = forecasting_core.Z_SCORES_BY_CONFIDENCE[confidence]
    point = np.zeros(horizon)
    lower = np.zeros(horizon)
    upper = np.zeros(horizon)

    latest_k_values = detrended_series[-k_used:]

    for h in range(1, horizon + 1):
        X, y = build_direct_cumulative_targets(detrended_series, k_used, h, diff_order)
        models = train_deep_ensemble(
            X, y, ensemble_size=ensemble_size, epochs=epochs,
            units=units_used, num_layers=num_layers, batch_size=batch_size,
        )
        mean_pred, std_pred = predict_ensemble(models, latest_k_values)

        # Reconstruct point AND both interval bounds through the SAME
        # full reverse chain (level reconstruction + transform reversal)
        # -- confirmed elsewhere in this project that CI bounds must go
        # through the identical reverse chain as the point forecast, not
        # be approximated afterward in the original scale.
        point[h - 1] = reconstruct_direct_forecast(mean_pred, detrend_info, h)
        lower[h - 1] = reconstruct_direct_forecast(mean_pred - z * std_pred, detrend_info, h)
        upper[h - 1] = reconstruct_direct_forecast(mean_pred + z * std_pred, detrend_info, h)

    return {
        "point": point, "lower": lower, "upper": upper,
        "k_used": k_used, "units_used": units_used, "search_history": search_history,
        "kfold_accuracy": kfold_accuracy,
        "detrend_transform": detrend_info["transform"], "detrend_diff_order": diff_order,
    }


if __name__ == "__main__":
    # Standalone test block -- small synthetic series, small settings, so
    # this runs quickly as a sanity check before being wired into anything.
    np.random.seed(0)
    n = 60
    trend = np.linspace(100, 130, n)
    series = trend * np.exp(np.random.randn(n) * 0.01)

    print("--- auto mode ---")
    result = lstm_direct_forecast(
        series, horizon=3, mode="auto",
        ensemble_size=2, epochs=5, max_k_search=3, cv_folds=2,
        unit_candidates=[8, 16],
    )
    print("k used:", result["k_used"], "units used:", result["units_used"])
    print("detrend transform:", result["detrend_transform"], "diff order:", result["detrend_diff_order"])
    print("search history:", result["search_history"])
    print("point:", result["point"])
    print("lower:", result["lower"])
    print("upper:", result["upper"])

    print("\n--- manual mode ---")
    result2 = lstm_direct_forecast(
        series, horizon=3, mode="manual", k=4, units=16,
        ensemble_size=2, epochs=5,
    )
    print("k used:", result2["k_used"], "units used:", result2["units_used"])
    print("point:", result2["point"])
