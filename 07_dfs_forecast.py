"""
Module 07: DFS Forecasting (wavelet decomposition + boosting CNN ensembles)
-----------------------------------------------------------------------------
Implements the algorithm from univariate_forecasting_methodology_agreed.md
Section 11 (the user's own paper, Algorithm S2 / Pseudocode of PRDFS,
Section 3.2), CORRECTED per Sections 16-18 (confirmed via direct code
inspection and fixed):

Pipeline per horizon step h:
  1. Statistically-justified detrending (Section 17, via
     `10_statistical_detrending.py`): F-test-chosen variance-stabilizing
     transform, THEN t-test-chosen differencing order (0, 1, or 2). This
     detrended "mother series" REPLACES the earlier fixed "always log,
     never difference" version.
  2. Wavelet-decompose the mother series into n+1 component ("child")
     series (one approximation + n detail levels) -- performed on the
     WHOLE available series at once (confirmed with the user; the
     leakage risk this implies is flagged in Section 16.6 of the plan
     file as a known, accepted trade-off, not silently ignored).
  3. Build a sliding-window training set: input = m most recent values
     across ALL n+1 child series (a small 2D "image"), EACH CHILD SERIES
     STANDARDIZED INDEPENDENTLY (Section 16.3 -- corrected from an
     earlier version that used one global scale across all bands
     combined). Target = the DIRECT-strategy value for horizon h
     (Section 18 -- order-aware: the future value itself, a cumulative
     sum, or a weighted cumulative sum, depending on which differencing
     order Section 17 chose) -- NOT separately standardized (Section
     16.4: once properly detrended, the target is already small and
     near-zero-centered, so no extra standardization step is needed).
  4. Train a small 2D CNN (member 1 of Ensemble_mean).
  5. If its residual isn't white noise (mean |residual| >= threshold),
     treat the residual AS THE NEW mother series: wavelet-decompose IT
     (now univariate -- one series, one set of child series), build a
     new training set, train member 2 to predict THIS residual. Repeat
     until white noise (or a safety cap on ensemble size is hit).
  6. Final point forecast = SUM of every member's forecast.
  7. Build a second, parallel ensemble (Ensemble_var) the same way, but
     starting from the SQUARED residuals of Ensemble_mean's final
     member, then the ABSOLUTE VALUE of each subsequent member's
     residual. Final variance forecast = SUM OF ABSOLUTE VALUES of every
     member's forecast.
  8. Interval = point +/- z * sqrt(variance forecast) in the DETRENDED
     scale, then reconstruct point/lower/upper all the way back to the
     original scale via reconstruct_direct_forecast (reverses Section
     17's transform+differencing in exact reverse order).

*** DESIGN NOTES / DEVIATIONS, FLAGGED EXPLICITLY ***

1. Wavelet mother/level are NOT auto-searched (unlike LSTM's k/units
   search) -- the paper describes choosing them by trial and error
   across separate runs, and searching them automatically here would
   multiply an already expensive CNN-ensemble-training cost. Exposed as
   user-configurable settings instead.

2. Safety cap on ensemble size (`max_members`): the paper's stopping
   rule (white noise check) has no guaranteed upper bound in principle.
   Not in the original pseudocode, added for practical robustness.

3. Direct multi-step (Section 14): each horizon step h gets its own
   COMPLETE, independently-built pair of ensembles.

4. Whole-series wavelet decomposition leakage risk (Section 16.6):
   confirmed accepted by the user, not fixed here -- doing so properly
   (redoing decomposition per-window) would multiply an already-costly
   method's training time.
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

detrend = statistical_detrending.detrend
compute_direct_target = statistical_detrending.compute_direct_target
reconstruct_direct_forecast = statistical_detrending.reconstruct_direct_forecast


# ---------------------------------------------------------------------------
# Wavelet decomposition into full-length component series
# ---------------------------------------------------------------------------

def wavelet_decompose_components(series: np.ndarray, wavelet: str = "db4", level: int = 2) -> np.ndarray:
    """
    Decompose `series` into `level + 1` full-length component series:
    one approximation (index 0) and `level` detail series (indices 1..level).
    Each is reconstructed via waverec with every OTHER coefficient band
    zeroed out, so all components share the original series' length and
    can be stacked as parallel "channels" at each time step.
    Returns an array of shape (len(series), level + 1).
    """
    import pywt

    coeffs = pywt.wavedec(series, wavelet, level=level)
    n_bands = len(coeffs)  # = level + 1
    components = np.zeros((len(series), n_bands))

    for band_idx in range(n_bands):
        zeroed = [np.zeros_like(c) for c in coeffs]
        zeroed[band_idx] = coeffs[band_idx]
        reconstructed = pywt.waverec(zeroed, wavelet)
        # waverec can return a slightly different length depending on
        # padding -- trim or pad to match the original series exactly.
        if len(reconstructed) > len(series):
            reconstructed = reconstructed[: len(series)]
        elif len(reconstructed) < len(series):
            reconstructed = np.pad(reconstructed, (0, len(series) - len(reconstructed)))
        components[:, band_idx] = reconstructed

    return components


# ---------------------------------------------------------------------------
# Sliding-window training set for the 2D CNN (Section 18: order-aware target)
# ---------------------------------------------------------------------------

def build_cnn_training_set(components: np.ndarray, target_series: np.ndarray, m: int, h: int,
                            diff_order: int) -> tuple:
    """
    components: shape (T, n_bands) -- output of wavelet_decompose_components.
    target_series: shape (T,) -- the (detrended) series each window's
                   target is drawn from (the mother series for ensemble
                   member 1; a residual series for later members).
    Window i uses components[i : i+m] as input (all bands); the target
    is computed via the SAME order-aware direct-strategy formula used by
    every other method (compute_direct_target, Section 18) -- NOT just
    the raw future value, since the mother series may now be a
    differenced (order 1 or 2) series.

    Returns (X, y): X has shape (n_samples, m, n_bands), y has shape (n_samples,).
    """
    T, n_bands = components.shape
    n_samples = T - m - h + 1
    if n_samples <= 0:
        return np.empty((0, m, n_bands)), np.empty((0,))

    X = np.zeros((n_samples, m, n_bands))
    y = np.zeros(n_samples)
    for i in range(n_samples):
        X[i] = components[i:i + m]
        window = target_series[i + m: i + m + h]
        y[i] = compute_direct_target(window, h, diff_order)
    return X, y


# ---------------------------------------------------------------------------
# 2D CNN model (small, fast -- colab_free tier default)
# ---------------------------------------------------------------------------

def _build_2d_cnn(m: int, n_bands: int, filters: int = 8):
    from tensorflow import keras
    kernel_h = min(3, m)
    kernel_w = min(2, n_bands)
    model = keras.Sequential([
        keras.layers.Input(shape=(m, n_bands, 1)),
        keras.layers.Conv2D(filters, kernel_size=(kernel_h, kernel_w), activation="relu", padding="same"),
        keras.layers.Flatten(),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def is_white_noise(residuals: np.ndarray, threshold: float = 0.05) -> bool:
    """Per the paper's stopping rule: white noise if mean absolute
    residual is below the threshold."""
    if len(residuals) == 0:
        return True
    return float(np.mean(np.abs(residuals))) < threshold


# ---------------------------------------------------------------------------
# Per-band (per-child-series) standardization helpers (Section 16.3)
# ---------------------------------------------------------------------------

def _per_band_stats(X: np.ndarray) -> tuple:
    """X: shape (n_samples, m, n_bands). Returns (means, stds), each
    shape (n_bands,) -- one mean/std PER CHILD SERIES, computed across
    all samples and all m timesteps for that band. This replaces a
    single global mean/std across the whole combined matrix (the bug
    found via code inspection, Section 16.3): different wavelet bands
    naturally sit at very different scales (e.g. the approximation band
    vs. a fine detail band), so one shared scale can wash out the finer
    bands' signal relative to the coarser ones."""
    n_bands = X.shape[2]
    means = np.zeros(n_bands)
    stds = np.zeros(n_bands)
    for j in range(n_bands):
        means[j] = X[:, :, j].mean()
        stds[j] = X[:, :, j].std() or 1.0
    return means, stds


def _apply_per_band_scaling(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """X: shape (..., m, n_bands) (a full training batch OR a single
    window). Standardizes each band independently using the given
    per-band means/stds."""
    return (X - means) / stds  # broadcasts against the last (n_bands) axis


# ---------------------------------------------------------------------------
# Boosting ensemble: build one member, then recurse on its residual
# ---------------------------------------------------------------------------

def _fit_one_member(input_source: np.ndarray, target_source: np.ndarray, m: int, h: int,
                     wavelet: str, level: int, diff_order: int, epochs: int,
                     batch_size: int, filters: int) -> dict:
    """
    Decomposes `input_source`, builds the training set against
    `target_source` (Section 18's order-aware direct target), trains one
    small CNN, and returns everything needed both to compute this
    member's residual (for the next boosting step) and to forecast with
    it later.

    Per Section 16.3: each CHILD SERIES (wavelet band) is standardized
    INDEPENDENTLY (its own mean/std), not with one shared global scale.

    Per Section 16.4, CORRECTED after testing: the target IS standardized
    (zero mean, unit std) before training, contrary to this project's
    earlier conclusion. That conclusion assumed detrending alone (making
    the mean-ensemble's target small, ~0.03 scale for log-returns) was
    enough to guarantee good training -- testing found this false for the
    VARIANCE ensemble specifically: its target (squared residuals) sits
    at an even smaller scale (~0.0006), and without standardization the
    network predicted variance forecasts over 20x larger than the
    largest actual squared-residual value ever seen in training. The
    real lesson: target standardization is general neural-network best
    practice for training stability regardless of whether the raw target
    scale is unusually LARGE (the original bug) or unusually SMALL (this
    one) -- not merely a workaround specific to the original bug, as
    Section 16.4 had concluded. Reinstated for both ensembles.
    """
    components = wavelet_decompose_components(input_source, wavelet, level)
    X, y = build_cnn_training_set(components, target_source, m, h, diff_order)

    n_bands = components.shape[1]

    if len(X) == 0:
        model = _build_2d_cnn(m, n_bands, filters=filters)
        return {"model": model, "components_full": components, "residual": np.array([]),
                "m": m, "n_bands": n_bands,
                "x_means": np.zeros(n_bands), "x_stds": np.ones(n_bands),
                "y_mean": 0.0, "y_std": 1.0}

    x_means, x_stds = _per_band_stats(X)
    X_scaled = _apply_per_band_scaling(X, x_means, x_stds)

    y_mean, y_std = float(y.mean()), float(y.std()) or 1.0
    y_scaled = (y - y_mean) / y_std

    model = _build_2d_cnn(m, n_bands, filters=filters)
    model.fit(X_scaled.reshape(-1, m, n_bands, 1), y_scaled, epochs=epochs, batch_size=batch_size, verbose=0)
    preds_scaled = model.predict(X_scaled.reshape(-1, m, n_bands, 1), verbose=0).flatten()
    preds = preds_scaled * y_std + y_mean
    residual = y - preds

    return {"model": model, "components_full": components, "residual": residual,
            "m": m, "n_bands": n_bands, "x_means": x_means, "x_stds": x_stds,
            "y_mean": y_mean, "y_std": y_std}


def kfold_cv_accuracy(series: np.ndarray, m: int, h: int, wavelet: str, level: int,
                       diff_order: int, epochs: int, batch_size: int, filters: int,
                       n_folds: int) -> float:
    """
    Shuffled K-fold CV (same fold-splitting convention as LSTM's -- valid
    here because, once the wavelet components and sliding-window (X, y)
    samples are built, this is exactly the same kind of tabular reduction
    LSTM already uses, so the samples can be shuffled the same way).

    Scoped to a SINGLE CNN fit per fold (i.e. what would be ensemble
    member 1 alone), NOT the full boosting ensemble -- running the whole
    white-noise-until-convergence ensemble inside every CV fold would
    multiply an already expensive method's cost by several times over.

    Returns None if there isn't enough data to run CV at all.
    """
    components = wavelet_decompose_components(series, wavelet, level)
    X, y = build_cnn_training_set(components, series, m, h, diff_order)
    n = len(X)
    if n < n_folds * 2:
        return None

    n_bands = components.shape[1]
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

        X_train, y_train = X[train_idx], y[train_idx]
        x_means, x_stds = _per_band_stats(X_train)
        y_mean, y_std = float(y_train.mean()), float(y_train.std()) or 1.0

        model = _build_2d_cnn(m, n_bands, filters=filters)
        model.fit(
            _apply_per_band_scaling(X_train, x_means, x_stds).reshape(-1, m, n_bands, 1),
            (y_train - y_mean) / y_std, epochs=epochs, batch_size=batch_size, verbose=0,
        )
        preds_scaled = model.predict(
            _apply_per_band_scaling(X[val_idx], x_means, x_stds).reshape(-1, m, n_bands, 1), verbose=0
        ).flatten()
        preds = preds_scaled * y_std + y_mean

        all_actual.extend(y[val_idx].tolist())
        all_pred.extend(preds.tolist())

    if not all_actual:
        return None
    return forecasting_core.maape_accuracy_pct(np.array(all_actual), np.array(all_pred))


def train_boosting_ensemble(series: np.ndarray, m: int, h: int, wavelet: str = "db4",
                             level: int = 2, diff_order: int = 1, epochs: int = 10,
                             batch_size: int = 8, filters: int = 8, max_members: int = 3,
                             threshold: float = 0.05, apply_abs: bool = False) -> list:
    """
    Builds the full sequential/boosting ensemble (Section 11, steps 2-8).
    `apply_abs=True` is used for Ensemble_var: every member after the
    first predicts the ABSOLUTE VALUE of the previous member's residual
    (keeps the quantity being modeled non-negative, like a variance).
    `max_members` is the safety cap described in the module docstring.

    `diff_order` is only meaningful for MEMBER 1 (whose target is drawn
    from the properly detrended mother series). From member 2 onward,
    the "series" being boosted on is a RESIDUAL -- already a plain
    difference-like quantity with no further transform/differencing
    structure of its own -- so those members always use diff_order=1
    (the residual itself IS the direct target, no cumulative summing
    needed -- but summing over h=1 point is a no-op, so an order-1 call
    with a single-point window is safe and consistent).
    Returns a list of member dicts (see _fit_one_member).
    """
    members = []
    current_input = series
    current_target = series
    current_diff_order = diff_order

    for _ in range(max_members):
        member = _fit_one_member(
            current_input, current_target, m, h, wavelet, level, current_diff_order,
            epochs, batch_size, filters,
        )
        members.append(member)

        residual = member["residual"]
        if is_white_noise(residual, threshold) or len(residual) == 0:
            break

        next_series = np.abs(residual) if apply_abs else residual
        current_input = next_series
        current_target = next_series
        current_diff_order = 0  # residual IS the direct target already -- no further summing

    return members


def predict_ensemble_sum(members: list, use_abs_sum: bool = False) -> float:
    """
    Forecasts with every member (using the tail of ITS OWN decomposed
    input series as the prediction window) and combines: plain sum for
    Ensemble_mean, sum of absolute values for Ensemble_var (guarantees
    a non-negative variance forecast). Applies each member's own
    PER-BAND input standardization AND target de-standardization
    consistently (Section 16.3, 16.4-corrected).
    """
    total = 0.0
    for member in members:
        m = member["m"]
        components = member["components_full"]
        if len(components) < m:
            continue
        x_input = components[-m:]
        x_scaled = _apply_per_band_scaling(x_input, member["x_means"], member["x_stds"])
        x_scaled = x_scaled.reshape(1, m, member["n_bands"], 1)
        pred_scaled = float(member["model"].predict(x_scaled, verbose=0)[0, 0])
        pred = pred_scaled * member["y_std"] + member["y_mean"]
        total += abs(pred) if use_abs_sum else pred
    return total


# ---------------------------------------------------------------------------
# Multivariate DFS (per multivariate_dfs_plan.md, confirmed with the user)
#
# Design: only MEMBER 1 of the mean ensemble is genuinely multivariate --
# it's built from every selected series (target + predictors), each
# independently detrended, wavelet-decomposed, and per-band-standardized,
# then horizontally concatenated into one wide input. Every subsequent
# boosting member (mean ensemble member 2+, and the ENTIRE variance
# ensemble) reuses the EXISTING univariate functions above completely
# unchanged, since a residual is just one ordinary series regardless of
# how many original series contributed to member 1's input (confirmed
# with the user -- "no need to go back to each individual time series
# again").
# ---------------------------------------------------------------------------

def align_detrended_series(detrend_infos: dict) -> dict:
    """
    Step 3 of the plan: different series can end up with different
    differencing orders (Section 17 is applied independently per
    series), which shrinks them by different amounts. Trims every
    series' detrended array down to the length of the SHORTEST one,
    removing observations from the EARLIEST (front) end only -- confirmed
    with the user: differencing itself already consumes points from the
    start, and the most recent data matters most for forecasting, so
    trimming from the front keeps every series aligned on their shared,
    most-recent history.

    detrend_infos: {column_name: detrend_info_dict} (each from detrend()).
    Returns a NEW dict {column_name: trimmed_detrended_array} -- the
    original detrend_info dicts (needed later for reconstruction) are
    left untouched.
    """
    min_len = min(len(info["detrended_series"]) for info in detrend_infos.values())
    return {col: info["detrended_series"][-min_len:] for col, info in detrend_infos.items()}


def _fit_first_member_multivariate(aligned_detrended: dict, target_col: str, predictor_cols: list,
                                    m: int, h: int, wavelet: str, level: int, target_diff_order: int,
                                    epochs: int, batch_size: int, filters: int) -> dict:
    """
    Builds and trains ONLY member 1 of the mean ensemble using every
    selected series (Steps 4-7 of the plan): wavelet-decompose each
    series separately, window each into raw (m, n_bands_i) blocks,
    horizontally concatenate across all series into one wide
    (m, total_bands) input per training record, then standardize
    per-band and train exactly like a univariate member (Section 16.3 /
    20's corrected standardization, reused identically).

    The target is drawn ONLY from the target series' own detrended array
    (never the predictors') using the same order-aware direct-strategy
    formula as everywhere else in this project.

    Returns a member dict with the SAME shape as `_fit_one_member`'s
    output (`components_full` is the horizontally-concatenated FULL
    components matrix across all series) -- this is what lets
    `predict_ensemble_sum` keep working completely unchanged for this
    multivariate member too.
    """
    all_cols = [target_col] + list(predictor_cols)

    per_series_components = {
        col: wavelet_decompose_components(aligned_detrended[col], wavelet, level) for col in all_cols
    }
    T = len(aligned_detrended[target_col])  # same for every series after alignment
    n_bands_per_series = {col: per_series_components[col].shape[1] for col in all_cols}
    n_samples = T - m - h + 1

    total_bands = sum(n_bands_per_series.values())
    concatenated_components_full = np.concatenate([per_series_components[col] for col in all_cols], axis=1)

    if n_samples <= 0:
        model = _build_2d_cnn(m, total_bands, filters=filters)
        return {"model": model, "components_full": concatenated_components_full, "residual": np.array([]),
                "m": m, "n_bands": total_bands,
                "x_means": np.zeros(total_bands), "x_stds": np.ones(total_bands),
                "y_mean": 0.0, "y_std": 1.0}

    per_series_X = {}
    for col in all_cols:
        components = per_series_components[col]
        X_i = np.zeros((n_samples, m, n_bands_per_series[col]))
        for i in range(n_samples):
            X_i[i] = components[i:i + m]
        per_series_X[col] = X_i
    X = np.concatenate([per_series_X[col] for col in all_cols], axis=2)  # (n_samples, m, total_bands)

    target_detrended = aligned_detrended[target_col]
    y = np.zeros(n_samples)
    for i in range(n_samples):
        window = target_detrended[i + m: i + m + h]
        y[i] = compute_direct_target(window, h, target_diff_order)

    x_means, x_stds = _per_band_stats(X)
    X_scaled = _apply_per_band_scaling(X, x_means, x_stds)
    y_mean, y_std = float(y.mean()), float(y.std()) or 1.0
    y_scaled = (y - y_mean) / y_std

    model = _build_2d_cnn(m, total_bands, filters=filters)
    model.fit(X_scaled.reshape(-1, m, total_bands, 1), y_scaled, epochs=epochs, batch_size=batch_size, verbose=0)
    preds_scaled = model.predict(X_scaled.reshape(-1, m, total_bands, 1), verbose=0).flatten()
    preds = preds_scaled * y_std + y_mean
    residual = y - preds

    return {"model": model, "components_full": concatenated_components_full, "residual": residual,
            "m": m, "n_bands": total_bands, "x_means": x_means, "x_stds": x_stds,
            "y_mean": y_mean, "y_std": y_std}


def kfold_cv_accuracy_multivariate(aligned_detrended: dict, target_col: str, predictor_cols: list,
                                    m: int, h: int, wavelet: str, level: int, target_diff_order: int,
                                    epochs: int, batch_size: int, filters: int, n_folds: int):
    """
    Multivariate counterpart of kfold_cv_accuracy -- same shuffled
    K-fold convention, scoped to a single member-1-style multivariate fit
    per fold (not the full ensemble), for the same cost reasons as the
    univariate version.
    """
    all_cols = [target_col] + list(predictor_cols)
    per_series_components = {
        col: wavelet_decompose_components(aligned_detrended[col], wavelet, level) for col in all_cols
    }
    T = len(aligned_detrended[target_col])
    n_bands_per_series = {col: per_series_components[col].shape[1] for col in all_cols}
    n_samples = T - m - h + 1
    if n_samples < n_folds * 2:
        return None

    per_series_X = {}
    for col in all_cols:
        components = per_series_components[col]
        X_i = np.zeros((n_samples, m, n_bands_per_series[col]))
        for i in range(n_samples):
            X_i[i] = components[i:i + m]
        per_series_X[col] = X_i
    X = np.concatenate([per_series_X[col] for col in all_cols], axis=2)
    total_bands = X.shape[2]

    target_detrended = aligned_detrended[target_col]
    y = np.zeros(n_samples)
    for i in range(n_samples):
        window = target_detrended[i + m: i + m + h]
        y[i] = compute_direct_target(window, h, target_diff_order)

    rng = np.random.default_rng(42)
    indices = rng.permutation(n_samples)
    fold_sizes = np.full(n_folds, n_samples // n_folds)
    fold_sizes[: n_samples % n_folds] += 1

    all_actual, all_pred = [], []
    start = 0
    for fold_size in fold_sizes:
        val_idx = indices[start:start + fold_size]
        train_idx = np.setdiff1d(indices, val_idx)
        start += fold_size
        if len(train_idx) < 5 or len(val_idx) < 1:
            continue
        X_train, y_train = X[train_idx], y[train_idx]
        x_means, x_stds = _per_band_stats(X_train)
        y_mean, y_std = float(y_train.mean()), float(y_train.std()) or 1.0

        model = _build_2d_cnn(m, total_bands, filters=filters)
        model.fit(
            _apply_per_band_scaling(X_train, x_means, x_stds).reshape(-1, m, total_bands, 1),
            (y_train - y_mean) / y_std, epochs=epochs, batch_size=batch_size, verbose=0,
        )
        preds_scaled = model.predict(
            _apply_per_band_scaling(X[val_idx], x_means, x_stds).reshape(-1, m, total_bands, 1), verbose=0
        ).flatten()
        preds = preds_scaled * y_std + y_mean

        all_actual.extend(y[val_idx].tolist())
        all_pred.extend(preds.tolist())

    if not all_actual:
        return None
    return forecasting_core.maape_accuracy_pct(np.array(all_actual), np.array(all_pred))


def dfs_forecast_multivariate(series_dict: dict, target_col: str, predictor_cols: list, horizon: int,
                               m: int = 10, wavelet: str = "db4", level: int = 2, epochs: int = 10,
                               batch_size: int = 8, filters: int = 8, max_members: int = 3,
                               white_noise_threshold: float = 0.05, confidence: str = "95%",
                               cv_folds: int = 2) -> dict:
    """
    Full multivariate DFS pipeline (Steps 1-10 of multivariate_dfs_plan.md).

    series_dict: {column_name: raw_series_array} -- must contain target_col
        and every name in predictor_cols, all the same length and time-
        aligned (as they naturally are, coming from the same uploaded file).

    Returns the same shape as dfs_forecast()'s output, plus a record of
    every series' own detrending choice for transparency:
    {
        "point": array[H], "lower": array[H], "upper": array[H],
        "n_mean_members": list[H], "n_var_members": list[H],
        "kfold_accuracy": float or None,
        "target_detrend_transform": str, "target_detrend_diff_order": int,
        "predictor_detrend_info": {col: {"transform": str, "diff_order": int}, ...},
    }
    """
    all_cols = [target_col] + list(predictor_cols)

    # Step 2: detrend every series independently
    detrend_infos = {col: detrend(series_dict[col]) for col in all_cols}

    # Step 3: align by trimming from the earliest end only
    aligned_detrended = align_detrended_series(detrend_infos)

    target_detrend_info = detrend_infos[target_col]
    target_diff_order = target_detrend_info["diff_order"]
    z = forecasting_core.Z_SCORES_BY_CONFIDENCE[confidence]

    kfold_accuracy = kfold_cv_accuracy_multivariate(
        aligned_detrended, target_col, predictor_cols, m=m, h=1, wavelet=wavelet, level=level,
        target_diff_order=target_diff_order, epochs=epochs, batch_size=batch_size, filters=filters,
        n_folds=cv_folds,
    )

    point = np.zeros(horizon)
    lower = np.zeros(horizon)
    upper = np.zeros(horizon)
    n_mean_members, n_var_members = [], []

    for h in range(1, horizon + 1):
        # Member 1: genuinely multivariate (Steps 4-7)
        member1 = _fit_first_member_multivariate(
            aligned_detrended, target_col, predictor_cols, m, h, wavelet, level, target_diff_order,
            epochs, batch_size, filters,
        )
        mean_members = [member1]
        residual = member1["residual"]

        # Member 2 onward: EXISTING univariate boosting code, unchanged (Step 8)
        if not (is_white_noise(residual, white_noise_threshold) or len(residual) == 0):
            remaining = train_boosting_ensemble(
                residual, m=m, h=h, wavelet=wavelet, level=level, diff_order=0,
                epochs=epochs, batch_size=batch_size, filters=filters,
                max_members=max(1, max_members - 1), threshold=white_noise_threshold, apply_abs=False,
            )
            mean_members.extend(remaining)

        mean_forecast = predict_ensemble_sum(mean_members, use_abs_sum=False)
        n_mean_members.append(len(mean_members))

        # Variance ensemble: entirely univariate from its very first member (Step 9)
        final_residual = mean_members[-1]["residual"]
        squared_residual = final_residual ** 2 if len(final_residual) else np.array([0.0])
        var_members = train_boosting_ensemble(
            squared_residual, m=m, h=h, wavelet=wavelet, level=level, diff_order=1,
            epochs=epochs, batch_size=batch_size, filters=filters,
            max_members=max_members, threshold=white_noise_threshold, apply_abs=True,
        )
        variance_forecast = predict_ensemble_sum(var_members, use_abs_sum=True)
        variance_forecast = max(variance_forecast, 0.0)
        n_var_members.append(len(var_members))

        std_forecast = np.sqrt(variance_forecast)

        # Step 10: reconstruct using ONLY the target series' own detrend_info
        point[h - 1] = reconstruct_direct_forecast(mean_forecast, target_detrend_info, h)
        lower[h - 1] = reconstruct_direct_forecast(mean_forecast - z * std_forecast, target_detrend_info, h)
        upper[h - 1] = reconstruct_direct_forecast(mean_forecast + z * std_forecast, target_detrend_info, h)

    return {
        "point": point, "lower": lower, "upper": upper,
        "n_mean_members": n_mean_members, "n_var_members": n_var_members,
        "kfold_accuracy": kfold_accuracy,
        "target_detrend_transform": target_detrend_info["transform"],
        "target_detrend_diff_order": target_diff_order,
        "predictor_detrend_info": {
            col: {"transform": detrend_infos[col]["transform"], "diff_order": detrend_infos[col]["diff_order"]}
            for col in predictor_cols
        },
    }


# ---------------------------------------------------------------------------
# Full orchestration (univariate)
# ---------------------------------------------------------------------------

def dfs_forecast(series: np.ndarray, horizon: int, m: int = 10, wavelet: str = "db4",
                  level: int = 2, epochs: int = 10, batch_size: int = 8, filters: int = 8,
                  max_members: int = 3, white_noise_threshold: float = 0.05,
                  confidence: str = "95%", cv_folds: int = 2) -> dict:
    """
    Full DFS pipeline for horizons 1..H: statistically-justified detrend
    (Section 17) -> for each h, build Ensemble_mean (boosting on the
    detrended mother series) and Ensemble_var (boosting on squared-then-
    absolute residuals) -> combine into point + interval (in the
    detrended scale) -> reconstruct back to the original scale via
    reconstruct_direct_forecast (Section 18, order-aware reverse chain).

    Also runs a single-member K-fold CV accuracy check (h=1) -- see
    kfold_cv_accuracy's docstring for why it's scoped to one member
    rather than the full ensemble.

    Returns {
        "point": array[H], "lower": array[H], "upper": array[H],
        "n_mean_members": list[H], "n_var_members": list[H],
        "kfold_accuracy": float or None,
        "detrend_transform": str, "detrend_diff_order": int,
    }
    """
    detrend_info = detrend(series)
    detrended_series = detrend_info["detrended_series"]
    diff_order = detrend_info["diff_order"]
    z = forecasting_core.Z_SCORES_BY_CONFIDENCE[confidence]

    kfold_accuracy = kfold_cv_accuracy(
        detrended_series, m=m, h=1, wavelet=wavelet, level=level, diff_order=diff_order,
        epochs=epochs, batch_size=batch_size, filters=filters, n_folds=cv_folds,
    )

    point = np.zeros(horizon)
    lower = np.zeros(horizon)
    upper = np.zeros(horizon)
    n_mean_members, n_var_members = [], []

    for h in range(1, horizon + 1):
        mean_members = train_boosting_ensemble(
            detrended_series, m=m, h=h, wavelet=wavelet, level=level, diff_order=diff_order,
            epochs=epochs, batch_size=batch_size, filters=filters,
            max_members=max_members, threshold=white_noise_threshold, apply_abs=False,
        )
        mean_forecast = predict_ensemble_sum(mean_members, use_abs_sum=False)
        n_mean_members.append(len(mean_members))

        # Variance ensemble starts from the FINAL mean-ensemble member's
        # squared residual (Section 11, step 10) -- always diff_order=1
        # (a squared-residual series has no transform/differencing
        # structure of its own to preserve).
        final_residual = mean_members[-1]["residual"]
        squared_residual = final_residual ** 2 if len(final_residual) else np.array([0.0])

        var_members = train_boosting_ensemble(
            squared_residual, m=m, h=h, wavelet=wavelet, level=level, diff_order=1,
            epochs=epochs, batch_size=batch_size, filters=filters,
            max_members=max_members, threshold=white_noise_threshold, apply_abs=True,
        )
        variance_forecast = predict_ensemble_sum(var_members, use_abs_sum=True)
        variance_forecast = max(variance_forecast, 0.0)  # guard against tiny negative float noise
        n_var_members.append(len(var_members))

        std_forecast = np.sqrt(variance_forecast)

        # Reconstruct point AND both interval bounds through the SAME
        # full reverse chain (Section 17/18) -- confirmed elsewhere in
        # this project that CI bounds must go through the identical
        # reverse chain as the point forecast, not be approximated
        # afterward in the original scale.
        point[h - 1] = reconstruct_direct_forecast(mean_forecast, detrend_info, h)
        lower[h - 1] = reconstruct_direct_forecast(mean_forecast - z * std_forecast, detrend_info, h)
        upper[h - 1] = reconstruct_direct_forecast(mean_forecast + z * std_forecast, detrend_info, h)

    return {
        "point": point, "lower": lower, "upper": upper,
        "n_mean_members": n_mean_members, "n_var_members": n_var_members,
        "kfold_accuracy": kfold_accuracy,
        "detrend_transform": detrend_info["transform"], "detrend_diff_order": diff_order,
    }


if __name__ == "__main__":
    # Standalone test block -- tiny synthetic series, minimal settings.
    np.random.seed(0)
    n = 60
    trend = np.linspace(100, 120, n)
    series = trend * np.exp(np.random.randn(n) * 0.01)

    result = dfs_forecast(
        series, horizon=2, m=5, wavelet="db2", level=1,
        epochs=3, batch_size=4, filters=4, max_members=2,
    )
    print("detrend transform:", result["detrend_transform"], "diff order:", result["detrend_diff_order"])
    print("point:", result["point"])
    print("lower:", result["lower"])
    print("upper:", result["upper"])
    print("mean members per horizon:", result["n_mean_members"])
    print("var members per horizon:", result["n_var_members"])
