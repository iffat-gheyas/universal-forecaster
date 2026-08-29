"""
Module 10: Statistical Detrending
--------------------------------------
Replaces the old fixed rule used across every forecasting method ("log
if positive, then always first-difference") with the statistically-
justified procedure agreed in univariate_forecasting_methodology_agreed.md
Section 17. Applies uniformly to LSTM, SARIMA-GARCH, N-BEATS, Transformer,
and DFS -- this module is the ONE place that logic lives, imported by
every method module rather than reimplemented per-method.

Procedure:
  1. Variance-stabilizing transform (or none), chosen via F-test: split
     the series into two halves, test whether their variances are
     significantly different. If not, no transform needed. If so, try
     candidates ("none" already failed, so: log, then sqrt) and keep the
     first one whose two halves are no longer significantly different.
  2. Differencing order (0, 1, or 2), chosen via Welch's t-test: split
     the (possibly transformed) series into two halves, test whether
     their MEANS are significantly different. If not, no differencing
     needed. If so, difference once and re-test; if still significant,
     difference again. Capped at second order regardless of outcome.

The reverse-transform undoes these two steps in EXACT REVERSE ORDER:
integrate (cumulative-sum) to undo whichever differencing order was
applied, THEN undo whichever variance-stabilizing transform was applied
(exponentiate for log, square for sqrt) -- same reverse-chain discipline
used throughout this project.
"""

import numpy as np
from scipy import stats


TRANSFORM_CANDIDATES = ["none", "log", "sqrt"]


def _apply_transform(series: np.ndarray, name: str) -> np.ndarray:
    if name == "none":
        return series.copy()
    if name == "log":
        return np.log(series)
    if name == "sqrt":
        return np.sqrt(series)
    raise ValueError(f"Unknown transform: {name!r}")


def _reverse_transform(series: np.ndarray, name: str) -> np.ndarray:
    if name == "none":
        return series.copy()
    if name == "log":
        return np.exp(series)
    if name == "sqrt":
        return np.square(series)
    raise ValueError(f"Unknown transform: {name!r}")


def reverse_transform_only(values: np.ndarray, transform: str) -> np.ndarray:
    """
    Public wrapper for undoing JUST the Section 17.1 variance-stabilizing
    transform (none/log/sqrt) -- no differencing/integration involved.
    For methods like SARIMA-GARCH that have their OWN principled way of
    handling differencing (SARIMA's own d/D order), only the transform
    needs to be reversed on our end; the model itself already returns
    forecasts back in its own (undifferenced) working scale.
    """
    return _reverse_transform(np.asarray(values, dtype=float), transform)


def _f_test_equal_variance(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided F-test for equal variances. Returns the p-value."""
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    if var_a == 0 or var_b == 0:
        return 0.0  # a zero-variance half is itself a red flag -- treat as "different"
    f_stat = var_a / var_b if var_a > var_b else var_b / var_a
    dof1 = len(a) - 1 if var_a > var_b else len(b) - 1
    dof2 = len(b) - 1 if var_a > var_b else len(a) - 1
    upper_tail = 1 - stats.f.cdf(f_stat, dof1, dof2)
    return float(2 * min(upper_tail, 1 - upper_tail))


def choose_variance_stabilizing_transform(series: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Section 17.1: split into two halves, F-test their variances. If not
    significantly different, no transform needed. Otherwise try log then
    sqrt (skipping any that produce non-finite values, e.g. log of a
    non-positive series) and keep the first that passes.

    Returns {"transform": str, "transformed_series": array, "p_value": float}.
    If nothing passes, returns the best-scoring candidate tried (highest
    p-value) rather than silently picking an arbitrary one.
    """
    n = len(series)
    half = n // 2

    results = []
    for name in TRANSFORM_CANDIDATES:
        try:
            transformed = _apply_transform(series, name)
        except Exception:
            continue
        if not np.all(np.isfinite(transformed)):
            continue
        first_half, second_half = transformed[:half], transformed[half:]
        if len(first_half) < 2 or len(second_half) < 2:
            continue
        p_value = _f_test_equal_variance(first_half, second_half)
        results.append({"transform": name, "transformed_series": transformed, "p_value": p_value})
        if p_value >= alpha:
            return results[-1]

    if not results:
        # Nothing was even computable (e.g. series too short) -- fall back to "none"
        return {"transform": "none", "transformed_series": series.copy(), "p_value": float("nan")}
    # No candidate passed the threshold -- return whichever had the highest p-value
    return max(results, key=lambda r: r["p_value"])


def choose_differencing_order(series: np.ndarray, alpha: float = 0.05, max_order: int = 2) -> dict:
    """
    Section 17.2: split into two halves, Welch's t-test on the means. If
    not significantly different, no differencing needed. Otherwise
    difference once and re-test; if still significant, difference again.
    Capped at max_order (default 2) regardless of outcome.

    Returns {"order": int, "differenced_series": array, "p_value": float}.
    """
    current = series.copy()
    p_value = float("nan")

    for order in range(0, max_order + 1):
        n = len(current)
        half = n // 2
        first_half, second_half = current[:half], current[half:]
        if len(first_half) < 2 or len(second_half) < 2:
            break
        t_stat, p_value = stats.ttest_ind(first_half, second_half, equal_var=False)
        if p_value >= alpha:
            return {"order": order, "differenced_series": current, "p_value": float(p_value)}
        if order < max_order:
            current = np.diff(current)

    return {"order": min(order, max_order), "differenced_series": current, "p_value": float(p_value)}


def detrend(series: np.ndarray, alpha: float = 0.05, max_diff_order: int = 2) -> dict:
    """
    Full Section 17 procedure: choose the variance-stabilizing transform,
    then choose the differencing order on top of it. Returns everything
    needed to reverse the process later:

    {
        "detrended_series": array,       -- the final target series
        "transform": str,                -- "none" | "log" | "sqrt"
        "diff_order": int,               -- 0, 1, or 2
        "transform_p_value": float,
        "diff_p_value": float,
        "pre_diff_series": array,        -- the transformed-but-not-yet-differenced
                                             series (needed to seed reverse integration)
    }
    """
    transform_result = choose_variance_stabilizing_transform(series, alpha=alpha)
    diff_result = choose_differencing_order(
        transform_result["transformed_series"], alpha=alpha, max_order=max_diff_order
    )
    return {
        "detrended_series": diff_result["differenced_series"],
        "transform": transform_result["transform"],
        "diff_order": diff_result["order"],
        "transform_p_value": transform_result["p_value"],
        "diff_p_value": diff_result["p_value"],
        "pre_diff_series": transform_result["transformed_series"],
    }


def reverse_detrend_cumulative(diff_forecast: np.ndarray, detrend_info: dict) -> np.ndarray:
    """
    Reverses `detrend()` for a vector of forecasted values in the
    DETRENDED (post-differencing) scale, using a DIRECT/cumulative-sum
    reconstruction (Section 10/16.2's convention): integrate order-by-
    order using the last known values from `pre_diff_series` as the
    seed(s), then undo the variance-stabilizing transform last.

    diff_forecast: array of length H, the forecasted values at the FINAL
        differencing order (e.g. if diff_order=1, these are forecasted
        first-differences).
    """
    order = detrend_info["diff_order"]
    pre_diff = detrend_info["pre_diff_series"]

    if order == 0:
        reconstructed = diff_forecast.copy()
    elif order == 1:
        last_level = pre_diff[-1]
        reconstructed = last_level + np.cumsum(diff_forecast)
    elif order == 2:
        # Second-order: forecast is of the SECOND difference. Integrate
        # once to get first-difference forecasts (seeded by the last
        # observed first-difference), then integrate again to get levels
        # (seeded by the last observed level).
        last_first_diff = pre_diff[-1] - pre_diff[-2]
        first_diff_forecast = last_first_diff + np.cumsum(diff_forecast)
        last_level = pre_diff[-1]
        reconstructed = last_level + np.cumsum(first_diff_forecast)
    else:
        raise ValueError(f"Unsupported diff_order: {order}")

    return _reverse_transform(reconstructed, detrend_info["transform"])


def compute_direct_target(future_window: np.ndarray, h: int, diff_order: int) -> float:
    """
    Given the h detrended-series values immediately following a reference
    window (future_window, length h), returns the scalar DIRECT-strategy
    target for horizon h, using the appropriate formula for diff_order
    (0, 1, or 2) -- see build_direct_cumulative_targets' docstring for
    the full derivation. Extracted as a standalone function so ANY
    method's dataset-builder can share the exact same target definition
    -- a 1-D lag window (LSTM/N-BEATS/Transformer) or a 2-D wavelet-
    component window (DFS) both need this same scalar-per-horizon rule,
    and keeping it in one place means the definition can't drift between
    methods.
    """
    if diff_order == 0:
        return float(future_window[-1])
    elif diff_order == 1:
        return float(np.sum(future_window))
    elif diff_order == 2:
        weights = np.arange(h, 0, -1)
        return float(np.sum(weights * future_window))
    else:
        raise ValueError(f"Unsupported diff_order: {diff_order}")


def build_direct_cumulative_targets(detrended_series: np.ndarray, k: int, h: int, diff_order: int) -> tuple:
    """
    Builds (X, y) for a DIRECT (non-recursive) horizon-h model, where y
    is defined so a SINGLE per-horizon reconstruction formula
    (reconstruct_direct_forecast) can recover the level forecast
    independently of any other horizon's model -- consistent with the
    project's direct-only multi-step strategy (Section 14): no model's
    output feeds into another model's input.

    X[i] = detrended_series[i : i+k]                     (k lag inputs)
    y[i] = compute_direct_target(detrended_series[i+k : i+k+h], h, diff_order)
    (see compute_direct_target's docstring / this module's earlier
    derivation notes for the order-0/1/2 formulas)
    """
    n = len(detrended_series)
    n_samples = n - k - h + 1
    if n_samples <= 0:
        return np.empty((0, k)), np.empty((0,))

    X = np.zeros((n_samples, k))
    y = np.zeros(n_samples)

    for i in range(n_samples):
        X[i] = detrended_series[i:i + k]
        window = detrended_series[i + k: i + k + h]
        y[i] = compute_direct_target(window, h, diff_order)

    return X, y


def reconstruct_direct_forecast(prediction: float, detrend_info: dict, h: int) -> float:
    """
    Reconstructs the ORIGINAL-scale forecast for a single horizon h,
    given that horizon's direct model's prediction (in the scale defined
    by build_direct_cumulative_targets for the SAME diff_order). Uses
    only seed values from detrend_info["pre_diff_series"] (the last
    known transformed level, and where needed the last known first
    difference) -- no dependency on any other horizon's forecast, so
    every horizon's model stays genuinely independent.
    """
    order = detrend_info["diff_order"]
    pre_diff = detrend_info["pre_diff_series"]

    if order == 0:
        level = prediction
    elif order == 1:
        level = pre_diff[-1] + prediction
    elif order == 2:
        last_level = pre_diff[-1]
        last_first_diff = pre_diff[-1] - pre_diff[-2]
        level = last_level + h * last_first_diff + prediction
    else:
        raise ValueError(f"Unsupported diff_order: {order}")

    return float(_reverse_transform(np.array([level]), detrend_info["transform"])[0])


if __name__ == "__main__":
    # Test 1: does the decision-making pick sensible choices on a series
    # that clearly needs both a variance-stabilizing transform and
    # differencing? (trending + multiplicative/heteroskedastic noise)
    np.random.seed(0)
    n = 100
    trend = np.linspace(50, 200, n)
    series = trend * np.exp(np.random.randn(n) * 0.05)

    result = detrend(series)
    print("Chosen transform:", result["transform"], f"(p={result['transform_p_value']:.4f})")
    print("Chosen diff order:", result["diff_order"], f"(p={result['diff_p_value']:.4f})")

    # Test 2: does the decision-making correctly detect when NEITHER a
    # transform NOR differencing is needed? (flat, homoskedastic noise)
    flat_series = 100 + np.random.randn(n) * 2
    flat_result = detrend(flat_series)
    print("\nFlat series -- transform:", flat_result["transform"],
          "diff order:", flat_result["diff_order"], "(expect 'none' and 0)")

    # Test 3: round-trip correctness of reverse_detrend_cumulative,
    # order=1, using a PERFECT-prediction scenario (feed in the TRUE
    # future differences, confirm reconstruction recovers the TRUE
    # future values exactly) -- this isolates the reverse-transform math
    # from any model's actual prediction accuracy.
    known_log_series = np.log(series)
    train_part = known_log_series[:-3]
    true_future = known_log_series[-3:]
    true_future_diffs = np.diff(np.concatenate([train_part[-1:], true_future]))

    fake_info = {"diff_order": 1, "transform": "log", "pre_diff_series": train_part}
    reconstructed = reverse_detrend_cumulative(true_future_diffs, fake_info)
    print("\nRound-trip test (order=1, perfect prediction):")
    print("Reconstructed:", reconstructed)
    print("True values:  ", series[-3:])
    assert np.allclose(reconstructed, series[-3:], rtol=1e-6), "Round-trip reconstruction FAILED"
    print("Round-trip check passed.")

    # Test 4: direct-strategy round-trip, all three diff orders. Rather
    # than reverse-engineer array indices (error-prone with shrinking
    # arrays from differencing), simulate the L -> D1 -> D2 recursive
    # relationship directly to get an unambiguous ground-truth future
    # level, then check the direct target formula + reconstruction
    # recover that exact value.
    print("\n--- Direct-strategy round-trip tests (all orders) ---")
    rng = np.random.default_rng(7)
    h = 4

    # Order 0: no differencing -- target IS the future level directly.
    L_t = 150.0
    future_levels = L_t + rng.normal(0, 5, size=h)  # arbitrary future levels
    detrended_window = future_levels  # order-0 "detrended series" = level series itself
    target = detrended_window[h - 1]  # per build_direct_cumulative_targets' order-0 rule
    fake_info0 = {"diff_order": 0, "transform": "none", "pre_diff_series": np.array([L_t])}
    recon0 = reconstruct_direct_forecast(target, fake_info0, h)
    print(f"order=0: reconstructed={recon0:.4f}  true={future_levels[-1]:.4f}")
    assert np.isclose(recon0, future_levels[-1], rtol=1e-6)

    # Order 1: simulate D1 sequence, derive true L(t+h) by direct summation.
    D1_t = 3.0
    future_D1 = D1_t + rng.normal(0, 1, size=h)  # D1(t+1)..D1(t+h)
    true_L_h1 = L_t + np.sum(future_D1)
    target1 = np.sum(future_D1)  # per build_direct_cumulative_targets' order-1 rule
    fake_info1 = {"diff_order": 1, "transform": "none", "pre_diff_series": np.array([L_t])}
    recon1 = reconstruct_direct_forecast(target1, fake_info1, h)
    print(f"order=1: reconstructed={recon1:.4f}  true={true_L_h1:.4f}")
    assert np.isclose(recon1, true_L_h1, rtol=1e-6)

    # Order 2: simulate D2 sequence, derive true L(t+h) by iterating the
    # actual recursive relationship (L += D1; D1 += D2, each step) --
    # this is the independent ground truth the weighted-sum formula must match.
    future_D2 = rng.normal(0, 0.5, size=h)  # D2(t+1)..D2(t+h)
    L_sim, D1_sim = L_t, D1_t
    for j in range(h):
        D1_sim = D1_sim + future_D2[j]
        L_sim = L_sim + D1_sim
    true_L_h2 = L_sim
    weights = np.arange(h, 0, -1)
    target2 = np.sum(weights * future_D2)  # per build_direct_cumulative_targets' order-2 rule
    fake_info2 = {"diff_order": 2, "transform": "none", "pre_diff_series": np.array([L_t - D1_t, L_t])}
    recon2 = reconstruct_direct_forecast(target2, fake_info2, h)
    print(f"order=2: reconstructed={recon2:.4f}  true={true_L_h2:.4f}")
    assert np.isclose(recon2, true_L_h2, rtol=1e-6)

    print("All direct-strategy round-trip checks passed.")

    # Test 5: confirm build_direct_cumulative_targets' target formula
    # matches what a hand-rolled sliding window computes, for order=1,
    # on a real array (sanity check on the indexing itself, separate
    # from the reconstruction math above).
    detrended_arr = np.array([1.0, 2.0, -1.0, 3.0, 0.5, 2.0, -0.5, 1.5, 1.0])
    k_test, h_test = 3, 2
    X5, y5 = build_direct_cumulative_targets(detrended_arr, k_test, h_test, diff_order=1)
    expected_y0 = detrended_arr[3] + detrended_arr[4]  # window ends at idx2, target = idx3+idx4
    assert np.isclose(y5[0], expected_y0)
    assert np.array_equal(X5[0], detrended_arr[0:3])
    print("build_direct_cumulative_targets indexing check passed.")

