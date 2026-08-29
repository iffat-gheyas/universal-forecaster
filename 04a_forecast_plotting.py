"""
Module 04a: Forecast Plotting (PNG export helpers)
-----------------------------------------------------
Small, focused module: takes data already computed elsewhere (this
module never fits models or transforms data) and renders matplotlib
figures for the three plot types the Forecasting section needs:
  - Historical series plot (raw uploaded data, before any modeling)
  - Test-set performance plot (observed vs. forecast vs. CI, relative
    time steps on the x-axis)
  - Out-of-sample forecast plot (forecasted values only, no history)

Each function returns a matplotlib Figure AND raw PNG bytes (via a
BytesIO buffer), so callers can both display it (st.pyplot) and offer it
as a download (st.download_button) without rendering twice.
"""

import io
import matplotlib
matplotlib.use("Agg")  # headless backend -- no display needed on a server
import matplotlib.pyplot as plt


def _fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def plot_historical_series(dates, values, series_name: str) -> tuple:
    """
    Plot the raw historical series against calendar dates (this is the
    one plot in this module that DOES use real dates, since there's no
    forecast horizon involved -- just the uploaded data as-is).
    Returns (matplotlib Figure, png_bytes).
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, values, color="#1f77b4", linewidth=1.5)
    ax.set_title(f"Historical Series — {series_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel(series_name)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, _fig_to_png_bytes(fig)


def plot_test_set_performance(actual, point_forecast, lower, upper, series_name: str) -> tuple:
    """
    Plot the withheld test set: observed values, point/mean forecast, and
    the 95% CI band -- all against RELATIVE time steps (t+1, t+2, ...),
    not calendar dates, per the agreed plan.
    Returns (matplotlib Figure, png_bytes).
    """
    horizon = len(actual)
    steps = list(range(1, horizon + 1))
    labels = [f"t+{s}" for s in steps]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(steps, actual, label="Observed", color="black", marker="o", linewidth=1.5)
    ax.plot(steps, point_forecast, label="Point forecast (mean)", color="#d62728",
             marker="o", linestyle="--", linewidth=1.5)
    ax.plot(steps, lower, label="Lower 95% CI", color="#1f77b4", linestyle=":", linewidth=1)
    ax.plot(steps, upper, label="Upper 95% CI", color="#1f77b4", linestyle=":", linewidth=1)
    ax.fill_between(steps, lower, upper, color="#1f77b4", alpha=0.1)
    ax.set_xticks(steps)
    ax.set_xticklabels(labels)
    ax.set_title(f"Test Set Performance — {series_name}")
    ax.set_xlabel("Relative time step")
    ax.set_ylabel(series_name)
    # Legend placed BELOW the axes rather than auto-placed inside it --
    # a flat/near-flat forecast line plus a wide shaded CI band (as
    # happens with some methods, e.g. a constant-mean SARIMA forecast)
    # can leave no truly empty spot inside the plot for auto-placement
    # to find. Below-axes placement is robust regardless of where the
    # observed/forecast lines fall.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, _fig_to_png_bytes(fig)


def plot_out_of_sample_forecast(point_forecast, lower, upper, series_name: str) -> tuple:
    """
    Plot ONLY the out-of-sample forecast -- no historical values, per the
    agreed plan. Three lines: forecasted mean, upper limit, lower limit,
    against relative time steps.
    Returns (matplotlib Figure, png_bytes).
    """
    horizon = len(point_forecast)
    steps = list(range(1, horizon + 1))
    labels = [f"t+{s}" for s in steps]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(steps, point_forecast, label="Forecast (mean)", color="#d62728",
             marker="o", linewidth=1.5)
    ax.plot(steps, lower, label="Lower 95% CI", color="#1f77b4", linestyle=":", linewidth=1)
    ax.plot(steps, upper, label="Upper 95% CI", color="#1f77b4", linestyle=":", linewidth=1)
    ax.fill_between(steps, lower, upper, color="#1f77b4", alpha=0.1)
    ax.set_xticks(steps)
    ax.set_xticklabels(labels)
    ax.set_title(f"Out-of-Sample Forecast — {series_name}")
    ax.set_xlabel("Relative time step")
    ax.set_ylabel(series_name)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, _fig_to_png_bytes(fig)
