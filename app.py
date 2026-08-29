"""
app.py — Universal Forecaster (Streamlit UI)
---------------------------------------------
Thin UI layer, in the same spirit as the Smart Data Cleaner's app.py:
this file renders widgets, collects input, and calls into numbered
modules for all real logic. It should never contain data-processing
logic itself — if you find yourself writing a real algorithm here,
it belongs in a module instead.

Navigation: a home screen with three sections (Data Collection, Data
Preprocessing, Forecasting). Each section is its own render function;
the dispatch at the bottom shows whichever is active. Navigation state
(`nav_page`) is intentionally NOT widget_/cache_ prefixed, so switching
sections never gets wiped by reset_session_state() -- only per-dataset
state should be cleared by that. There is no separate Visualization
section -- every forecasting method already plots inline (test-set or
out-of-sample chart, PNG download) as part of its own results, so a
standalone visualization module would just duplicate that.

Currently wired up:
  - Module 01 (01_data_ingestion.py) — stock data collection
  - Module 02 (02_data_processing.py) — generic upload, USER-EDITABLE
    column type detection, text-as-missing detection, frequency detection
    + confirmation, time-step grid construction, interpolation method
    ranking (9 popular methods, compared on the DETRENDED series via
    held-out masking) + cleaning (interpolate residual, add trend back),
    step-count aggregation, and CSV/Excel export of the processed result.
  - Module 03 (03_forecasting.py) — independent upload of a cleaned file,
    column type detection (editable), univariate/multivariate mode
    selection, historical series plot (PNG download), and forecast setup
    (horizon/confidence/frequency).
  - Module 05 (05_lstm_forecast.py) — LSTM, direct multi-step, deep
    ensemble intervals, auto (trial-and-error) or manual (k, units)
    hyperparameters, validate-against-holdout or forecast-future modes.
  - Module 06 (06_sarima_garch.py) — SARIMA-GARCH, auto (AIC search) or
    manual orders, same validate/forecast mode split.
  - Module 07 (07_dfs_forecast.py) — DFS, the novel wavelet + boosting
    CNN ensemble method from the user's paper. Wavelet/level/window are
    user-set (not auto-searched, given cost); ensemble size grows until
    residuals look like white noise, capped by a safety limit.
  - Module 08 (08_nbeats_forecast.py) — N-BEATS (generic variant), the
    M4-competition-winning-family architecture. Predicts the whole
    horizon in one multi-output pass per ensemble member (unlike LSTM/
    DFS's per-step training).
  - Module 09 (09_transformer_forecast.py) — Transformer encoder
    (self-attention + positional encoding), the most computationally
    expensive method here at full scale, kept lightweight by default.
    All five methods (LSTM, SARIMA-GARCH, DFS, N-BEATS, Transformer) now
    run independently; a shared K-fold CV leaderboard comparing them is
    the next planned step.

Not yet wired:
  - Module 07 — continual improvement / accuracy log

Session state discipline (reused from the cleaner):
  - Every interactive widget's key is prefixed `widget_`
  - Every cached result is prefixed `cache_`
  - `reset_session_state()` is the ONE place that clears per-dataset state.
    Any new per-dataset cache added later goes in THIS function.
"""

import importlib.util
import sys
import time as _time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

APP_DIR = Path(__file__).parent


def load_module(filename: str):
    """
    Load a numbered module (e.g. '01_data_ingestion.py') by file path.
    A normal `import` statement can't be used because filenames starting
    with a digit aren't valid Python identifiers.
    """
    module_name = filename.replace(".py", "").replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, APP_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


data_ingestion = load_module("01_data_ingestion.py")
data_processing = load_module("02_data_processing.py")
forecasting = load_module("03_forecasting.py")
forecast_plotting = load_module("04a_forecast_plotting.py")
lstm_forecast = load_module("05_lstm_forecast.py")
sarima_garch = load_module("06_sarima_garch.py")
dfs_forecast_module = load_module("07_dfs_forecast.py")
nbeats_forecast_module = load_module("08_nbeats_forecast.py")
transformer_forecast_module = load_module("09_transformer_forecast.py")


# ---------------------------------------------------------------------------
# Shared reset function — the ONE place per-dataset session state is cleared.
# ---------------------------------------------------------------------------
def reset_session_state():
    """Clear every widget_-prefixed key and any cached results."""
    for key in list(st.session_state.keys()):
        if key.startswith("widget_") or key.startswith("cache_"):
            del st.session_state[key]


# ---------------------------------------------------------------------------
# Page setup + navigation state
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Universal Forecaster", layout="wide")

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "home"


def go_to(page: str):
    st.session_state["nav_page"] = page


with st.sidebar:
    st.header("Session")
    if st.button("Home", key="widget_home_btn"):
        go_to("home")
        st.rerun()
    if st.button("Reset session", key="widget_reset_btn"):
        reset_session_state()
        st.rerun()


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------
def render_home():
    st.title("Universal Forecaster")
    st.caption("A modular toolkit: collect data, preprocess it, forecast it.")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Data Collection", key="widget_nav_collection", use_container_width=True):
            go_to("collection")
            st.rerun()
        st.caption("Pull historical stock prices via yfinance.")
    with c2:
        if st.button("Data Preprocessing", key="widget_nav_preprocessing", use_container_width=True):
            go_to("preprocessing")
            st.rerun()
        st.caption("Upload any dataset, clean it, make it forecast-ready.")
    with c3:
        if st.button("Forecasting", key="widget_nav_forecasting", use_container_width=True):
            go_to("forecasting")
            st.rerun()
        st.caption("Five methods, univariate and multivariate, with plots built in.")


# ---------------------------------------------------------------------------
# Data Collection (Module 1)
# ---------------------------------------------------------------------------
def render_data_collection():
    if st.button("← Back to Home", key="widget_back_from_collection"):
        go_to("home")
        st.rerun()

    st.header("Module 1 — Stock Data Collection")
    st.write(
        "Pulls historical daily OHLCV data via yfinance, with anti-blocking "
        "measures (batched requests, randomized delays, retry/backoff). "
        "Saves both raw and de-duplicated/sorted 'clean' versions."
    )

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        tickers_input = st.text_input(
            "Ticker(s), comma-separated (max 10)",
            value="TSLA, AAPL",
            key="widget_tickers",
            help="e.g. TSLA, AAPL, F, GM — each becomes its own column in the combined file.",
        )
    with col2:
        start_date = st.date_input(
            "Start date", value=pd.Timestamp("2023-01-01"), key="widget_start_date"
        )
    with col3:
        end_date = st.date_input(
            "End date", value=pd.Timestamp.today(), key="widget_end_date"
        )
    with col4:
        interval_label_to_code = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
        interval_label = st.selectbox(
            "Frequency", options=list(interval_label_to_code.keys()), key="widget_interval"
        )
        interval = interval_label_to_code[interval_label]

    save_excel = st.checkbox(
        "Also save as Excel (.xlsx)", value=True, key="widget_save_excel"
    )

    st.caption(
        f"Capped at {data_ingestion.MAX_TICKERS} tickers per run — this keeps request "
        "volume polite to the data source and protects our ability to keep collecting "
        "data from it in the future."
    )

    collect_clicked = st.button("Collect Data", key="widget_collect_btn", type="primary")
    COLLECT_COOLDOWN_SECONDS = 8

    if collect_clicked:
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        last_collect = st.session_state.get("cache_last_collect_time")
        now = _time.time()

        if not tickers:
            st.error("Enter at least one ticker symbol.")
        elif len(tickers) > data_ingestion.MAX_TICKERS:
            st.error(
                f"You entered {len(tickers)} tickers — the cap is "
                f"{data_ingestion.MAX_TICKERS} per run."
            )
        elif last_collect is not None and (now - last_collect) < COLLECT_COOLDOWN_SECONDS:
            wait = COLLECT_COOLDOWN_SECONDS - (now - last_collect)
            st.warning(f"Please wait {wait:.0f}s before collecting again.")
        else:
            with st.spinner(f"Collecting data for {', '.join(tickers)}..."):
                try:
                    results = data_ingestion.collect_stock_data(
                        tickers,
                        start=str(start_date),
                        end=str(end_date),
                        interval=interval,
                        save_excel=save_excel,
                    )
                    st.session_state["cache_collection_results"] = results
                    st.session_state["cache_last_collect_time"] = now
                    combined = data_ingestion.combine_price_series(results)
                    if not combined.empty:
                        st.session_state["cache_combined_table"] = combined
                        st.session_state["cache_combined_paths"] = data_ingestion.save_combined(
                            combined, save_excel=save_excel
                        )
                except Exception as e:
                    st.error(f"Data collection failed: {e}")

    results = st.session_state.get("cache_collection_results")
    if results:
        fetched = list(results.keys())
        missing = [
            t.strip().upper()
            for t in tickers_input.split(",")
            if t.strip() and t.strip().upper() not in fetched
        ]
        if missing:
            st.warning(f"No data collected for: {', '.join(missing)} — see logs for details.")

        for ticker, result in results.items():
            with st.expander(f"{ticker} — {len(result['data'])} rows", expanded=True):
                qc = result["quality_report"]
                m1, m2, m3 = st.columns(3)
                m1.metric("Rows", qc["rows"])
                m2.metric("Duplicate dates", qc["duplicate_dates"])
                m3.metric(
                    "Missing business days (approx.)",
                    qc["missing_business_days_approx"]
                    if qc["missing_business_days_approx"] is not None else "N/A",
                )
                st.dataframe(result["data"].head(10), use_container_width=True)

                dl1, dl2 = st.columns(2)
                with dl1:
                    csv_path = result["clean_paths"]["csv"]
                    st.download_button(
                        "Download clean CSV",
                        data=Path(csv_path).read_bytes(),
                        file_name=Path(csv_path).name,
                        mime="text/csv",
                        key=f"widget_dl_csv_{ticker}",
                    )
                with dl2:
                    xlsx_path = result["clean_paths"]["xlsx"]
                    if xlsx_path is not None:
                        st.download_button(
                            "Download clean Excel",
                            data=Path(xlsx_path).read_bytes(),
                            file_name=Path(xlsx_path).name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"widget_dl_xlsx_{ticker}",
                        )
                    else:
                        st.caption("Excel not generated (checkbox was off, or openpyxl missing).")

    combined_table = st.session_state.get("cache_combined_table")
    if combined_table is not None and not combined_table.empty:
        st.subheader("Combined Price Table (one column per ticker, aligned by date)")
        st.dataframe(combined_table.head(10), use_container_width=True)

        combined_paths = st.session_state.get("cache_combined_paths", {})
        dl1, dl2 = st.columns(2)
        with dl1:
            csv_path = combined_paths.get("csv")
            if csv_path is not None:
                st.download_button(
                    "Download combined CSV",
                    data=Path(csv_path).read_bytes(),
                    file_name=Path(csv_path).name,
                    mime="text/csv",
                    key="widget_dl_combined_csv",
                )
        with dl2:
            xlsx_path = combined_paths.get("xlsx")
            if xlsx_path is not None:
                st.download_button(
                    "Download combined Excel",
                    data=Path(xlsx_path).read_bytes(),
                    file_name=Path(xlsx_path).name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="widget_dl_combined_xlsx",
                )


# ---------------------------------------------------------------------------
# Data Preprocessing (Module 2) — built incrementally
# ---------------------------------------------------------------------------
def render_data_preprocessing():
    if st.button("← Back to Home", key="widget_back_from_preprocessing"):
        go_to("home")
        st.rerun()

    st.header("Module 2 — Data Preprocessing")
    st.write(
        "Upload any dataset with a date/time column and one or more value "
        "columns. Live now: upload, preview, column-type detection, "
        "text-as-missing handling, and frequency detection + confirmation. "
        "Coming next: converting calendar dates to time steps, ranking "
        "interpolation methods for missing values, a clean/download step, "
        "and step-count aggregation."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        key="widget_upload_file",
    )

    if uploaded_file is not None:
        if st.session_state.get("cache_uploaded_filename") != uploaded_file.name:
            try:
                df_uploaded = data_processing.load_uploaded_file(uploaded_file)
                st.session_state["cache_uploaded_df"] = df_uploaded
                st.session_state["cache_uploaded_filename"] = uploaded_file.name
                st.session_state.pop("cache_processed_df", None)
            except Exception as e:
                st.error(f"Could not read file: {e}")

    df_uploaded = st.session_state.get("cache_uploaded_df")

    if df_uploaded is not None:
        st.subheader(f"Preview — {st.session_state.get('cache_uploaded_filename')}")
        st.dataframe(df_uploaded.head(10), use_container_width=True)
        st.caption(f"{len(df_uploaded)} rows, {len(df_uploaded.columns)} columns")

        categories = data_processing.categorize_columns(df_uploaded)
        st.subheader("Detected column types")
        st.caption("Edit the 'Detected type' column below if anything looks wrong.")
        type_label_to_key = {"Datetime": "datetime", "Numeric Series": "numeric_series", "Other": "other"}
        key_to_type_label = {v: k for k, v in type_label_to_key.items()}
        cat_display = pd.DataFrame({
            "Column": list(categories.keys()),
            "Detected type": [key_to_type_label[categories[c]] for c in categories],
        })
        edited_cat_display = st.data_editor(
            cat_display,
            use_container_width=True,
            hide_index=True,
            disabled=["Column"],
            column_config={
                "Detected type": st.column_config.SelectboxColumn(
                    "Detected type",
                    options=["Datetime", "Numeric Series", "Other"],
                    required=True,
                )
            },
            key="widget_category_editor",
        )
        categories = {
            row["Column"]: type_label_to_key[row["Detected type"]]
            for _, row in edited_cat_display.iterrows()
        }

        date_candidates = [c for c, cat in categories.items() if cat == "datetime"]
        numeric_candidates = [c for c, cat in categories.items() if cat == "numeric_series"]

        if len(date_candidates) == 1:
            date_col = date_candidates[0]
            st.caption(f"Date/time column detected automatically: **{date_col}**")
        else:
            if not date_candidates:
                st.warning("Couldn't confidently detect a date column — please select one.")
            else:
                st.info(
                    f"Multiple possible date columns found ({', '.join(date_candidates)}) "
                    "— please confirm which one."
                )
            date_col = st.selectbox(
                "Which column is the date/timestamp?",
                options=list(df_uploaded.columns),
                key="widget_date_col",
            )

        value_cols = st.multiselect(
            "Which column(s) are the time series values to process/forecast?",
            options=[c for c in df_uploaded.columns if c != date_col],
            default=[c for c in numeric_candidates if c != date_col],
            key="widget_value_cols",
        )

        if date_col and value_cols:
            coerced_df, coercion_log = data_processing.coerce_numeric_with_missing(
                df_uploaded, value_cols
            )
            if coercion_log:
                st.warning(
                    "Some values in your time series columns were text, not "
                    "numbers, and are now treated as missing:"
                )
                for entry in coercion_log:
                    st.write(f"- {entry}")

            freq_info = data_processing.infer_native_frequency(df_uploaded[date_col])
            st.subheader("Sampling frequency")
            st.write(
                f"**Detected native frequency:** {freq_info['label']}  "
                f"(median gap: {freq_info['median_gap']})"
            )

            freq_options = list(data_processing.FREQ_LABEL_TO_CODE.keys())
            default_index = (
                freq_options.index(freq_info["label"])
                if freq_info["label"] in freq_options else 0
            )
            confirmed_freq = st.selectbox(
                "Confirm the sampling frequency (change it if the detection above looks wrong)",
                options=freq_options,
                index=default_index,
                key="widget_confirmed_freq",
            )

            st.session_state["cache_confirmed_freq"] = confirmed_freq
            st.session_state["cache_coerced_df"] = coerced_df
            st.session_state["cache_date_col"] = date_col
            st.session_state["cache_value_cols"] = value_cols

            if st.button("Apply Frequency & Build Time-Step Grid", key="widget_build_grid_btn", type="primary"):
                try:
                    grid_df, grid_log = data_processing.build_regular_grid(
                        coerced_df, date_col, value_cols, confirmed_freq
                    )
                    st.session_state["cache_grid_df"] = grid_df
                    st.session_state["cache_grid_log"] = grid_log
                    # A new grid invalidates any downstream cleaning/aggregation
                    # from a previous run -- clear it here, in the one shared place.
                    st.session_state.pop("cache_ranking", None)
                    st.session_state.pop("cache_cleaned_df", None)
                    st.session_state.pop("cache_agg_df", None)
                except Exception as e:
                    st.error(f"Could not build the time-step grid: {e}")

    grid_df = st.session_state.get("cache_grid_df")
    if grid_df is not None:
        value_cols = st.session_state["cache_value_cols"]
        date_col = st.session_state["cache_date_col"]

        st.subheader("Time-Step Grid")
        st.write(
            "Calendar dates are replaced by an evenly-spaced `time_step` "
            "(1, 2, 3, ...); the original date is kept alongside it for "
            "reference/audit purposes."
        )
        with st.expander("What was done", expanded=True):
            for entry in st.session_state.get("cache_grid_log", []):
                st.write(f"- {entry}")
        st.dataframe(grid_df.head(10), use_container_width=True)

        n_missing = int(grid_df[value_cols].isna().any(axis=1).sum())

        if n_missing == 0:
            st.success("No missing values or missing time steps — nothing to interpolate.")
            st.session_state["cache_cleaned_df"] = grid_df
        else:
            st.subheader("Missing Value / Missing Time Step Handling")
            st.write(
                f"{n_missing} time step(s) need interpolation. The app removes each "
                "column's long-term linear trend first, then compares several "
                "interpolation methods on the detrended values — hiding known "
                "points and measuring how accurately each method reconstructs "
                "them — so the ranking reflects local reconstruction quality, "
                "not just which method happens to track the overall trend."
            )

            if st.button("Rank Interpolation Methods", key="widget_rank_btn"):
                with st.spinner("Testing interpolation methods against held-out known values..."):
                    ranking = data_processing.rank_interpolation_methods_multi(grid_df, value_cols)
                    st.session_state["cache_ranking"] = ranking

            ranking = st.session_state.get("cache_ranking")
            if ranking is not None and not ranking.empty:
                st.dataframe(ranking, use_container_width=True, hide_index=True)

                method_options = ranking["method"].tolist()
                chosen_method = st.selectbox(
                    "Interpolation method to use (defaults to the best-ranked method above)",
                    options=method_options,
                    index=0,
                    key="widget_chosen_method",
                )

                if st.button("Clean Data", key="widget_clean_btn", type="primary"):
                    cleaned_df, clean_log = data_processing.apply_interpolation(
                        grid_df, value_cols, chosen_method
                    )
                    st.session_state["cache_cleaned_df"] = cleaned_df
                    st.session_state["cache_clean_log"] = clean_log
                    st.session_state.pop("cache_agg_df", None)

    cleaned_df = st.session_state.get("cache_cleaned_df")
    if cleaned_df is not None:
        value_cols = st.session_state["cache_value_cols"]
        date_col = st.session_state["cache_date_col"]

        st.subheader("Cleaned Data")
        clean_log = st.session_state.get("cache_clean_log")
        if clean_log:
            with st.expander("What was done", expanded=False):
                for entry in clean_log:
                    st.write(f"- {entry}")
        st.dataframe(cleaned_df.head(10), use_container_width=True)
        st.caption(f"{len(cleaned_df)} time steps, {int(cleaned_df[value_cols].isna().sum().sum())} remaining missing value(s).")

        st.subheader("Optional: Combine Consecutive Time Steps")
        st.write(
            "Aggregate every N consecutive time steps into one — by row count, "
            "not by calendar unit. This is separate from the frequency chosen "
            "above; use it if you want coarser steps than what the calendar "
            "frequency alone gives you."
        )
        acol1, acol2, acol3 = st.columns([1, 1, 1])
        with acol1:
            step_size = st.number_input(
                "Combine every N time steps", min_value=1, value=1,
                key="widget_step_size",
            )
        with acol2:
            agg_method_step = st.selectbox(
                "Aggregation method", options=data_processing.AGG_METHODS,
                key="widget_step_agg_method",
            )
        with acol3:
            st.write("")
            st.write("")
            if st.button("Aggregate Steps", key="widget_aggregate_btn"):
                agg_df, agg_log = data_processing.aggregate_by_step_count(
                    cleaned_df, date_col, value_cols, int(step_size), agg_method_step
                )
                st.session_state["cache_agg_df"] = agg_df
                st.session_state["cache_agg_log"] = agg_log

        agg_df = st.session_state.get("cache_agg_df")
        if agg_df is not None:
            with st.expander("What was done", expanded=False):
                for entry in st.session_state.get("cache_agg_log", []):
                    st.write(f"- {entry}")
            st.dataframe(agg_df.head(10), use_container_width=True)

        final_df = agg_df if agg_df is not None else cleaned_df

        st.subheader("Download")
        csv_bytes = final_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download processed data (CSV)",
            data=csv_bytes,
            file_name="processed_data.csv",
            mime="text/csv",
            key="widget_dl_final_csv",
        )
        try:
            import io
            xlsx_buffer = io.BytesIO()
            final_df.to_excel(xlsx_buffer, index=False, engine="openpyxl")
            st.download_button(
                "Download processed data (Excel)",
                data=xlsx_buffer.getvalue(),
                file_name="processed_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="widget_dl_final_xlsx",
            )
        except ImportError:
            st.caption("Excel export needs openpyxl (already in requirements.txt).")


def render_forecasting():
    if st.button("← Back to Home", key="widget_back_from_forecasting"):
        go_to("home")
        st.rerun()

    st.header("Module 3 — Forecasting")
    st.write(
        "Upload a cleaned, evenly-spaced dataset (e.g. the output of Module 2) "
        "to set up a forecast. This section works independently from Data "
        "Preprocessing — it has its own upload, so you can bring in any "
        "already-cleaned file without needing to re-run preprocessing here."
    )

    uploaded_file = st.file_uploader(
        "Upload cleaned CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        key="widget_fc_upload_file",
    )

    if uploaded_file is not None:
        if st.session_state.get("cache_fc_uploaded_filename") != uploaded_file.name:
            try:
                df_fc = data_processing.load_uploaded_file(uploaded_file)
                st.session_state["cache_fc_uploaded_df"] = df_fc
                st.session_state["cache_fc_uploaded_filename"] = uploaded_file.name
                # New file -> clear any downstream mode/target selections
                for k in ["cache_fc_mode", "cache_fc_target_col", "cache_fc_predictor_cols"]:
                    st.session_state.pop(k, None)
            except Exception as e:
                st.error(f"Could not read file: {e}")

    df_fc = st.session_state.get("cache_fc_uploaded_df")

    if df_fc is not None:
        st.subheader(f"Preview — {st.session_state.get('cache_fc_uploaded_filename')}")
        st.dataframe(df_fc.head(10), use_container_width=True)
        st.caption(f"{len(df_fc)} rows, {len(df_fc.columns)} columns")

        categories = data_processing.categorize_columns(df_fc)
        st.subheader("Detected column types")
        st.caption("Edit the 'Detected type' column below if anything looks wrong.")
        type_label_to_key = {
            "Datetime": "datetime", "Numeric Series": "numeric_series",
            "Step Index": "step_index", "Other": "other",
        }
        key_to_type_label = {v: k for k, v in type_label_to_key.items()}
        cat_display = pd.DataFrame({
            "Column": list(categories.keys()),
            "Detected type": [key_to_type_label[categories[c]] for c in categories],
        })
        edited_cat_display = st.data_editor(
            cat_display,
            use_container_width=True,
            hide_index=True,
            disabled=["Column"],
            column_config={
                "Detected type": st.column_config.SelectboxColumn(
                    "Detected type",
                    options=["Datetime", "Numeric Series", "Step Index", "Other"],
                    required=True,
                )
            },
            key="widget_fc_category_editor",
        )
        categories = {
            row["Column"]: type_label_to_key[row["Detected type"]]
            for _, row in edited_cat_display.iterrows()
        }

        date_candidates = [c for c, cat in categories.items() if cat == "datetime"]
        numeric_candidates = [c for c, cat in categories.items() if cat == "numeric_series"]

        if len(date_candidates) == 1:
            date_col = date_candidates[0]
            st.caption(f"Date/time column detected automatically: **{date_col}**")
        else:
            if not date_candidates:
                st.warning("Couldn't confidently detect a date column — please select one.")
            else:
                st.info(
                    f"Multiple possible date columns found ({', '.join(date_candidates)}) "
                    "— please confirm which one."
                )
            date_col = st.selectbox(
                "Which column is the date/timestamp?",
                options=list(df_fc.columns),
                key="widget_fc_date_col",
            )

        value_cols = st.multiselect(
            "Which column(s) are time series values available for forecasting?",
            options=[c for c in df_fc.columns if c != date_col],
            default=[c for c in numeric_candidates if c != date_col],
            key="widget_fc_value_cols",
        )

        st.session_state["cache_fc_date_col"] = date_col
        st.session_state["cache_fc_value_cols"] = value_cols

        if value_cols:
            st.subheader("Historical Series Plot")
            plot_col_choice = st.selectbox(
                "Which series to plot?", options=value_cols, key="widget_fc_hist_plot_col",
            )
            if st.button("Generate Historical Plot", key="widget_fc_hist_plot_btn"):
                hist_dates = pd.to_datetime(df_fc[date_col])
                hist_values = df_fc[plot_col_choice]
                fig, png_bytes = forecast_plotting.plot_historical_series(
                    hist_dates, hist_values, plot_col_choice
                )
                st.session_state["cache_fc_hist_plot_png"] = png_bytes
                st.session_state["cache_fc_hist_plot_col_used"] = plot_col_choice

            hist_png = st.session_state.get("cache_fc_hist_plot_png")
            if hist_png is not None:
                st.image(hist_png, use_container_width=True)
                st.download_button(
                    "Download historical plot (.png)",
                    data=hist_png,
                    file_name=f"{st.session_state.get('cache_fc_hist_plot_col_used', 'series')}_historical.png",
                    mime="image/png",
                    key="widget_fc_hist_plot_dl",
                )

        if value_cols:
            st.subheader("What would you like to do?")
            mode = st.radio(
                "Choose an analysis mode",
                options=forecasting.MODE_OPTIONS,
                key="widget_fc_mode",
                help=(
                    "Univariate: forecast a series using only its own history. "
                    "Multivariate: forecast one target series using other series "
                    "as additional predictors (e.g. Tesla + other automaker stocks)."
                ),
            )

            if mode == "Multivariate time series analysis" and not forecasting.can_run_multivariate(value_cols):
                st.warning(
                    f"Multivariate mode needs at least {forecasting.MIN_MULTIVARIATE_SERIES} "
                    "selected value columns (one target + at least one predictor). "
                    "Select more columns above, or switch to univariate."
                )
            else:
                if len(value_cols) == 1:
                    target_col = value_cols[0]
                    st.caption(f"Target series: **{target_col}**")
                else:
                    target_col = st.selectbox(
                        "Which column is the target series to forecast?",
                        options=value_cols,
                        key="widget_fc_target_col",
                    )

                predictor_cols = forecasting.get_predictor_columns(mode, target_col, value_cols)
                if mode == "Multivariate time series analysis":
                    predictor_cols = st.multiselect(
                        "Which column(s) should be used as predictors?",
                        options=[c for c in value_cols if c != target_col],
                        default=predictor_cols,
                        key="widget_fc_predictor_cols",
                    )
                    st.caption(f"Forecasting **{target_col}** using **{', '.join(predictor_cols) if predictor_cols else '(no predictors selected)'}** as predictor(s).")
                else:
                    st.caption(f"Forecasting **{target_col}** using only its own history (no other series).")

                st.session_state["cache_fc_mode"] = mode
                st.session_state["cache_fc_target_col"] = target_col
                st.session_state["cache_fc_predictor_cols"] = predictor_cols

                if mode == "Multivariate time series analysis":
                    st.subheader("Forecast Setup")
                    fcol1, fcol2 = st.columns(2)
                    with fcol1:
                        horizon = st.number_input(
                            "How many out-of-sample steps to forecast (t+1 ... t+H)?",
                            min_value=1, value=10, key="widget_mv_horizon",
                        )
                    with fcol2:
                        confidence = st.selectbox(
                            "Confidence level for the forecast interval",
                            options=list(forecasting.Z_SCORES_BY_CONFIDENCE.keys()),
                            index=1, key="widget_mv_confidence",
                        )

                    freq_info = data_processing.infer_native_frequency(df_fc[date_col])
                    freq_options = list(data_processing.FREQ_LABEL_TO_CODE.keys())
                    # Same as univariate mode: only used to date output rows
                    # for genuinely-future forecasts, no effect on the
                    # computation itself, so no need to ask for confirmation.
                    confirmed_freq = freq_info["label"] if freq_info["label"] in freq_options else "Daily"

                    if not predictor_cols:
                        st.warning("Select at least one predictor column above to continue.")
                    else:
                        needed_cols = [date_col, target_col] + predictor_cols
                        mv_df = df_fc[needed_cols].dropna().sort_values(date_col)
                        mv_dates = pd.to_datetime(mv_df[date_col])
                        series_dict = {col: mv_df[col].values for col in [target_col] + predictor_cols}

                        st.session_state["cache_mv_series_dict"] = series_dict
                        st.session_state["cache_mv_dates"] = mv_dates
                        st.session_state["cache_mv_confirmed_freq"] = confirmed_freq

                        st.info(
                            "All five methods now support multivariate forecasting: "
                            "DFS, LSTM, SARIMA-GARCH, N-BEATS, and Transformer."
                        )

                        st.subheader("DFS Forecast (multivariate)")
                        st.write(
                            f"Forecasting **{target_col}** using its own history plus "
                            f"**{', '.join(predictor_cols)}** as predictor(s). Every "
                            "series is detrended independently, wavelet-decomposed "
                            "separately, then combined into one wide input for the "
                            "first ensemble member — member 2 onward and the entire "
                            "variance ensemble work exactly like the univariate version."
                        )

                        mvcol1, mvcol2, mvcol3 = st.columns(3)
                        with mvcol1:
                            _mv_all_wavelets = sorted(__import__("pywt").wavelist(kind="discrete"))
                            _mv_default_idx = _mv_all_wavelets.index("db4") if "db4" in _mv_all_wavelets else 0
                            mv_dfs_wavelet = st.selectbox(
                                "Wavelet", options=_mv_all_wavelets, index=_mv_default_idx,
                                key="widget_mv_dfs_wavelet",
                            )
                        with mvcol2:
                            mv_dfs_level = st.number_input(
                                "Decomposition level", min_value=1, value=2, key="widget_mv_dfs_level",
                            )
                        with mvcol3:
                            mv_dfs_m = st.number_input(
                                "Window size (m)", min_value=2, value=10, key="widget_mv_dfs_m",
                            )

                        mvcol4, mvcol5, mvcol6 = st.columns(3)
                        with mvcol4:
                            mv_dfs_epochs = st.number_input(
                                "Training epochs per CNN", min_value=1, value=15, key="widget_mv_dfs_epochs",
                            )
                        with mvcol5:
                            mv_dfs_max_members = st.number_input(
                                "Max ensemble members (safety cap)", min_value=1, value=3,
                                key="widget_mv_dfs_max_members",
                            )
                        with mvcol6:
                            mv_dfs_threshold = st.number_input(
                                "White-noise threshold (mean |residual|)", min_value=0.0001,
                                value=0.05, step=0.01, key="widget_mv_dfs_threshold",
                            )

                        mv_dfs_cv_folds = st.number_input(
                            "K-fold CV folds (member-1-only check)", min_value=2, value=2,
                            key="widget_mv_dfs_cv_folds",
                        )

                        st.caption(
                            f"This trains up to {int(mv_dfs_max_members)} CNNs for the mean "
                            f"ensemble AND up to {int(mv_dfs_max_members)} more for the "
                            f"variance ensemble, PER horizon step ({int(horizon)} steps) — "
                            "can take several minutes, and scales with the number of "
                            "predictors selected."
                        )

                        mv_run_mode = st.radio(
                            "What should this run do?",
                            options=[
                                "Validate against held-out actual data (recommended)",
                                "Forecast genuinely unknown future",
                            ],
                            key="widget_mv_run_mode",
                        )
                        mv_validate = mv_run_mode.startswith("Validate")
                        st.caption(f"Using H = **{int(horizon)}** (set above, in 'Forecast Setup').")

                        if st.button("Run Multivariate DFS Forecast", key="widget_mv_dfs_run_btn", type="primary"):
                            with st.spinner("Training multivariate DFS ensembles — this may take several minutes..."):
                                try:
                                    if mv_validate:
                                        mv_train_dict = {col: arr[:-int(horizon)] for col, arr in series_dict.items()}
                                        mv_actual_holdout = series_dict[target_col][-int(horizon):]
                                    else:
                                        mv_train_dict = series_dict
                                        mv_actual_holdout = None

                                    mv_result = dfs_forecast_module.dfs_forecast_multivariate(
                                        mv_train_dict, target_col, predictor_cols, horizon=int(horizon),
                                        m=int(mv_dfs_m), wavelet=mv_dfs_wavelet, level=int(mv_dfs_level),
                                        epochs=int(mv_dfs_epochs), max_members=int(mv_dfs_max_members),
                                        white_noise_threshold=float(mv_dfs_threshold),
                                        confidence=confidence, cv_folds=int(mv_dfs_cv_folds),
                                    )

                                    if mv_validate:
                                        mv_final_df = pd.DataFrame({
                                            "relative_step": [f"t+{i}" for i in range(1, int(horizon) + 1)],
                                            f"{target_col}_actual": mv_actual_holdout,
                                            f"{target_col}_point_forecast": mv_result["point"],
                                            f"{target_col}_lower_bound": mv_result["lower"],
                                            f"{target_col}_upper_bound": mv_result["upper"],
                                        })
                                    else:
                                        freq_code = data_processing.FREQ_LABEL_TO_CODE[confirmed_freq]
                                        future_dates = pd.date_range(
                                            start=mv_dates.iloc[-1], periods=int(horizon) + 1, freq=freq_code
                                        )[1:]
                                        mv_final_df = pd.DataFrame({
                                            "Date": future_dates,
                                            f"{target_col}_point_forecast": mv_result["point"],
                                            f"{target_col}_lower_bound": mv_result["lower"],
                                            f"{target_col}_upper_bound": mv_result["upper"],
                                        })

                                    st.session_state["cache_mv_dfs_result"] = mv_final_df
                                    st.session_state["cache_mv_dfs_validate_mode"] = mv_validate
                                    st.session_state["cache_mv_dfs_n_mean"] = mv_result["n_mean_members"]
                                    st.session_state["cache_mv_dfs_n_var"] = mv_result["n_var_members"]
                                    st.session_state["cache_mv_dfs_kfold_acc"] = mv_result["kfold_accuracy"]
                                    st.session_state["cache_mv_dfs_target_detrend"] = (
                                        mv_result["target_detrend_transform"], mv_result["target_detrend_diff_order"]
                                    )
                                    st.session_state["cache_mv_dfs_predictor_detrend"] = mv_result["predictor_detrend_info"]
                                except Exception as e:
                                    st.error(f"Multivariate DFS forecast failed: {e}")

                        mv_dfs_result_df = st.session_state.get("cache_mv_dfs_result")
                        if mv_dfs_result_df is not None:
                            mv_dfs_was_validate = st.session_state.get("cache_mv_dfs_validate_mode", False)
                            mv_dfs_acc = st.session_state.get("cache_mv_dfs_kfold_acc")
                            target_transform, target_order = st.session_state.get(
                                "cache_mv_dfs_target_detrend", (None, None)
                            )
                            st.write(
                                f"**{target_col}** detrending: transform=**{target_transform}**, "
                                f"diff order=**{target_order}**"
                            )
                            pred_detrend = st.session_state.get("cache_mv_dfs_predictor_detrend", {})
                            if pred_detrend:
                                pred_lines = "; ".join(
                                    f"{c}: transform={v['transform']}, diff order={v['diff_order']}"
                                    for c, v in pred_detrend.items()
                                )
                                st.caption(f"Predictor detrending: {pred_lines}")
                            st.write(
                                f"Mean-ensemble members per horizon step: "
                                f"**{st.session_state.get('cache_mv_dfs_n_mean')}**, "
                                f"variance-ensemble members per horizon step: "
                                f"**{st.session_state.get('cache_mv_dfs_n_var')}**"
                            )
                            if mv_dfs_acc is not None:
                                st.metric("Average K-fold CV accuracy", f"{mv_dfs_acc:.1f}%")
                            st.dataframe(mv_dfs_result_df, use_container_width=True, hide_index=True)

                            if mv_dfs_was_validate:
                                mv_fig, mv_png_bytes = forecast_plotting.plot_test_set_performance(
                                    mv_dfs_result_df[f"{target_col}_actual"].values,
                                    mv_dfs_result_df[f"{target_col}_point_forecast"].values,
                                    mv_dfs_result_df[f"{target_col}_lower_bound"].values,
                                    mv_dfs_result_df[f"{target_col}_upper_bound"].values,
                                    target_col,
                                )
                            else:
                                mv_fig, mv_png_bytes = forecast_plotting.plot_out_of_sample_forecast(
                                    mv_dfs_result_df[f"{target_col}_point_forecast"].values,
                                    mv_dfs_result_df[f"{target_col}_lower_bound"].values,
                                    mv_dfs_result_df[f"{target_col}_upper_bound"].values,
                                    target_col,
                                )
                            st.image(mv_png_bytes, use_container_width=True)

                            mvdl1, mvdl2, mvdl3 = st.columns(3)
                            with mvdl1:
                                st.download_button(
                                    "Download plot (.png)", data=mv_png_bytes,
                                    file_name=f"{target_col}_multivariate_dfs.png", mime="image/png",
                                    key="widget_mv_dfs_dl_png",
                                )
                            with mvdl2:
                                mv_csv_bytes = mv_dfs_result_df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "Download forecast (CSV)", data=mv_csv_bytes,
                                    file_name=f"{target_col}_multivariate_dfs.csv", mime="text/csv",
                                    key="widget_mv_dfs_dl_csv",
                                )
                            with mvdl3:
                                import io
                                mv_xlsx_buffer = io.BytesIO()
                                mv_dfs_result_df.to_excel(mv_xlsx_buffer, index=False, engine="openpyxl")
                                st.download_button(
                                    "Download forecast (Excel)", data=mv_xlsx_buffer.getvalue(),
                                    file_name=f"{target_col}_multivariate_dfs.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="widget_mv_dfs_dl_xlsx",
                                )

                        st.divider()
                        st.subheader("LSTM Forecast (multivariate)")
                        st.write(
                            f"Forecasting **{target_col}** using its own history plus "
                            f"**{', '.join(predictor_cols)}** as predictor(s). Every "
                            "series is detrended independently; at each lag position, "
                            "the target's value AND every predictor's value (same time "
                            "step) are combined into one feature vector — the LSTM's "
                            "natural input shape for multiple series."
                        )

                        mv_lstm_hp_mode = st.radio(
                            "How should the lag count (k) and hidden units be chosen?",
                            options=["Trial and error (auto-search)", "I'll specify them"],
                            key="widget_mv_lstm_hp_mode",
                        )
                        mv_lstm_auto = mv_lstm_hp_mode.startswith("Trial")

                        mlcol1, mlcol2, mlcol3 = st.columns(3)
                        with mlcol1:
                            mv_lstm_ensemble_size = st.number_input(
                                "Ensemble size", min_value=2, value=5, key="widget_mv_lstm_ensemble_size",
                            )
                        with mlcol2:
                            mv_lstm_epochs = st.number_input(
                                "Training epochs", min_value=1, value=15, key="widget_mv_lstm_epochs",
                            )

                        if mv_lstm_auto:
                            with mlcol3:
                                mv_lstm_cv_folds = st.number_input(
                                    "K-fold CV folds", min_value=1, value=2, key="widget_mv_lstm_cv_folds",
                                )
                            mmcol1, mmcol2 = st.columns(2)
                            with mmcol1:
                                mv_lstm_max_k = st.number_input(
                                    "Max lag count to try", min_value=1, value=8, key="widget_mv_lstm_max_k",
                                )
                            with mmcol2:
                                mv_lstm_units_text = st.text_input(
                                    "Hidden unit candidates to try (comma-separated)",
                                    value="8, 16, 32", key="widget_mv_lstm_units_candidates",
                                )
                            mv_lstm_unit_candidates = [
                                int(u.strip()) for u in mv_lstm_units_text.split(",") if u.strip()
                            ]
                            mv_lstm_k_manual, mv_lstm_units_manual = None, None
                        else:
                            with mlcol3:
                                mv_lstm_k_manual = st.number_input(
                                    "Lag count k", min_value=1, value=5, key="widget_mv_lstm_k_manual",
                                )
                            mv_lstm_units_manual = st.number_input(
                                "Hidden units", min_value=1, value=16, key="widget_mv_lstm_units_manual",
                            )
                            mv_lstm_cv_folds, mv_lstm_max_k, mv_lstm_unit_candidates = 2, 8, None

                        mv_lstm_run_mode = st.radio(
                            "What should this run do?",
                            options=[
                                "Validate against held-out actual data (recommended)",
                                "Forecast genuinely unknown future",
                            ],
                            key="widget_mv_lstm_run_mode",
                        )
                        mv_lstm_validate = mv_lstm_run_mode.startswith("Validate")
                        st.caption(f"Using H = **{int(horizon)}** (set above, in 'Forecast Setup').")

                        if st.button("Run Multivariate LSTM Forecast", key="widget_mv_lstm_run_btn", type="primary"):
                            with st.spinner("Training multivariate LSTM ensembles..."):
                                try:
                                    if mv_lstm_validate:
                                        mv_lstm_train_dict = {col: arr[:-int(horizon)] for col, arr in series_dict.items()}
                                        mv_lstm_actual_holdout = series_dict[target_col][-int(horizon):]
                                    else:
                                        mv_lstm_train_dict = series_dict
                                        mv_lstm_actual_holdout = None

                                    mv_lstm_result = lstm_forecast.lstm_direct_forecast_multivariate(
                                        mv_lstm_train_dict, target_col, predictor_cols, horizon=int(horizon),
                                        mode="auto" if mv_lstm_auto else "manual",
                                        k=mv_lstm_k_manual, units=mv_lstm_units_manual,
                                        ensemble_size=int(mv_lstm_ensemble_size), epochs=int(mv_lstm_epochs),
                                        confidence=confidence,
                                        max_k_search=int(mv_lstm_max_k) if mv_lstm_auto else 8,
                                        unit_candidates=mv_lstm_unit_candidates,
                                        cv_folds=int(mv_lstm_cv_folds),
                                    )

                                    if mv_lstm_validate:
                                        mv_lstm_final_df = pd.DataFrame({
                                            "relative_step": [f"t+{i}" for i in range(1, int(horizon) + 1)],
                                            f"{target_col}_actual": mv_lstm_actual_holdout,
                                            f"{target_col}_point_forecast": mv_lstm_result["point"],
                                            f"{target_col}_lower_bound": mv_lstm_result["lower"],
                                            f"{target_col}_upper_bound": mv_lstm_result["upper"],
                                        })
                                    else:
                                        freq_code = data_processing.FREQ_LABEL_TO_CODE[confirmed_freq]
                                        future_dates = pd.date_range(
                                            start=mv_dates.iloc[-1], periods=int(horizon) + 1, freq=freq_code
                                        )[1:]
                                        mv_lstm_final_df = pd.DataFrame({
                                            "Date": future_dates,
                                            f"{target_col}_point_forecast": mv_lstm_result["point"],
                                            f"{target_col}_lower_bound": mv_lstm_result["lower"],
                                            f"{target_col}_upper_bound": mv_lstm_result["upper"],
                                        })

                                    st.session_state["cache_mv_lstm_result"] = mv_lstm_final_df
                                    st.session_state["cache_mv_lstm_validate_mode"] = mv_lstm_validate
                                    st.session_state["cache_mv_lstm_k_used"] = mv_lstm_result["k_used"]
                                    st.session_state["cache_mv_lstm_units_used"] = mv_lstm_result["units_used"]
                                    st.session_state["cache_mv_lstm_kfold_acc"] = mv_lstm_result["kfold_accuracy"]
                                    st.session_state["cache_mv_lstm_target_detrend"] = (
                                        mv_lstm_result["target_detrend_transform"], mv_lstm_result["target_detrend_diff_order"]
                                    )
                                    st.session_state["cache_mv_lstm_predictor_detrend"] = mv_lstm_result["predictor_detrend_info"]
                                except Exception as e:
                                    st.error(f"Multivariate LSTM forecast failed: {e}")

                        mv_lstm_result_df = st.session_state.get("cache_mv_lstm_result")
                        if mv_lstm_result_df is not None:
                            mv_lstm_was_validate = st.session_state.get("cache_mv_lstm_validate_mode", False)
                            mv_lstm_acc = st.session_state.get("cache_mv_lstm_kfold_acc")
                            mv_lstm_target_transform, mv_lstm_target_order = st.session_state.get(
                                "cache_mv_lstm_target_detrend", (None, None)
                            )
                            st.write(
                                f"Lag count used: **k = {st.session_state.get('cache_mv_lstm_k_used')}**, "
                                f"hidden units used: **{st.session_state.get('cache_mv_lstm_units_used')}**"
                            )
                            st.write(
                                f"**{target_col}** detrending: transform=**{mv_lstm_target_transform}**, "
                                f"diff order=**{mv_lstm_target_order}**"
                            )
                            mv_lstm_pred_detrend = st.session_state.get("cache_mv_lstm_predictor_detrend", {})
                            if mv_lstm_pred_detrend:
                                mv_lstm_pred_lines = "; ".join(
                                    f"{c}: transform={v['transform']}, diff order={v['diff_order']}"
                                    for c, v in mv_lstm_pred_detrend.items()
                                )
                                st.caption(f"Predictor detrending: {mv_lstm_pred_lines}")
                            if mv_lstm_acc is not None:
                                st.metric("Average K-fold CV accuracy", f"{mv_lstm_acc:.1f}%")
                            st.dataframe(mv_lstm_result_df, use_container_width=True, hide_index=True)

                            if mv_lstm_was_validate:
                                mv_lstm_fig, mv_lstm_png_bytes = forecast_plotting.plot_test_set_performance(
                                    mv_lstm_result_df[f"{target_col}_actual"].values,
                                    mv_lstm_result_df[f"{target_col}_point_forecast"].values,
                                    mv_lstm_result_df[f"{target_col}_lower_bound"].values,
                                    mv_lstm_result_df[f"{target_col}_upper_bound"].values,
                                    target_col,
                                )
                            else:
                                mv_lstm_fig, mv_lstm_png_bytes = forecast_plotting.plot_out_of_sample_forecast(
                                    mv_lstm_result_df[f"{target_col}_point_forecast"].values,
                                    mv_lstm_result_df[f"{target_col}_lower_bound"].values,
                                    mv_lstm_result_df[f"{target_col}_upper_bound"].values,
                                    target_col,
                                )
                            st.image(mv_lstm_png_bytes, use_container_width=True)

                            mldl1, mldl2, mldl3 = st.columns(3)
                            with mldl1:
                                st.download_button(
                                    "Download plot (.png)", data=mv_lstm_png_bytes,
                                    file_name=f"{target_col}_multivariate_lstm.png", mime="image/png",
                                    key="widget_mv_lstm_dl_png",
                                )
                            with mldl2:
                                mv_lstm_csv_bytes = mv_lstm_result_df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "Download forecast (CSV)", data=mv_lstm_csv_bytes,
                                    file_name=f"{target_col}_multivariate_lstm.csv", mime="text/csv",
                                    key="widget_mv_lstm_dl_csv",
                                )
                            with mldl3:
                                import io
                                mv_lstm_xlsx_buffer = io.BytesIO()
                                mv_lstm_result_df.to_excel(mv_lstm_xlsx_buffer, index=False, engine="openpyxl")
                                st.download_button(
                                    "Download forecast (Excel)", data=mv_lstm_xlsx_buffer.getvalue(),
                                    file_name=f"{target_col}_multivariate_lstm.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="widget_mv_lstm_dl_xlsx",
                                )

                        st.divider()
                        st.subheader("SARIMA-GARCH Forecast (multivariate)")
                        st.write(
                            f"Forecasting **{target_col}** using SARIMAX with "
                            f"**{', '.join(predictor_cols)}** as exogenous predictor(s). "
                            "The target keeps its own SARIMA-handled differencing "
                            "(unchanged from univariate mode); predictors are fully "
                            "detrended externally and only ever used at lags h steps "
                            "or more in the past — real forecasting never needs an "
                            "unknown future predictor value. A window of consecutive "
                            "predictor lags (plus a seasonal lag, if enabled) is "
                            "included, so SARIMAX's own fit decides which lags matter."
                        )

                        mv_sg_mode = st.radio(
                            "How should SARIMA and GARCH orders be chosen?",
                            options=["Trial and error (auto-search)", "I'll specify them"],
                            key="widget_mv_sg_hp_mode",
                        )
                        mv_sg_auto = mv_sg_mode.startswith("Trial")

                        if mv_sg_auto:
                            mscol1, mscol2 = st.columns(2)
                            with mscol1:
                                mv_sg_seasonal = st.checkbox(
                                    "Search seasonal (SARIMA) orders too",
                                    value=False, key="widget_mv_sg_seasonal",
                                )
                            with mscol2:
                                mv_sg_season_length = st.number_input(
                                    "Seasonal period length (if searching seasonal orders)",
                                    min_value=2, value=12, key="widget_mv_sg_season_length",
                                    disabled=not mv_sg_seasonal,
                                )
                            msp1, msp2, msp3 = st.columns(3)
                            with msp1:
                                mv_sg_max_p = st.number_input("Max p", min_value=0, value=3, key="widget_mv_sg_max_p")
                            with msp2:
                                mv_sg_max_d = st.number_input("Max d", min_value=0, value=2, key="widget_mv_sg_max_d")
                            with msp3:
                                mv_sg_max_q = st.number_input("Max q", min_value=0, value=3, key="widget_mv_sg_max_q")
                            mgp1, mgp2 = st.columns(2)
                            with mgp1:
                                mv_sg_max_garch_p = st.number_input("Max GARCH p", min_value=1, value=2, key="widget_mv_sg_max_garch_p")
                            with mgp2:
                                mv_sg_max_garch_q = st.number_input("Max GARCH q", min_value=1, value=2, key="widget_mv_sg_max_garch_q")
                            mv_sg_order = mv_sg_seasonal_order = None
                            mv_sg_garch_p = mv_sg_garch_q = None
                        else:
                            st.caption("SARIMA order (p, d, q):")
                            mop1, mop2, mop3 = st.columns(3)
                            with mop1:
                                mv_sg_p = st.number_input("p", min_value=0, value=1, key="widget_mv_sg_p")
                            with mop2:
                                mv_sg_d = st.number_input("d", min_value=0, value=1, key="widget_mv_sg_d")
                            with mop3:
                                mv_sg_q = st.number_input("q", min_value=0, value=1, key="widget_mv_sg_q")
                            st.caption("Seasonal order (P, D, Q, s) — leave s=0 for no seasonality:")
                            mos1, mos2, mos3, mos4 = st.columns(4)
                            with mos1:
                                mv_sg_P = st.number_input("P", min_value=0, value=0, key="widget_mv_sg_P")
                            with mos2:
                                mv_sg_D = st.number_input("D", min_value=0, value=0, key="widget_mv_sg_D")
                            with mos3:
                                mv_sg_Q = st.number_input("Q", min_value=0, value=0, key="widget_mv_sg_Q")
                            with mos4:
                                mv_sg_s = st.number_input("s (seasonal period)", min_value=0, value=0, key="widget_mv_sg_s")
                            st.caption("GARCH order (p, q):")
                            mgc1, mgc2 = st.columns(2)
                            with mgc1:
                                mv_sg_garch_p = st.number_input("GARCH p", min_value=1, value=1, key="widget_mv_sg_garch_p")
                            with mgc2:
                                mv_sg_garch_q = st.number_input("GARCH q", min_value=1, value=1, key="widget_mv_sg_garch_q")
                            mv_sg_order = (int(mv_sg_p), int(mv_sg_d), int(mv_sg_q))
                            mv_sg_seasonal_order = (int(mv_sg_P), int(mv_sg_D), int(mv_sg_Q), int(mv_sg_s))
                            mv_sg_seasonal = int(mv_sg_s) > 0
                            mv_sg_season_length = int(mv_sg_s) if mv_sg_s else None
                            mv_sg_max_p = mv_sg_max_d = mv_sg_max_q = mv_sg_max_garch_p = mv_sg_max_garch_q = None

                        mv_sg_lag_window = st.number_input(
                            "Predictor lag window (consecutive lags per predictor, starting h steps back)",
                            min_value=1, value=3, key="widget_mv_sg_lag_window",
                        )
                        mv_sg_cv_folds = st.number_input(
                            "Rolling-origin CV folds", min_value=2, value=3, key="widget_mv_sg_cv_folds",
                        )

                        st.caption(f"Using H = **{int(horizon)}** (set above, in 'Forecast Setup').")

                        if st.button("Run Multivariate SARIMA-GARCH Forecast", key="widget_mv_sg_run_btn", type="primary"):
                            with st.spinner("Fitting multivariate SARIMA-GARCH — this may take a moment..."):
                                try:
                                    mv_sg_train_dict = {col: arr[:-int(horizon)] for col, arr in series_dict.items()}
                                    mv_sg_actual_holdout = series_dict[target_col][-int(horizon):]

                                    mv_sg_result = sarima_garch.sarima_garch_forecast_multivariate(
                                        mv_sg_train_dict, target_col, predictor_cols, horizon=int(horizon),
                                        mode="auto" if mv_sg_auto else "manual",
                                        order=mv_sg_order, seasonal_order=mv_sg_seasonal_order,
                                        garch_p=None if mv_sg_auto else int(mv_sg_garch_p),
                                        garch_q=None if mv_sg_auto else int(mv_sg_garch_q),
                                        seasonal=mv_sg_seasonal, season_length=mv_sg_season_length,
                                        confidence=confidence,
                                        max_p=int(mv_sg_max_p) if mv_sg_auto else 3,
                                        max_d=int(mv_sg_max_d) if mv_sg_auto else 2,
                                        max_q=int(mv_sg_max_q) if mv_sg_auto else 3,
                                        max_garch_p=int(mv_sg_max_garch_p) if mv_sg_auto else 2,
                                        max_garch_q=int(mv_sg_max_garch_q) if mv_sg_auto else 2,
                                        predictor_lag_window=int(mv_sg_lag_window),
                                        cv_folds=int(mv_sg_cv_folds),
                                    )

                                    mv_sg_final_df = pd.DataFrame({
                                        "relative_step": [f"t+{i}" for i in range(1, int(horizon) + 1)],
                                        f"{target_col}_actual": mv_sg_actual_holdout,
                                        f"{target_col}_point_forecast": mv_sg_result["point"],
                                        f"{target_col}_lower_bound": mv_sg_result["lower"],
                                        f"{target_col}_upper_bound": mv_sg_result["upper"],
                                    })

                                    st.session_state["cache_mv_sg_result"] = mv_sg_final_df
                                    st.session_state["cache_mv_sg_order"] = mv_sg_result["order"]
                                    st.session_state["cache_mv_sg_seasonal_order"] = mv_sg_result["seasonal_order"]
                                    st.session_state["cache_mv_sg_garch_order"] = mv_sg_result["garch_order"]
                                    st.session_state["cache_mv_sg_cv_acc"] = mv_sg_result["rolling_cv_accuracy"]
                                    st.session_state["cache_mv_sg_target_transform"] = mv_sg_result["target_detrend_transform"]
                                    st.session_state["cache_mv_sg_predictor_detrend"] = mv_sg_result["predictor_detrend_info"]
                                    st.session_state["cache_mv_sg_offsets"] = mv_sg_result["predictor_lag_offsets_by_horizon"]
                                except Exception as e:
                                    st.error(f"Multivariate SARIMA-GARCH forecast failed: {e}")

                        mv_sg_result_df = st.session_state.get("cache_mv_sg_result")
                        if mv_sg_result_df is not None:
                            mv_sg_acc = st.session_state.get("cache_mv_sg_cv_acc")
                            st.write(
                                f"SARIMA order: **{st.session_state.get('cache_mv_sg_order')}**, "
                                f"seasonal order: **{st.session_state.get('cache_mv_sg_seasonal_order')}**, "
                                f"GARCH order: **{st.session_state.get('cache_mv_sg_garch_order')}**"
                            )
                            st.write(
                                f"**{target_col}** detrending: transform="
                                f"**{st.session_state.get('cache_mv_sg_target_transform')}** "
                                "(differencing handled internally by SARIMA's own order)"
                            )
                            mv_sg_pred_detrend = st.session_state.get("cache_mv_sg_predictor_detrend", {})
                            if mv_sg_pred_detrend:
                                mv_sg_pred_lines = "; ".join(
                                    f"{c}: transform={v['transform']}, diff order={v['diff_order']}"
                                    for c, v in mv_sg_pred_detrend.items()
                                )
                                st.caption(f"Predictor detrending: {mv_sg_pred_lines}")
                            mv_sg_offsets = st.session_state.get("cache_mv_sg_offsets", {})
                            if mv_sg_offsets:
                                st.caption(f"Predictor lag offsets used per horizon: {mv_sg_offsets}")
                            if mv_sg_acc is not None:
                                st.metric("Average rolling-origin CV accuracy", f"{mv_sg_acc:.1f}%")
                                st.caption(
                                    "Not shuffled K-fold — SARIMA is a sequential model, so "
                                    "folds can't be randomly shuffled without breaking the "
                                    "autoregressive structure it depends on."
                                )
                            st.dataframe(mv_sg_result_df, use_container_width=True, hide_index=True)

                            mv_sg_fig, mv_sg_png_bytes = forecast_plotting.plot_test_set_performance(
                                mv_sg_result_df[f"{target_col}_actual"].values,
                                mv_sg_result_df[f"{target_col}_point_forecast"].values,
                                mv_sg_result_df[f"{target_col}_lower_bound"].values,
                                mv_sg_result_df[f"{target_col}_upper_bound"].values,
                                target_col,
                            )
                            st.image(mv_sg_png_bytes, use_container_width=True)

                            msdl1, msdl2, msdl3 = st.columns(3)
                            with msdl1:
                                st.download_button(
                                    "Download plot (.png)", data=mv_sg_png_bytes,
                                    file_name=f"{target_col}_multivariate_sarima_garch.png", mime="image/png",
                                    key="widget_mv_sg_dl_png",
                                )
                            with msdl2:
                                mv_sg_csv_bytes = mv_sg_result_df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "Download forecast (CSV)", data=mv_sg_csv_bytes,
                                    file_name=f"{target_col}_multivariate_sarima_garch.csv", mime="text/csv",
                                    key="widget_mv_sg_dl_csv",
                                )
                            with msdl3:
                                import io
                                mv_sg_xlsx_buffer = io.BytesIO()
                                mv_sg_result_df.to_excel(mv_sg_xlsx_buffer, index=False, engine="openpyxl")
                                st.download_button(
                                    "Download forecast (Excel)", data=mv_sg_xlsx_buffer.getvalue(),
                                    file_name=f"{target_col}_multivariate_sarima_garch.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="widget_mv_sg_dl_xlsx",
                                )

                        st.divider()
                        st.subheader("N-BEATS Forecast (multivariate)")
                        st.write(
                            f"Forecasting **{target_col}** using its own history plus "
                            f"**{', '.join(predictor_cols)}** as predictor(s). Every "
                            "series is detrended independently; at each lookback "
                            "position, the target's value AND every predictor's value "
                            "(same time step) are combined into one wide input window, "
                            "flattened before entering N-BEATS' feedforward blocks."
                        )

                        mnbcol1, mnbcol2, mnbcol3 = st.columns(3)
                        with mnbcol1:
                            mv_nb_lookback = st.number_input(
                                "Lookback window", min_value=2, value=10, key="widget_mv_nb_lookback",
                            )
                        with mnbcol2:
                            mv_nb_blocks = st.number_input(
                                "Number of blocks", min_value=1, value=2, key="widget_mv_nb_blocks",
                            )
                        with mnbcol3:
                            mv_nb_hidden_units = st.number_input(
                                "Hidden units per block", min_value=1, value=16, key="widget_mv_nb_hidden_units",
                            )

                        mnbcol4, mnbcol5, mnbcol6 = st.columns(3)
                        with mnbcol4:
                            mv_nb_hidden_layers = st.number_input(
                                "Hidden layers per block", min_value=1, value=2, key="widget_mv_nb_hidden_layers",
                            )
                        with mnbcol5:
                            mv_nb_ensemble_size = st.number_input(
                                "Ensemble size", min_value=2, value=3, key="widget_mv_nb_ensemble_size",
                            )
                        with mnbcol6:
                            mv_nb_epochs = st.number_input(
                                "Training epochs", min_value=1, value=15, key="widget_mv_nb_epochs",
                            )

                        mv_nb_cv_folds = st.number_input(
                            "K-fold CV folds", min_value=2, value=2, key="widget_mv_nb_cv_folds",
                        )

                        mv_nb_run_mode = st.radio(
                            "What should this run do?",
                            options=[
                                "Validate against held-out actual data (recommended)",
                                "Forecast genuinely unknown future",
                            ],
                            key="widget_mv_nb_run_mode",
                        )
                        mv_nb_validate = mv_nb_run_mode.startswith("Validate")
                        st.caption(f"Using H = **{int(horizon)}** (set above, in 'Forecast Setup').")

                        if st.button("Run Multivariate N-BEATS Forecast", key="widget_mv_nb_run_btn", type="primary"):
                            with st.spinner("Training multivariate N-BEATS ensemble..."):
                                try:
                                    if mv_nb_validate:
                                        mv_nb_train_dict = {col: arr[:-int(horizon)] for col, arr in series_dict.items()}
                                        mv_nb_actual_holdout = series_dict[target_col][-int(horizon):]
                                    else:
                                        mv_nb_train_dict = series_dict
                                        mv_nb_actual_holdout = None

                                    mv_nb_result = nbeats_forecast_module.nbeats_forecast_multivariate(
                                        mv_nb_train_dict, target_col, predictor_cols, horizon=int(horizon),
                                        lookback=int(mv_nb_lookback), n_blocks=int(mv_nb_blocks),
                                        hidden_units=int(mv_nb_hidden_units),
                                        n_hidden_layers=int(mv_nb_hidden_layers),
                                        ensemble_size=int(mv_nb_ensemble_size), epochs=int(mv_nb_epochs),
                                        confidence=confidence, cv_folds=int(mv_nb_cv_folds),
                                    )

                                    if mv_nb_validate:
                                        mv_nb_final_df = pd.DataFrame({
                                            "relative_step": [f"t+{i}" for i in range(1, int(horizon) + 1)],
                                            f"{target_col}_actual": mv_nb_actual_holdout,
                                            f"{target_col}_point_forecast": mv_nb_result["point"],
                                            f"{target_col}_lower_bound": mv_nb_result["lower"],
                                            f"{target_col}_upper_bound": mv_nb_result["upper"],
                                        })
                                    else:
                                        freq_code = data_processing.FREQ_LABEL_TO_CODE[confirmed_freq]
                                        future_dates = pd.date_range(
                                            start=mv_dates.iloc[-1], periods=int(horizon) + 1, freq=freq_code
                                        )[1:]
                                        mv_nb_final_df = pd.DataFrame({
                                            "Date": future_dates,
                                            f"{target_col}_point_forecast": mv_nb_result["point"],
                                            f"{target_col}_lower_bound": mv_nb_result["lower"],
                                            f"{target_col}_upper_bound": mv_nb_result["upper"],
                                        })

                                    st.session_state["cache_mv_nb_result"] = mv_nb_final_df
                                    st.session_state["cache_mv_nb_validate_mode"] = mv_nb_validate
                                    st.session_state["cache_mv_nb_kfold_acc"] = mv_nb_result["kfold_accuracy"]
                                    st.session_state["cache_mv_nb_target_detrend"] = (
                                        mv_nb_result["target_detrend_transform"], mv_nb_result["target_detrend_diff_order"]
                                    )
                                    st.session_state["cache_mv_nb_predictor_detrend"] = mv_nb_result["predictor_detrend_info"]
                                except Exception as e:
                                    st.error(f"Multivariate N-BEATS forecast failed: {e}")

                        mv_nb_result_df = st.session_state.get("cache_mv_nb_result")
                        if mv_nb_result_df is not None:
                            mv_nb_was_validate = st.session_state.get("cache_mv_nb_validate_mode", False)
                            mv_nb_acc = st.session_state.get("cache_mv_nb_kfold_acc")
                            mv_nb_target_transform, mv_nb_target_order = st.session_state.get(
                                "cache_mv_nb_target_detrend", (None, None)
                            )
                            st.write(
                                f"**{target_col}** detrending: transform=**{mv_nb_target_transform}**, "
                                f"diff order=**{mv_nb_target_order}**"
                            )
                            mv_nb_pred_detrend = st.session_state.get("cache_mv_nb_predictor_detrend", {})
                            if mv_nb_pred_detrend:
                                mv_nb_pred_lines = "; ".join(
                                    f"{c}: transform={v['transform']}, diff order={v['diff_order']}"
                                    for c, v in mv_nb_pred_detrend.items()
                                )
                                st.caption(f"Predictor detrending: {mv_nb_pred_lines}")
                            if mv_nb_acc is not None:
                                st.metric("Average K-fold CV accuracy", f"{mv_nb_acc:.1f}%")
                            st.dataframe(mv_nb_result_df, use_container_width=True, hide_index=True)

                            if mv_nb_was_validate:
                                mv_nb_fig, mv_nb_png_bytes = forecast_plotting.plot_test_set_performance(
                                    mv_nb_result_df[f"{target_col}_actual"].values,
                                    mv_nb_result_df[f"{target_col}_point_forecast"].values,
                                    mv_nb_result_df[f"{target_col}_lower_bound"].values,
                                    mv_nb_result_df[f"{target_col}_upper_bound"].values,
                                    target_col,
                                )
                            else:
                                mv_nb_fig, mv_nb_png_bytes = forecast_plotting.plot_out_of_sample_forecast(
                                    mv_nb_result_df[f"{target_col}_point_forecast"].values,
                                    mv_nb_result_df[f"{target_col}_lower_bound"].values,
                                    mv_nb_result_df[f"{target_col}_upper_bound"].values,
                                    target_col,
                                )
                            st.image(mv_nb_png_bytes, use_container_width=True)

                            mnbdl1, mnbdl2, mnbdl3 = st.columns(3)
                            with mnbdl1:
                                st.download_button(
                                    "Download plot (.png)", data=mv_nb_png_bytes,
                                    file_name=f"{target_col}_multivariate_nbeats.png", mime="image/png",
                                    key="widget_mv_nb_dl_png",
                                )
                            with mnbdl2:
                                mv_nb_csv_bytes = mv_nb_result_df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "Download forecast (CSV)", data=mv_nb_csv_bytes,
                                    file_name=f"{target_col}_multivariate_nbeats.csv", mime="text/csv",
                                    key="widget_mv_nb_dl_csv",
                                )
                            with mnbdl3:
                                import io
                                mv_nb_xlsx_buffer = io.BytesIO()
                                mv_nb_result_df.to_excel(mv_nb_xlsx_buffer, index=False, engine="openpyxl")
                                st.download_button(
                                    "Download forecast (Excel)", data=mv_nb_xlsx_buffer.getvalue(),
                                    file_name=f"{target_col}_multivariate_nbeats.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="widget_mv_nb_dl_xlsx",
                                )

                        st.divider()
                        st.subheader("Transformer Forecast (multivariate)")
                        st.write(
                            f"Forecasting **{target_col}** using its own history plus "
                            f"**{', '.join(predictor_cols)}** as predictor(s). Every "
                            "series is detrended independently; at each lookback "
                            "position, the target's value AND every predictor's value "
                            "(same time step) form one wider per-timestep vector — "
                            "self-attention handles this natively, no architecture "
                            "change needed beyond the input width."
                        )

                        mtcol1, mtcol2, mtcol3 = st.columns(3)
                        with mtcol1:
                            mv_tf_lookback = st.number_input(
                                "Lookback window", min_value=2, value=10, key="widget_mv_tf_lookback",
                            )
                        with mtcol2:
                            mv_tf_num_heads = st.number_input(
                                "Attention heads", min_value=1, value=2, key="widget_mv_tf_num_heads",
                            )
                        with mtcol3:
                            mv_tf_key_dim = st.number_input(
                                "Key dimension (model width)", min_value=2, value=8, key="widget_mv_tf_key_dim",
                            )

                        mtcol4, mtcol5, mtcol6 = st.columns(3)
                        with mtcol4:
                            mv_tf_ff_dim = st.number_input(
                                "Feedforward dimension", min_value=2, value=16, key="widget_mv_tf_ff_dim",
                            )
                        with mtcol5:
                            mv_tf_num_blocks = st.number_input(
                                "Encoder blocks", min_value=1, value=1, key="widget_mv_tf_num_blocks",
                            )
                        with mtcol6:
                            mv_tf_ensemble_size = st.number_input(
                                "Ensemble size", min_value=2, value=3, key="widget_mv_tf_ensemble_size",
                            )

                        mtcol7, mtcol8 = st.columns(2)
                        with mtcol7:
                            mv_tf_epochs = st.number_input(
                                "Training epochs", min_value=1, value=15, key="widget_mv_tf_epochs",
                            )
                        with mtcol8:
                            mv_tf_cv_folds = st.number_input(
                                "K-fold CV folds", min_value=2, value=2, key="widget_mv_tf_cv_folds",
                            )

                        st.caption(
                            f"This trains {int(mv_tf_ensemble_size)} independent "
                            "Transformer encoders. Attention cost grows faster than "
                            "the other methods as lookback/width increase — reduce "
                            "those first if this is too slow on Colab free tier."
                        )

                        mv_tf_run_mode = st.radio(
                            "What should this run do?",
                            options=[
                                "Validate against held-out actual data (recommended)",
                                "Forecast genuinely unknown future",
                            ],
                            key="widget_mv_tf_run_mode",
                        )
                        mv_tf_validate = mv_tf_run_mode.startswith("Validate")
                        st.caption(f"Using H = **{int(horizon)}** (set above, in 'Forecast Setup').")

                        if st.button("Run Multivariate Transformer Forecast", key="widget_mv_tf_run_btn", type="primary"):
                            with st.spinner("Training multivariate Transformer ensemble — this may take a while..."):
                                try:
                                    if mv_tf_validate:
                                        mv_tf_train_dict = {col: arr[:-int(horizon)] for col, arr in series_dict.items()}
                                        mv_tf_actual_holdout = series_dict[target_col][-int(horizon):]
                                    else:
                                        mv_tf_train_dict = series_dict
                                        mv_tf_actual_holdout = None

                                    mv_tf_result = transformer_forecast_module.transformer_forecast_multivariate(
                                        mv_tf_train_dict, target_col, predictor_cols, horizon=int(horizon),
                                        lookback=int(mv_tf_lookback), num_heads=int(mv_tf_num_heads),
                                        key_dim=int(mv_tf_key_dim), ff_dim=int(mv_tf_ff_dim),
                                        num_blocks=int(mv_tf_num_blocks),
                                        ensemble_size=int(mv_tf_ensemble_size), epochs=int(mv_tf_epochs),
                                        confidence=confidence, cv_folds=int(mv_tf_cv_folds),
                                    )

                                    if mv_tf_validate:
                                        mv_tf_final_df = pd.DataFrame({
                                            "relative_step": [f"t+{i}" for i in range(1, int(horizon) + 1)],
                                            f"{target_col}_actual": mv_tf_actual_holdout,
                                            f"{target_col}_point_forecast": mv_tf_result["point"],
                                            f"{target_col}_lower_bound": mv_tf_result["lower"],
                                            f"{target_col}_upper_bound": mv_tf_result["upper"],
                                        })
                                    else:
                                        freq_code = data_processing.FREQ_LABEL_TO_CODE[confirmed_freq]
                                        future_dates = pd.date_range(
                                            start=mv_dates.iloc[-1], periods=int(horizon) + 1, freq=freq_code
                                        )[1:]
                                        mv_tf_final_df = pd.DataFrame({
                                            "Date": future_dates,
                                            f"{target_col}_point_forecast": mv_tf_result["point"],
                                            f"{target_col}_lower_bound": mv_tf_result["lower"],
                                            f"{target_col}_upper_bound": mv_tf_result["upper"],
                                        })

                                    st.session_state["cache_mv_tf_result"] = mv_tf_final_df
                                    st.session_state["cache_mv_tf_validate_mode"] = mv_tf_validate
                                    st.session_state["cache_mv_tf_kfold_acc"] = mv_tf_result["kfold_accuracy"]
                                    st.session_state["cache_mv_tf_target_detrend"] = (
                                        mv_tf_result["target_detrend_transform"], mv_tf_result["target_detrend_diff_order"]
                                    )
                                    st.session_state["cache_mv_tf_predictor_detrend"] = mv_tf_result["predictor_detrend_info"]
                                except Exception as e:
                                    st.error(f"Multivariate Transformer forecast failed: {e}")

                        mv_tf_result_df = st.session_state.get("cache_mv_tf_result")
                        if mv_tf_result_df is not None:
                            mv_tf_was_validate = st.session_state.get("cache_mv_tf_validate_mode", False)
                            mv_tf_acc = st.session_state.get("cache_mv_tf_kfold_acc")
                            mv_tf_target_transform, mv_tf_target_order = st.session_state.get(
                                "cache_mv_tf_target_detrend", (None, None)
                            )
                            st.write(
                                f"**{target_col}** detrending: transform=**{mv_tf_target_transform}**, "
                                f"diff order=**{mv_tf_target_order}**"
                            )
                            mv_tf_pred_detrend = st.session_state.get("cache_mv_tf_predictor_detrend", {})
                            if mv_tf_pred_detrend:
                                mv_tf_pred_lines = "; ".join(
                                    f"{c}: transform={v['transform']}, diff order={v['diff_order']}"
                                    for c, v in mv_tf_pred_detrend.items()
                                )
                                st.caption(f"Predictor detrending: {mv_tf_pred_lines}")
                            if mv_tf_acc is not None:
                                st.metric("Average K-fold CV accuracy", f"{mv_tf_acc:.1f}%")
                            st.dataframe(mv_tf_result_df, use_container_width=True, hide_index=True)

                            if mv_tf_was_validate:
                                mv_tf_fig, mv_tf_png_bytes = forecast_plotting.plot_test_set_performance(
                                    mv_tf_result_df[f"{target_col}_actual"].values,
                                    mv_tf_result_df[f"{target_col}_point_forecast"].values,
                                    mv_tf_result_df[f"{target_col}_lower_bound"].values,
                                    mv_tf_result_df[f"{target_col}_upper_bound"].values,
                                    target_col,
                                )
                            else:
                                mv_tf_fig, mv_tf_png_bytes = forecast_plotting.plot_out_of_sample_forecast(
                                    mv_tf_result_df[f"{target_col}_point_forecast"].values,
                                    mv_tf_result_df[f"{target_col}_lower_bound"].values,
                                    mv_tf_result_df[f"{target_col}_upper_bound"].values,
                                    target_col,
                                )
                            st.image(mv_tf_png_bytes, use_container_width=True)

                            mtdl1, mtdl2, mtdl3 = st.columns(3)
                            with mtdl1:
                                st.download_button(
                                    "Download plot (.png)", data=mv_tf_png_bytes,
                                    file_name=f"{target_col}_multivariate_transformer.png", mime="image/png",
                                    key="widget_mv_tf_dl_png",
                                )
                            with mtdl2:
                                mv_tf_csv_bytes = mv_tf_result_df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "Download forecast (CSV)", data=mv_tf_csv_bytes,
                                    file_name=f"{target_col}_multivariate_transformer.csv", mime="text/csv",
                                    key="widget_mv_tf_dl_csv",
                                )
                            with mtdl3:
                                import io
                                mv_tf_xlsx_buffer = io.BytesIO()
                                mv_tf_result_df.to_excel(mv_tf_xlsx_buffer, index=False, engine="openpyxl")
                                st.download_button(
                                    "Download forecast (Excel)", data=mv_tf_xlsx_buffer.getvalue(),
                                    file_name=f"{target_col}_multivariate_transformer.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="widget_mv_tf_dl_xlsx",
                                )
                else:
                    st.subheader("Forecast Setup")
                    fcol1, fcol2 = st.columns(2)
                    with fcol1:
                        horizon = st.number_input(
                            "How many out-of-sample steps to forecast (t+1 ... t+H)?",
                            min_value=1, value=10, key="widget_fc_horizon",
                        )
                    with fcol2:
                        confidence = st.selectbox(
                            "Confidence level for the forecast interval",
                            options=list(forecasting.Z_SCORES_BY_CONFIDENCE.keys()),
                            index=1,
                            key="widget_fc_confidence",
                        )

                    freq_info = data_processing.infer_native_frequency(df_fc[date_col])
                    freq_options = list(data_processing.FREQ_LABEL_TO_CODE.keys())
                    # Frequency here is only used to date the OUTPUT rows when
                    # forecasting genuinely unknown future steps -- it has no
                    # effect on the forecasting computation itself, and data
                    # quality/frequency was already handled in Module 2. No
                    # need to ask the user to "confirm" it again; just use the
                    # detected value silently, falling back to Daily if the
                    # detected label isn't one of the standard options.
                    confirmed_freq = freq_info["label"] if freq_info["label"] in freq_options else "Daily"

                    series_df = df_fc[[date_col, target_col]].dropna().sort_values(date_col)
                    series = series_df[target_col].values
                    dates = pd.to_datetime(series_df[date_col])

                    st.session_state["cache_fc_series"] = series
                    st.session_state["cache_fc_dates"] = dates
                    st.session_state["cache_fc_confirmed_freq"] = confirmed_freq

                    st.info(
                        "All five methods (LSTM, SARIMA-GARCH, DFS, N-BEATS, "
                        "Transformer) are ready to try below — each runs on its own for "
                        "now. A shared K-fold CV leaderboard comparing all of them is "
                        "the next planned step."
                    )

                    st.subheader("LSTM Forecast (direct multi-step, deep ensemble)")
                    st.write(
                        "Trains a separate small LSTM ensemble per horizon step "
                        "(t+1, t+2, ..., t+H) on log-differenced data. Architecture is "
                        "fixed at one LSTM layer (standard practice for this problem "
                        "size); ensemble size and epochs are lightweight defaults for "
                        "Colab's free tier — a paying client's run can use larger values "
                        "for better accuracy."
                    )

                    hp_mode = st.radio(
                        "How should the lag count (k) and hidden units be chosen?",
                        options=["Trial and error (auto-search)", "I'll specify them"],
                        key="widget_lstm_hp_mode",
                        help=(
                            "Auto-search tries several (k, units) combinations via "
                            "K-fold CV and picks the best — thorough, but can take a "
                            "while since each candidate needs its own CV cycle. "
                            "Specifying them yourself skips the search entirely."
                        ),
                    )
                    auto_mode = hp_mode.startswith("Trial")

                    lcol1, lcol2, lcol3 = st.columns(3)
                    with lcol1:
                        ensemble_size = st.number_input(
                            "Ensemble size", min_value=2, value=5,
                            key="widget_lstm_ensemble_size",
                        )
                    with lcol2:
                        epochs = st.number_input(
                            "Training epochs", min_value=1, value=15,
                            key="widget_lstm_epochs",
                        )

                    if auto_mode:
                        with lcol3:
                            cv_folds = st.number_input(
                                "K-fold CV folds", min_value=1, value=2,
                                key="widget_lstm_cv_folds",
                                help="More folds validate more thoroughly but take "
                                     "longer — choose based on your available compute.",
                            )
                        mcol1, mcol2 = st.columns(2)
                        with mcol1:
                            max_k_search = st.number_input(
                                "Max lag count to try", min_value=1, value=8,
                                key="widget_lstm_max_k",
                            )
                        with mcol2:
                            units_text = st.text_input(
                                "Hidden unit candidates to try (comma-separated)",
                                value="8, 16, 32", key="widget_lstm_units_candidates",
                            )
                        unit_candidates = [int(u.strip()) for u in units_text.split(",") if u.strip()]
                        k_manual, units_manual = None, None
                    else:
                        with lcol3:
                            k_manual = st.number_input(
                                "Lag count k", min_value=1, value=5,
                                key="widget_lstm_k_manual",
                            )
                        units_manual = st.number_input(
                            "Hidden units", min_value=1, value=16,
                            key="widget_lstm_units_manual",
                        )
                        cv_folds, max_k_search, unit_candidates = 2, 8, None

                    st.caption(
                        f"This trains {int(horizon)} separate ensembles of "
                        f"{int(ensemble_size)} models each"
                        + (" plus the (k, units) search cycle" if auto_mode else "")
                        + " — can take a while depending on these settings."
                    )

                    run_mode = st.radio(
                        "What should this run do?",
                        options=[
                            "Validate against held-out actual data (recommended)",
                            "Forecast genuinely unknown future",
                        ],
                        key="widget_lstm_run_mode",
                        help=(
                            "Validate: the last H actual values are set aside BEFORE "
                            "training — the model never sees them — then forecasts "
                            "that same period, so you can compare forecast vs. what "
                            "really happened. Forecast: trains on everything and "
                            "predicts beyond the end of your data, where there's no "
                            "actual value to compare against yet."
                        ),
                    )
                    validate_mode = run_mode.startswith("Validate")

                    if validate_mode:
                        st.caption(
                            f"Using H = **{int(horizon)}** (set above, in 'Forecast "
                            f"Setup') — the last {int(horizon)} actual values will be "
                            "held out and never used in training, then forecast for "
                            "comparison. Change H there if you want a different "
                            "validation window size."
                        )
                    else:
                        st.caption(
                            f"Using H = **{int(horizon)}** (set above, in 'Forecast "
                            f"Setup') — will forecast {int(horizon)} steps beyond the "
                            "end of your data."
                        )

                    if validate_mode and len(series) <= int(horizon) + 10:
                        st.warning(
                            f"Series has {len(series)} points; holding out {int(horizon)} "
                            "for validation leaves very little to train on. Consider a "
                            "smaller horizon for validation, or use more data."
                        )

                    if st.button("Run LSTM Forecast", key="widget_lstm_run_btn", type="primary"):
                        with st.spinner("Training LSTM ensembles — this may take a few minutes..."):
                            try:
                                if validate_mode:
                                    train_series = series[:-int(horizon)]
                                    actual_holdout = series[-int(horizon):]
                                else:
                                    train_series = series
                                    actual_holdout = None

                                result = lstm_forecast.lstm_direct_forecast(
                                    train_series, horizon=int(horizon),
                                    mode="auto" if auto_mode else "manual",
                                    k=k_manual, units=units_manual,
                                    ensemble_size=int(ensemble_size), epochs=int(epochs),
                                    confidence=confidence,
                                    max_k_search=int(max_k_search) if auto_mode else 8,
                                    unit_candidates=unit_candidates,
                                    cv_folds=int(cv_folds) if auto_mode else 2,
                                )

                                if validate_mode:
                                    final_df = pd.DataFrame({
                                        "relative_step": [f"t+{i}" for i in range(1, int(horizon) + 1)],
                                        f"{target_col}_actual": actual_holdout,
                                        f"{target_col}_point_forecast": result["point"],
                                        f"{target_col}_lower_bound": result["lower"],
                                        f"{target_col}_upper_bound": result["upper"],
                                    })
                                else:
                                    freq_code = data_processing.FREQ_LABEL_TO_CODE[confirmed_freq]
                                    future_dates = pd.date_range(
                                        start=dates.iloc[-1], periods=int(horizon) + 1, freq=freq_code
                                    )[1:]
                                    final_df = pd.DataFrame({
                                        "Date": future_dates,
                                        f"{target_col}_point_forecast": result["point"],
                                        f"{target_col}_lower_bound": result["lower"],
                                        f"{target_col}_upper_bound": result["upper"],
                                    })

                                st.session_state["cache_lstm_result"] = final_df
                                st.session_state["cache_lstm_validate_mode"] = validate_mode
                                st.session_state["cache_lstm_k_used"] = result["k_used"]
                                st.session_state["cache_lstm_units_used"] = result["units_used"]
                                st.session_state["cache_lstm_kfold_acc"] = result["kfold_accuracy"]
                            except Exception as e:
                                st.error(f"LSTM forecast failed: {e}")

                    lstm_result = st.session_state.get("cache_lstm_result")
                    if lstm_result is not None:
                        was_validate = st.session_state.get("cache_lstm_validate_mode", False)
                        lstm_acc = st.session_state.get("cache_lstm_kfold_acc")
                        st.write(
                            f"Lag count used: **k = {st.session_state.get('cache_lstm_k_used')}**, "
                            f"hidden units used: **{st.session_state.get('cache_lstm_units_used')}**"
                        )
                        if lstm_acc is not None:
                            st.metric("Average K-fold CV accuracy", f"{lstm_acc:.1f}%")
                        st.dataframe(lstm_result, use_container_width=True, hide_index=True)

                        if was_validate:
                            fig, png_bytes = forecast_plotting.plot_test_set_performance(
                                lstm_result[f"{target_col}_actual"].values,
                                lstm_result[f"{target_col}_point_forecast"].values,
                                lstm_result[f"{target_col}_lower_bound"].values,
                                lstm_result[f"{target_col}_upper_bound"].values,
                                target_col,
                            )
                        else:
                            fig, png_bytes = forecast_plotting.plot_out_of_sample_forecast(
                                lstm_result[f"{target_col}_point_forecast"].values,
                                lstm_result[f"{target_col}_lower_bound"].values,
                                lstm_result[f"{target_col}_upper_bound"].values,
                                target_col,
                            )
                        st.image(png_bytes, use_container_width=True)

                        dl1, dl2, dl3 = st.columns(3)
                        with dl1:
                            st.download_button(
                                "Download plot (.png)", data=png_bytes,
                                file_name=f"{target_col}_lstm_forecast.png", mime="image/png",
                                key="widget_lstm_dl_png",
                            )
                        with dl2:
                            csv_bytes = lstm_result.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Download forecast (CSV)", data=csv_bytes,
                                file_name=f"{target_col}_lstm_forecast.csv", mime="text/csv",
                                key="widget_lstm_dl_csv",
                            )
                        with dl3:
                            import io
                            xlsx_buffer = io.BytesIO()
                            lstm_result.to_excel(xlsx_buffer, index=False, engine="openpyxl")
                            st.download_button(
                                "Download forecast (Excel)", data=xlsx_buffer.getvalue(),
                                file_name=f"{target_col}_lstm_forecast.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="widget_lstm_dl_xlsx",
                            )

                    st.divider()
                    st.subheader("SARIMA-GARCH Forecast")
                    st.write(
                        "SARIMA models the trend/seasonality of the (log-transformed) "
                        "series; GARCH models the volatility of SARIMA's residuals, "
                        "giving a realistic, time-varying interval rather than a "
                        "constant-width one. Seasonal order is never found by manually "
                        "reading ACF/PACF plots — only by automatic search or direct "
                        "specification."
                    )

                    sg_mode = st.radio(
                        "How should SARIMA and GARCH orders be chosen?",
                        options=["Trial and error (auto-search)", "I'll specify them"],
                        key="widget_sg_hp_mode",
                        help=(
                            "Auto-search runs an AIC grid search over both SARIMA "
                            "(p,d,q)(P,D,Q,s) and GARCH (p,q) orders — thorough but can "
                            "take a while. Specifying them yourself skips the search."
                        ),
                    )
                    sg_auto = sg_mode.startswith("Trial")

                    if sg_auto:
                        scol1, scol2 = st.columns(2)
                        with scol1:
                            sg_seasonal = st.checkbox(
                                "Search seasonal (SARIMA) orders too",
                                value=False, key="widget_sg_seasonal",
                                help="Only turn this on if you expect real seasonality "
                                     "(e.g. sales data) — adds more search combinations.",
                            )
                        with scol2:
                            sg_season_length = st.number_input(
                                "Seasonal period length (if searching seasonal orders)",
                                min_value=2, value=12,
                                key="widget_sg_season_length", disabled=not sg_seasonal,
                            )
                        sp_col1, sp_col2, sp_col3 = st.columns(3)
                        with sp_col1:
                            sg_max_p = st.number_input("Max p", min_value=0, value=3, key="widget_sg_max_p")
                        with sp_col2:
                            sg_max_d = st.number_input("Max d", min_value=0, value=2, key="widget_sg_max_d")
                        with sp_col3:
                            sg_max_q = st.number_input("Max q", min_value=0, value=3, key="widget_sg_max_q")
                        gp_col1, gp_col2 = st.columns(2)
                        with gp_col1:
                            sg_max_garch_p = st.number_input("Max GARCH p", min_value=1, value=2, key="widget_sg_max_garch_p")
                        with gp_col2:
                            sg_max_garch_q = st.number_input("Max GARCH q", min_value=1, value=2, key="widget_sg_max_garch_q")
                        sg_order = sg_seasonal_order = None
                        sg_garch_p = sg_garch_q = None
                    else:
                        st.caption("SARIMA order (p, d, q):")
                        op_col1, op_col2, op_col3 = st.columns(3)
                        with op_col1:
                            sg_p = st.number_input("p", min_value=0, value=1, key="widget_sg_p")
                        with op_col2:
                            sg_d = st.number_input("d", min_value=0, value=1, key="widget_sg_d")
                        with op_col3:
                            sg_q = st.number_input("q", min_value=0, value=1, key="widget_sg_q")
                        st.caption("Seasonal order (P, D, Q, s) — leave s=0 for no seasonality:")
                        os_col1, os_col2, os_col3, os_col4 = st.columns(4)
                        with os_col1:
                            sg_P = st.number_input("P", min_value=0, value=0, key="widget_sg_P")
                        with os_col2:
                            sg_D = st.number_input("D", min_value=0, value=0, key="widget_sg_D")
                        with os_col3:
                            sg_Q = st.number_input("Q", min_value=0, value=0, key="widget_sg_Q")
                        with os_col4:
                            sg_s = st.number_input("s (seasonal period)", min_value=0, value=0, key="widget_sg_s")
                        st.caption("GARCH order (p, q):")
                        gc_col1, gc_col2 = st.columns(2)
                        with gc_col1:
                            sg_garch_p = st.number_input("GARCH p", min_value=1, value=1, key="widget_sg_garch_p")
                        with gc_col2:
                            sg_garch_q = st.number_input("GARCH q", min_value=1, value=1, key="widget_sg_garch_q")
                        sg_order = (int(sg_p), int(sg_d), int(sg_q))
                        sg_seasonal_order = (int(sg_P), int(sg_D), int(sg_Q), int(sg_s))
                        sg_seasonal = int(sg_s) > 0
                        sg_season_length = int(sg_s) if sg_s else None
                        sg_max_p = sg_max_d = sg_max_q = sg_max_garch_p = sg_max_garch_q = None

                    sg_run_mode = st.radio(
                        "What should this run do?",
                        options=[
                            "Validate against held-out actual data (recommended)",
                            "Forecast genuinely unknown future",
                        ],
                        key="widget_sg_run_mode",
                    )
                    sg_validate = sg_run_mode.startswith("Validate")
                    st.caption(f"Using H = **{int(horizon)}** (set above, in 'Forecast Setup').")

                    sg_cv_folds = st.number_input(
                        "Rolling-origin CV folds", min_value=2, value=3,
                        key="widget_sg_cv_folds",
                        help="More folds validate more thoroughly but take longer — "
                             "choose based on your available compute.",
                    )

                    if st.button("Run SARIMA-GARCH Forecast", key="widget_sg_run_btn", type="primary"):
                        with st.spinner("Fitting SARIMA-GARCH — this may take a moment..."):
                            try:
                                if sg_validate:
                                    sg_train_series = series[:-int(horizon)]
                                    sg_actual_holdout = series[-int(horizon):]
                                else:
                                    sg_train_series = series
                                    sg_actual_holdout = None

                                sg_result = sarima_garch.sarima_garch_forecast(
                                    sg_train_series, horizon=int(horizon),
                                    mode="auto" if sg_auto else "manual",
                                    order=sg_order, seasonal_order=sg_seasonal_order,
                                    garch_p=None if sg_auto else int(sg_garch_p),
                                    garch_q=None if sg_auto else int(sg_garch_q),
                                    seasonal=sg_seasonal, season_length=sg_season_length,
                                    confidence=confidence,
                                    max_p=int(sg_max_p) if sg_auto else 3,
                                    max_d=int(sg_max_d) if sg_auto else 2,
                                    max_q=int(sg_max_q) if sg_auto else 3,
                                    max_garch_p=int(sg_max_garch_p) if sg_auto else 2,
                                    max_garch_q=int(sg_max_garch_q) if sg_auto else 2,
                                    cv_folds=int(sg_cv_folds),
                                )

                                if sg_validate:
                                    sg_final_df = pd.DataFrame({
                                        "relative_step": [f"t+{i}" for i in range(1, int(horizon) + 1)],
                                        f"{target_col}_actual": sg_actual_holdout,
                                        f"{target_col}_point_forecast": sg_result["point"],
                                        f"{target_col}_lower_bound": sg_result["lower"],
                                        f"{target_col}_upper_bound": sg_result["upper"],
                                    })
                                else:
                                    freq_code = data_processing.FREQ_LABEL_TO_CODE[confirmed_freq]
                                    future_dates = pd.date_range(
                                        start=dates.iloc[-1], periods=int(horizon) + 1, freq=freq_code
                                    )[1:]
                                    sg_final_df = pd.DataFrame({
                                        "Date": future_dates,
                                        f"{target_col}_point_forecast": sg_result["point"],
                                        f"{target_col}_lower_bound": sg_result["lower"],
                                        f"{target_col}_upper_bound": sg_result["upper"],
                                    })

                                st.session_state["cache_sg_result"] = sg_final_df
                                st.session_state["cache_sg_validate_mode"] = sg_validate
                                st.session_state["cache_sg_order"] = sg_result["order"]
                                st.session_state["cache_sg_seasonal_order"] = sg_result["seasonal_order"]
                                st.session_state["cache_sg_garch_order"] = sg_result["garch_order"]
                                st.session_state["cache_sg_cv_acc"] = sg_result["rolling_cv_accuracy"]
                            except Exception as e:
                                st.error(f"SARIMA-GARCH forecast failed: {e}")

                    sg_result_df = st.session_state.get("cache_sg_result")
                    if sg_result_df is not None:
                        sg_was_validate = st.session_state.get("cache_sg_validate_mode", False)
                        sg_acc = st.session_state.get("cache_sg_cv_acc")
                        st.write(
                            f"SARIMA order: **{st.session_state.get('cache_sg_order')}**, "
                            f"seasonal order: **{st.session_state.get('cache_sg_seasonal_order')}**, "
                            f"GARCH order: **{st.session_state.get('cache_sg_garch_order')}**"
                        )
                        if sg_acc is not None:
                            st.metric("Average rolling-origin CV accuracy", f"{sg_acc:.1f}%")
                            st.caption(
                                "Not shuffled K-fold — SARIMA is a sequential model, so "
                                "folds can't be randomly shuffled without breaking the "
                                "autoregressive structure it depends on. This uses "
                                "rolling-origin validation instead: multiple points along "
                                "the training data, each scored on the next step only."
                            )
                        st.dataframe(sg_result_df, use_container_width=True, hide_index=True)

                        if sg_was_validate:
                            sg_fig, sg_png_bytes = forecast_plotting.plot_test_set_performance(
                                sg_result_df[f"{target_col}_actual"].values,
                                sg_result_df[f"{target_col}_point_forecast"].values,
                                sg_result_df[f"{target_col}_lower_bound"].values,
                                sg_result_df[f"{target_col}_upper_bound"].values,
                                target_col,
                            )
                        else:
                            sg_fig, sg_png_bytes = forecast_plotting.plot_out_of_sample_forecast(
                                sg_result_df[f"{target_col}_point_forecast"].values,
                                sg_result_df[f"{target_col}_lower_bound"].values,
                                sg_result_df[f"{target_col}_upper_bound"].values,
                                target_col,
                            )
                        st.image(sg_png_bytes, use_container_width=True)

                        sdl1, sdl2, sdl3 = st.columns(3)
                        with sdl1:
                            st.download_button(
                                "Download plot (.png)", data=sg_png_bytes,
                                file_name=f"{target_col}_sarima_garch.png", mime="image/png",
                                key="widget_sg_dl_png",
                            )
                        with sdl2:
                            sg_csv_bytes = sg_result_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Download forecast (CSV)", data=sg_csv_bytes,
                                file_name=f"{target_col}_sarima_garch.csv", mime="text/csv",
                                key="widget_sg_dl_csv",
                            )
                        with sdl3:
                            import io
                            sg_xlsx_buffer = io.BytesIO()
                            sg_result_df.to_excel(sg_xlsx_buffer, index=False, engine="openpyxl")
                            st.download_button(
                                "Download forecast (Excel)", data=sg_xlsx_buffer.getvalue(),
                                file_name=f"{target_col}_sarima_garch.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="widget_sg_dl_xlsx",
                            )

                    st.divider()
                    st.subheader("DFS Forecast (wavelet decomposition + boosting CNN ensembles)")
                    st.write(
                        "The novel method from the user's own paper. Wavelet-decomposes "
                        "the (log-transformed) series into several frequency components, "
                        "trains a small CNN on a sliding window across all of them, and "
                        "keeps adding CNN 'members' — each one fitting what the previous "
                        "members got wrong — until the leftover error looks like white "
                        "noise. A second, parallel ensemble does the same for volatility, "
                        "giving the forecast interval. This is the heaviest of the three "
                        "methods — it trains multiple CNNs per horizon step."
                    )

                    dcol1, dcol2, dcol3 = st.columns(3)
                    with dcol1:
                        _all_wavelets = sorted(__import__("pywt").wavelist(kind="discrete"))
                        _default_wavelet_idx = _all_wavelets.index("db4") if "db4" in _all_wavelets else 0
                        dfs_wavelet = st.selectbox(
                            "Wavelet",
                            options=_all_wavelets,
                            index=_default_wavelet_idx,
                            key="widget_dfs_wavelet",
                            help=(
                                "Full list from PyWavelets — try different ones to see "
                                "which forecasts best for your data. 'db4' is a "
                                "reasonable general-purpose default; not auto-searched, "
                                "per your preference — you pick, the app just runs it."
                            ),
                        )
                    with dcol2:
                        dfs_level = st.number_input(
                            "Decomposition level", min_value=1, value=2,
                            key="widget_dfs_level",
                            help="A higher level splits the series into more frequency "
                                 "components — more detail, but each extra level also "
                                 "means more CNNs to train and less usable data per "
                                 "component near the series' edges. As a loose guide, "
                                 "levels beyond roughly log2(number of data points) "
                                 "mostly add boundary-effect noise rather than real "
                                 "signal — but nothing stops you from trying higher.",
                        )
                    with dcol3:
                        dfs_m = st.number_input(
                            "Window size (m)", min_value=2, value=10,
                            key="widget_dfs_m",
                        )
                    st.caption(
                        "Wavelet/level/window aren't auto-searched — the paper describes "
                        "choosing them by trial and error across separate runs, and "
                        "auto-searching them here would multiply an already expensive "
                        "training cost. Try a few combinations manually if the default "
                        "doesn't perform well."
                    )

                    dcol4, dcol5, dcol6 = st.columns(3)
                    with dcol4:
                        dfs_epochs = st.number_input(
                            "Training epochs per CNN", min_value=1, value=15,
                            key="widget_dfs_epochs",
                        )
                    with dcol5:
                        dfs_max_members = st.number_input(
                            "Max ensemble members (safety cap)", min_value=1, value=3,
                            key="widget_dfs_max_members",
                            help="The paper grows the ensemble until residuals look like "
                                 "white noise, with no guaranteed upper bound — this caps "
                                 "it so a run can't train forever.",
                        )
                    with dcol6:
                        dfs_threshold = st.number_input(
                            "White-noise threshold (mean |residual|)", min_value=0.0001, value=0.05, step=0.01, key="widget_dfs_threshold",
                        )

                    st.caption(
                        f"This trains up to {int(dfs_max_members)} CNNs for the mean "
                        f"ensemble AND up to {int(dfs_max_members)} more for the variance "
                        f"ensemble, PER horizon step ({int(horizon)} steps) — can take "
                        "several minutes."
                    )

                    dfs_run_mode = st.radio(
                        "What should this run do?",
                        options=[
                            "Validate against held-out actual data (recommended)",
                            "Forecast genuinely unknown future",
                        ],
                        key="widget_dfs_run_mode",
                    )
                    dfs_validate = dfs_run_mode.startswith("Validate")
                    st.caption(f"Using H = **{int(horizon)}** (set above, in 'Forecast Setup').")

                    dfs_cv_folds = st.number_input(
                        "K-fold CV folds (member-1-only check)", min_value=2, value=2,
                        key="widget_dfs_cv_folds",
                        help="More folds validate more thoroughly but take longer — "
                             "choose based on your available compute.",
                    )

                    if st.button("Run DFS Forecast", key="widget_dfs_run_btn", type="primary"):
                        with st.spinner("Training DFS ensembles — this may take several minutes..."):
                            try:
                                if dfs_validate:
                                    dfs_train_series = series[:-int(horizon)]
                                    dfs_actual_holdout = series[-int(horizon):]
                                else:
                                    dfs_train_series = series
                                    dfs_actual_holdout = None

                                dfs_result = dfs_forecast_module.dfs_forecast(
                                    dfs_train_series, horizon=int(horizon),
                                    m=int(dfs_m), wavelet=dfs_wavelet, level=int(dfs_level),
                                    epochs=int(dfs_epochs), max_members=int(dfs_max_members),
                                    white_noise_threshold=float(dfs_threshold),
                                    confidence=confidence, cv_folds=int(dfs_cv_folds),
                                )

                                if dfs_validate:
                                    dfs_final_df = pd.DataFrame({
                                        "relative_step": [f"t+{i}" for i in range(1, int(horizon) + 1)],
                                        f"{target_col}_actual": dfs_actual_holdout,
                                        f"{target_col}_point_forecast": dfs_result["point"],
                                        f"{target_col}_lower_bound": dfs_result["lower"],
                                        f"{target_col}_upper_bound": dfs_result["upper"],
                                    })
                                else:
                                    freq_code = data_processing.FREQ_LABEL_TO_CODE[confirmed_freq]
                                    future_dates = pd.date_range(
                                        start=dates.iloc[-1], periods=int(horizon) + 1, freq=freq_code
                                    )[1:]
                                    dfs_final_df = pd.DataFrame({
                                        "Date": future_dates,
                                        f"{target_col}_point_forecast": dfs_result["point"],
                                        f"{target_col}_lower_bound": dfs_result["lower"],
                                        f"{target_col}_upper_bound": dfs_result["upper"],
                                    })

                                st.session_state["cache_dfs_result"] = dfs_final_df
                                st.session_state["cache_dfs_validate_mode"] = dfs_validate
                                st.session_state["cache_dfs_n_mean"] = dfs_result["n_mean_members"]
                                st.session_state["cache_dfs_n_var"] = dfs_result["n_var_members"]
                                st.session_state["cache_dfs_kfold_acc"] = dfs_result["kfold_accuracy"]
                            except Exception as e:
                                st.error(f"DFS forecast failed: {e}")

                    dfs_result_df = st.session_state.get("cache_dfs_result")
                    if dfs_result_df is not None:
                        dfs_was_validate = st.session_state.get("cache_dfs_validate_mode", False)
                        dfs_acc = st.session_state.get("cache_dfs_kfold_acc")
                        st.write(
                            f"Mean-ensemble members per horizon step: "
                            f"**{st.session_state.get('cache_dfs_n_mean')}**, "
                            f"variance-ensemble members per horizon step: "
                            f"**{st.session_state.get('cache_dfs_n_var')}**"
                        )
                        if dfs_acc is not None:
                            st.metric("Average K-fold CV accuracy", f"{dfs_acc:.1f}%")
                            st.caption(
                                "Based on a single CNN fit (what would be ensemble "
                                "member 1 alone), not the full boosted ensemble — "
                                "running the whole ensemble inside every CV fold would "
                                "multiply this already expensive method's cost several "
                                "times over."
                            )
                        st.dataframe(dfs_result_df, use_container_width=True, hide_index=True)

                        if dfs_was_validate:
                            dfs_fig, dfs_png_bytes = forecast_plotting.plot_test_set_performance(
                                dfs_result_df[f"{target_col}_actual"].values,
                                dfs_result_df[f"{target_col}_point_forecast"].values,
                                dfs_result_df[f"{target_col}_lower_bound"].values,
                                dfs_result_df[f"{target_col}_upper_bound"].values,
                                target_col,
                            )
                        else:
                            dfs_fig, dfs_png_bytes = forecast_plotting.plot_out_of_sample_forecast(
                                dfs_result_df[f"{target_col}_point_forecast"].values,
                                dfs_result_df[f"{target_col}_lower_bound"].values,
                                dfs_result_df[f"{target_col}_upper_bound"].values,
                                target_col,
                            )
                        st.image(dfs_png_bytes, use_container_width=True)

                        ddl1, ddl2, ddl3 = st.columns(3)
                        with ddl1:
                            st.download_button(
                                "Download plot (.png)", data=dfs_png_bytes,
                                file_name=f"{target_col}_dfs.png", mime="image/png",
                                key="widget_dfs_dl_png",
                            )
                        with ddl2:
                            dfs_csv_bytes = dfs_result_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Download forecast (CSV)", data=dfs_csv_bytes,
                                file_name=f"{target_col}_dfs.csv", mime="text/csv",
                                key="widget_dfs_dl_csv",
                            )
                        with ddl3:
                            import io
                            dfs_xlsx_buffer = io.BytesIO()
                            dfs_result_df.to_excel(dfs_xlsx_buffer, index=False, engine="openpyxl")
                            st.download_button(
                                "Download forecast (Excel)", data=dfs_xlsx_buffer.getvalue(),
                                file_name=f"{target_col}_dfs.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="widget_dfs_dl_xlsx",
                            )

                    st.divider()
                    st.subheader("N-BEATS Forecast")
                    st.write(
                        "The architecture that placed at/near the top of the M4 "
                        "forecasting competition and is widely cited since. A stack of "
                        "small feedforward blocks, each one fixing what earlier blocks "
                        "got wrong (same boosting spirit as DFS above) — but unlike "
                        "LSTM/DFS, one model call predicts the WHOLE horizon at once, "
                        "rather than training a separate model per step."
                    )

                    ncol1, ncol2, ncol3 = st.columns(3)
                    with ncol1:
                        nb_lookback = st.number_input(
                            "Lookback window", min_value=2, value=10, key="widget_nb_lookback",
                        )
                    with ncol2:
                        nb_blocks = st.number_input(
                            "Number of blocks", min_value=1, value=2, key="widget_nb_blocks",
                        )
                    with ncol3:
                        nb_hidden_units = st.number_input(
                            "Hidden units per block", min_value=1, value=16, key="widget_nb_hidden_units",
                        )

                    ncol4, ncol5, ncol6 = st.columns(3)
                    with ncol4:
                        nb_hidden_layers = st.number_input(
                            "Hidden layers per block", min_value=1, value=2, key="widget_nb_hidden_layers",
                        )
                    with ncol5:
                        nb_ensemble_size = st.number_input(
                            "Ensemble size", min_value=2, value=3, key="widget_nb_ensemble_size",
                        )
                    with ncol6:
                        nb_epochs = st.number_input(
                            "Training epochs", min_value=1, value=15, key="widget_nb_epochs",
                        )

                    nb_cv_folds = st.number_input(
                        "K-fold CV folds", min_value=2, value=2, key="widget_nb_cv_folds",
                        help="More folds validate more thoroughly but take longer — "
                             "choose based on your available compute.",
                    )

                    st.caption(
                        f"This trains {int(nb_ensemble_size)} independent N-BEATS "
                        "networks (one forward pass each covers the whole horizon, so "
                        "this is lighter than LSTM/DFS's per-step training, but still "
                        "scales with all the settings above)."
                    )

                    nb_run_mode = st.radio(
                        "What should this run do?",
                        options=[
                            "Validate against held-out actual data (recommended)",
                            "Forecast genuinely unknown future",
                        ],
                        key="widget_nb_run_mode",
                    )
                    nb_validate = nb_run_mode.startswith("Validate")
                    st.caption(f"Using H = **{int(horizon)}** (set above, in 'Forecast Setup').")

                    if st.button("Run N-BEATS Forecast", key="widget_nb_run_btn", type="primary"):
                        with st.spinner("Training N-BEATS ensemble..."):
                            try:
                                if nb_validate:
                                    nb_train_series = series[:-int(horizon)]
                                    nb_actual_holdout = series[-int(horizon):]
                                else:
                                    nb_train_series = series
                                    nb_actual_holdout = None

                                nb_result = nbeats_forecast_module.nbeats_forecast(
                                    nb_train_series, horizon=int(horizon),
                                    lookback=int(nb_lookback), n_blocks=int(nb_blocks),
                                    hidden_units=int(nb_hidden_units),
                                    n_hidden_layers=int(nb_hidden_layers),
                                    ensemble_size=int(nb_ensemble_size), epochs=int(nb_epochs),
                                    confidence=confidence, cv_folds=int(nb_cv_folds),
                                )

                                if nb_validate:
                                    nb_final_df = pd.DataFrame({
                                        "relative_step": [f"t+{i}" for i in range(1, int(horizon) + 1)],
                                        f"{target_col}_actual": nb_actual_holdout,
                                        f"{target_col}_point_forecast": nb_result["point"],
                                        f"{target_col}_lower_bound": nb_result["lower"],
                                        f"{target_col}_upper_bound": nb_result["upper"],
                                    })
                                else:
                                    freq_code = data_processing.FREQ_LABEL_TO_CODE[confirmed_freq]
                                    future_dates = pd.date_range(
                                        start=dates.iloc[-1], periods=int(horizon) + 1, freq=freq_code
                                    )[1:]
                                    nb_final_df = pd.DataFrame({
                                        "Date": future_dates,
                                        f"{target_col}_point_forecast": nb_result["point"],
                                        f"{target_col}_lower_bound": nb_result["lower"],
                                        f"{target_col}_upper_bound": nb_result["upper"],
                                    })

                                st.session_state["cache_nb_result"] = nb_final_df
                                st.session_state["cache_nb_validate_mode"] = nb_validate
                                st.session_state["cache_nb_kfold_acc"] = nb_result["kfold_accuracy"]
                            except Exception as e:
                                st.error(f"N-BEATS forecast failed: {e}")

                    nb_result_df = st.session_state.get("cache_nb_result")
                    if nb_result_df is not None:
                        nb_was_validate = st.session_state.get("cache_nb_validate_mode", False)
                        nb_acc = st.session_state.get("cache_nb_kfold_acc")
                        if nb_acc is not None:
                            st.metric("Average K-fold CV accuracy", f"{nb_acc:.1f}%")
                        st.dataframe(nb_result_df, use_container_width=True, hide_index=True)

                        if nb_was_validate:
                            nb_fig, nb_png_bytes = forecast_plotting.plot_test_set_performance(
                                nb_result_df[f"{target_col}_actual"].values,
                                nb_result_df[f"{target_col}_point_forecast"].values,
                                nb_result_df[f"{target_col}_lower_bound"].values,
                                nb_result_df[f"{target_col}_upper_bound"].values,
                                target_col,
                            )
                        else:
                            nb_fig, nb_png_bytes = forecast_plotting.plot_out_of_sample_forecast(
                                nb_result_df[f"{target_col}_point_forecast"].values,
                                nb_result_df[f"{target_col}_lower_bound"].values,
                                nb_result_df[f"{target_col}_upper_bound"].values,
                                target_col,
                            )
                        st.image(nb_png_bytes, use_container_width=True)

                        ndl1, ndl2, ndl3 = st.columns(3)
                        with ndl1:
                            st.download_button(
                                "Download plot (.png)", data=nb_png_bytes,
                                file_name=f"{target_col}_nbeats.png", mime="image/png",
                                key="widget_nb_dl_png",
                            )
                        with ndl2:
                            nb_csv_bytes = nb_result_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Download forecast (CSV)", data=nb_csv_bytes,
                                file_name=f"{target_col}_nbeats.csv", mime="text/csv",
                                key="widget_nb_dl_csv",
                            )
                        with ndl3:
                            import io
                            nb_xlsx_buffer = io.BytesIO()
                            nb_result_df.to_excel(nb_xlsx_buffer, index=False, engine="openpyxl")
                            st.download_button(
                                "Download forecast (Excel)", data=nb_xlsx_buffer.getvalue(),
                                file_name=f"{target_col}_nbeats.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="widget_nb_dl_xlsx",
                            )

                    st.divider()
                    st.subheader("Transformer Forecast")
                    st.write(
                        "Self-attention over the lookback window (the architecture "
                        "family behind models like the Temporal Fusion Transformer) — "
                        "genuinely the most computationally expensive method here at "
                        "full scale, so defaults below are kept deliberately light. "
                        "Predicts the whole horizon in one pass per ensemble member, "
                        "same as N-BEATS."
                    )

                    tcol1, tcol2, tcol3 = st.columns(3)
                    with tcol1:
                        tf_lookback = st.number_input(
                            "Lookback window", min_value=2, value=10, key="widget_tf_lookback",
                        )
                    with tcol2:
                        tf_num_heads = st.number_input(
                            "Attention heads", min_value=1, value=2, key="widget_tf_num_heads",
                        )
                    with tcol3:
                        tf_key_dim = st.number_input(
                            "Key dimension (model width)", min_value=2, value=8, key="widget_tf_key_dim",
                        )

                    tcol4, tcol5, tcol6 = st.columns(3)
                    with tcol4:
                        tf_ff_dim = st.number_input(
                            "Feedforward dimension", min_value=2, value=16, key="widget_tf_ff_dim",
                        )
                    with tcol5:
                        tf_num_blocks = st.number_input(
                            "Encoder blocks", min_value=1, value=1, key="widget_tf_num_blocks",
                        )
                    with tcol6:
                        tf_ensemble_size = st.number_input(
                            "Ensemble size", min_value=2, value=3, key="widget_tf_ensemble_size",
                        )

                    tcol7, tcol8 = st.columns(2)
                    with tcol7:
                        tf_epochs = st.number_input(
                            "Training epochs", min_value=1, value=15, key="widget_tf_epochs",
                        )
                    with tcol8:
                        tf_cv_folds = st.number_input(
                            "K-fold CV folds", min_value=2, value=2, key="widget_tf_cv_folds",
                            help="More folds validate more thoroughly but take longer — "
                                 "choose based on your available compute.",
                        )

                    st.caption(
                        f"This trains {int(tf_ensemble_size)} independent Transformer "
                        "encoders. Attention cost grows faster than LSTM/N-BEATS as "
                        "lookback/width increase — if this is too slow on Colab free "
                        "tier, reduce lookback, heads, or key dimension first, or "
                        "consider Colab Pro for a real client run."
                    )

                    tf_run_mode = st.radio(
                        "What should this run do?",
                        options=[
                            "Validate against held-out actual data (recommended)",
                            "Forecast genuinely unknown future",
                        ],
                        key="widget_tf_run_mode",
                    )
                    tf_validate = tf_run_mode.startswith("Validate")
                    st.caption(f"Using H = **{int(horizon)}** (set above, in 'Forecast Setup').")

                    if st.button("Run Transformer Forecast", key="widget_tf_run_btn", type="primary"):
                        with st.spinner("Training Transformer ensemble — this may take a while..."):
                            try:
                                if tf_validate:
                                    tf_train_series = series[:-int(horizon)]
                                    tf_actual_holdout = series[-int(horizon):]
                                else:
                                    tf_train_series = series
                                    tf_actual_holdout = None

                                tf_result = transformer_forecast_module.transformer_forecast(
                                    tf_train_series, horizon=int(horizon),
                                    lookback=int(tf_lookback), num_heads=int(tf_num_heads),
                                    key_dim=int(tf_key_dim), ff_dim=int(tf_ff_dim),
                                    num_blocks=int(tf_num_blocks),
                                    ensemble_size=int(tf_ensemble_size), epochs=int(tf_epochs),
                                    confidence=confidence, cv_folds=int(tf_cv_folds),
                                )

                                if tf_validate:
                                    tf_final_df = pd.DataFrame({
                                        "relative_step": [f"t+{i}" for i in range(1, int(horizon) + 1)],
                                        f"{target_col}_actual": tf_actual_holdout,
                                        f"{target_col}_point_forecast": tf_result["point"],
                                        f"{target_col}_lower_bound": tf_result["lower"],
                                        f"{target_col}_upper_bound": tf_result["upper"],
                                    })
                                else:
                                    freq_code = data_processing.FREQ_LABEL_TO_CODE[confirmed_freq]
                                    future_dates = pd.date_range(
                                        start=dates.iloc[-1], periods=int(horizon) + 1, freq=freq_code
                                    )[1:]
                                    tf_final_df = pd.DataFrame({
                                        "Date": future_dates,
                                        f"{target_col}_point_forecast": tf_result["point"],
                                        f"{target_col}_lower_bound": tf_result["lower"],
                                        f"{target_col}_upper_bound": tf_result["upper"],
                                    })

                                st.session_state["cache_tf_result"] = tf_final_df
                                st.session_state["cache_tf_validate_mode"] = tf_validate
                                st.session_state["cache_tf_kfold_acc"] = tf_result["kfold_accuracy"]
                            except Exception as e:
                                st.error(f"Transformer forecast failed: {e}")

                    tf_result_df = st.session_state.get("cache_tf_result")
                    if tf_result_df is not None:
                        tf_was_validate = st.session_state.get("cache_tf_validate_mode", False)
                        tf_acc = st.session_state.get("cache_tf_kfold_acc")
                        if tf_acc is not None:
                            st.metric("Average K-fold CV accuracy", f"{tf_acc:.1f}%")
                        st.dataframe(tf_result_df, use_container_width=True, hide_index=True)

                        if tf_was_validate:
                            tf_fig, tf_png_bytes = forecast_plotting.plot_test_set_performance(
                                tf_result_df[f"{target_col}_actual"].values,
                                tf_result_df[f"{target_col}_point_forecast"].values,
                                tf_result_df[f"{target_col}_lower_bound"].values,
                                tf_result_df[f"{target_col}_upper_bound"].values,
                                target_col,
                            )
                        else:
                            tf_fig, tf_png_bytes = forecast_plotting.plot_out_of_sample_forecast(
                                tf_result_df[f"{target_col}_point_forecast"].values,
                                tf_result_df[f"{target_col}_lower_bound"].values,
                                tf_result_df[f"{target_col}_upper_bound"].values,
                                target_col,
                            )
                        st.image(tf_png_bytes, use_container_width=True)

                        tdl1, tdl2, tdl3 = st.columns(3)
                        with tdl1:
                            st.download_button(
                                "Download plot (.png)", data=tf_png_bytes,
                                file_name=f"{target_col}_transformer.png", mime="image/png",
                                key="widget_tf_dl_png",
                            )
                        with tdl2:
                            tf_csv_bytes = tf_result_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Download forecast (CSV)", data=tf_csv_bytes,
                                file_name=f"{target_col}_transformer.csv", mime="text/csv",
                                key="widget_tf_dl_csv",
                            )
                        with tdl3:
                            import io
                            tf_xlsx_buffer = io.BytesIO()
                            tf_result_df.to_excel(tf_xlsx_buffer, index=False, engine="openpyxl")
                            st.download_button(
                                "Download forecast (Excel)", data=tf_xlsx_buffer.getvalue(),
                                file_name=f"{target_col}_transformer.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="widget_tf_dl_xlsx",
                            )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
page = st.session_state["nav_page"]
if page == "home":
    render_home()
elif page == "collection":
    render_data_collection()
elif page == "preprocessing":
    render_data_preprocessing()
elif page == "forecasting":
    render_forecasting()
