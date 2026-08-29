"""
Module 01: Stock Data Collector
--------------------------------
Standalone module for collecting historical stock price data via yfinance.

Anti-blocking measures built in:
  1. Batch download (yfinance pulls multiple tickers in ONE request round
     instead of one request per ticker) — fewer total requests.
  2. Browser-impersonating session via curl_cffi — Yahoo Finance has
     tightened bot detection on its endpoints; a plain requests session
     is more likely to get 429'd/blocked than one that looks like a real
     browser (Chrome) making the request.
  3. Randomized delay between requests when falling back to per-ticker
     looping — avoids a suspiciously regular request pattern.
  4. Retry with exponential backoff on failure, instead of hammering the
     endpoint again immediately.
  5. Raw data is cached to disk with a versioned filename — re-running the
     script won't needlessly re-request data you already have.

Does NOT do: high-concurrency/parallel scraping. That's the single
biggest way to get blocked. Everything here is deliberately sequential
and paced.

Usage (standalone test):
    python 01_data_ingestion.py
"""

import time
import random
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# --- Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stock_collector")

RAW_DATA_DIR = Path("data/raw")
CLEAN_DATA_DIR = Path("data/clean")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Hard cap on tickers per collection run. This isn't arbitrary: yfinance's
# batch download already fetches all tickers in one request round, but a
# very large ticker list still means a very large response and a higher
# chance something in the batch trips Yahoo's anti-bot detection. Keeping
# individual runs modest is part of staying polite to the data source so
# we don't jeopardize being able to collect data at all in the future.
MAX_TICKERS = 10


def get_yf_session():
    """
    Returns a browser-impersonating session for yfinance requests.
    Falls back to yfinance's default session if curl_cffi isn't installed.
    """
    try:
        from curl_cffi import requests as cffi_requests
        session = cffi_requests.Session(impersonate="chrome")
        logger.info("Using curl_cffi browser-impersonating session.")
        return session
    except ImportError:
        logger.warning(
            "curl_cffi not installed — falling back to yfinance's default "
            "session. Install with: pip install curl_cffi"
        )
        return None


def fetch_single_ticker(ticker: str, start: str, end: str, interval: str = "1d",
                         session=None, max_retries: int = 3, base_delay: float = 2.0) -> pd.DataFrame:
    """Fetch historical OHLCV data for one ticker, with retry + exponential backoff.

    interval: yfinance interval string, e.g. "1d" (daily), "1wk" (weekly),
    "1mo" (monthly). Intraday intervals (e.g. "1h") exist in yfinance but
    Yahoo restricts how far back they can be requested, so they're left out
    of the default UI options for now.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Fetching {ticker} interval={interval} (attempt {attempt}/{max_retries})...")
            df = yf.download(
                ticker, start=start, end=end, interval=interval,
                progress=False, session=session, auto_adjust=True,
            )
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")
            df["ticker"] = ticker
            return df
        except Exception as e:
            wait = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
            logger.warning(f"Error fetching {ticker}: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    logger.error(f"Failed to fetch {ticker} after {max_retries} attempts.")
    return pd.DataFrame()


def fetch_multiple_tickers(tickers: list, start: str, end: str, interval: str = "1d",
                            delay_range=(1.5, 3.5), use_batch: bool = True) -> dict:
    """
    Fetch historical data for multiple tickers.

    use_batch=True (recommended, default): one batched request for all
        tickers via yfinance's multi-ticker download — fewest total requests.
    use_batch=False, or if batch fails: loops tickers individually with a
        randomized delay between each call.
    """
    session = get_yf_session()
    results = {}

    if use_batch and len(tickers) > 1:
        try:
            logger.info(f"Batch downloading {len(tickers)} tickers: {tickers} (interval={interval})")
            data = yf.download(
                tickers, start=start, end=end, interval=interval, group_by="ticker",
                progress=False, session=session, auto_adjust=True,
            )
            got_any = False
            for t in tickers:
                try:
                    df = data[t].copy()
                    if df.dropna(how="all").empty:
                        raise KeyError
                    df["ticker"] = t
                    results[t] = df
                    got_any = True
                except KeyError:
                    logger.warning(f"No data for {t} in batch result — will retry individually.")
            if got_any:
                # Fill in any tickers the batch call missed
                missing = [t for t in tickers if t not in results]
                for t in missing:
                    results[t] = fetch_single_ticker(t, start, end, interval=interval, session=session)
                    time.sleep(random.uniform(*delay_range))
                return results
        except Exception as e:
            logger.warning(f"Batch download failed ({e}), falling back to per-ticker loop.")

    # Per-ticker loop (fallback, or explicit use_batch=False)
    for t in tickers:
        results[t] = fetch_single_ticker(t, start, end, interval=interval, session=session)
        delay = random.uniform(*delay_range)
        logger.info(f"Sleeping {delay:.1f}s before next ticker...")
        time.sleep(delay)

    return results


def _save_excel_if_possible(df: pd.DataFrame, csv_path: Path) -> Path | None:
    """
    Best-effort Excel export alongside a CSV save. Requires openpyxl.
    Returns the xlsx path, or None if the export couldn't be done (missing
    dependency) — this never blocks the CSV save, which stays the primary,
    guaranteed output.
    """
    try:
        xlsx_path = csv_path.with_suffix(".xlsx")
        df.to_excel(xlsx_path, engine="openpyxl")
        logger.info(f"Saved Excel version: {xlsx_path}")
        return xlsx_path
    except ImportError:
        logger.warning(
            "openpyxl not installed — skipped Excel export. "
            "Install with: pip install openpyxl"
        )
        return None


def save_raw(df: pd.DataFrame, ticker: str, save_excel: bool = False) -> dict:
    """
    Save raw, untouched data with a versioned filename for reproducibility.
    Always saves CSV; optionally also saves an .xlsx copy alongside it.
    Returns {"csv": Path, "xlsx": Path or None}.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    csv_path = RAW_DATA_DIR / f"{ticker.lower()}_raw_{date_str}.csv"
    df.to_csv(csv_path)
    logger.info(f"Saved raw data: {csv_path}")
    xlsx_path = _save_excel_if_possible(df, csv_path) if save_excel else None
    return {"csv": csv_path, "xlsx": xlsx_path}


def basic_quality_check(df: pd.DataFrame, ticker: str, interval: str = "1d") -> dict:
    """Basic data quality checks: duplicate dates, missing business days.

    The missing-business-days check assumes daily data; for weekly/monthly
    intervals it's not a meaningful signal, so it's skipped in that case.
    """
    report = {"rows": len(df), "duplicate_dates": int(df.index.duplicated().sum())}
    report["date_range"] = (str(df.index.min()), str(df.index.max()))
    if interval == "1d":
        full_range = pd.bdate_range(df.index.min(), df.index.max())
        missing = full_range.difference(df.index)
        # NOTE: this counts all non-trading business days (incl. market holidays)
        # as "missing" — it's a rough signal, not an exact gap count. Genuine
        # unexpected gaps (vs. known holidays) get handled properly in Module 2.
        report["missing_business_days_approx"] = len(missing)
    else:
        report["missing_business_days_approx"] = None
    logger.info(f"[{ticker}] Quality check: {report}")
    return report


def save_clean(df: pd.DataFrame, ticker: str, save_excel: bool = False) -> dict:
    """
    Save a de-duplicated, sorted version separately from raw (raw stays untouched).
    Always saves CSV; optionally also saves an .xlsx copy alongside it.
    Returns {"csv": Path, "xlsx": Path or None}.
    """
    df_clean = df[~df.index.duplicated(keep="first")].sort_index()
    date_str = datetime.now().strftime("%Y-%m-%d")
    csv_path = CLEAN_DATA_DIR / f"{ticker.lower()}_clean_{date_str}.csv"
    df_clean.to_csv(csv_path)
    logger.info(f"Saved clean data: {csv_path}")
    xlsx_path = _save_excel_if_possible(df_clean, csv_path) if save_excel else None
    return {"csv": csv_path, "xlsx": xlsx_path}


def collect_stock_data(tickers: list, start: str, end: str, interval: str = "1d",
                        save_excel: bool = False) -> dict:
    """
    Orchestrates the full collection for one or more tickers:
    fetch -> save raw -> quality check -> save clean.

    interval: "1d" (daily), "1wk" (weekly), or "1mo" (monthly).

    Returns a dict keyed by ticker, each value a dict with:
      "data": clean DataFrame
      "quality_report": dict from basic_quality_check
      "raw_paths": {"csv": Path, "xlsx": Path or None}
      "clean_paths": {"csv": Path, "xlsx": Path or None}
    Tickers that failed to fetch are omitted (see logs for the reason).
    """
    if len(tickers) > MAX_TICKERS:
        raise ValueError(
            f"Requested {len(tickers)} tickers, but the cap is {MAX_TICKERS} per "
            "collection run. This keeps request volume polite to the data source "
            "and reduces the risk of getting rate-limited or blocked in the "
            "future. Split into multiple smaller runs if you need more tickers."
        )

    raw_results = fetch_multiple_tickers(tickers, start, end, interval=interval)
    results = {}

    for ticker, df in raw_results.items():
        if df.empty:
            logger.error(f"Skipping {ticker} — no data collected.")
            continue
        raw_paths = save_raw(df, ticker, save_excel=save_excel)
        quality_report = basic_quality_check(df, ticker, interval=interval)
        clean_paths = save_clean(df, ticker, save_excel=save_excel)
        clean_df = pd.read_csv(clean_paths["csv"], index_col=0, parse_dates=True)
        results[ticker] = {
            "data": clean_df,
            "quality_report": quality_report,
            "raw_paths": raw_paths,
            "clean_paths": clean_paths,
        }

    return results


def combine_price_series(results: dict, field: str = "Close") -> pd.DataFrame:
    """
    Combine one price field (default: Close, which is split/dividend-adjusted
    since collection uses auto_adjust=True) from multiple tickers' results
    into a single wide-format table: one column per ticker, aligned by date.

    This does NOT make any new network request — it only reshapes data
    already collected in this run, so it carries no additional blocking risk.

    Dates where a given ticker has no data (e.g. it started trading later,
    or a holiday specific to its exchange) become NaN in that column —
    real gap-handling happens in Module 2, not here.
    """
    if not results:
        return pd.DataFrame()
    series_list = []
    for ticker, result in results.items():
        df = result["data"]
        if field not in df.columns:
            logger.warning(f"{ticker}: field '{field}' not found — skipped in combined table.")
            continue
        series_list.append(df[field].rename(ticker))
    if not series_list:
        return pd.DataFrame()
    return pd.concat(series_list, axis=1).sort_index()


def save_combined(df: pd.DataFrame, save_excel: bool = False) -> dict:
    """
    Save the combined multi-ticker price table (one column per ticker).
    Always saves CSV; optionally also saves an .xlsx copy alongside it.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    csv_path = CLEAN_DATA_DIR / f"combined_prices_{date_str}.csv"
    df.to_csv(csv_path)
    logger.info(f"Saved combined price table: {csv_path}")
    xlsx_path = _save_excel_if_possible(df, csv_path) if save_excel else None
    return {"csv": csv_path, "xlsx": xlsx_path}


if __name__ == "__main__":
    # Standalone test block — run this file directly to sanity-check the module
    # before wiring it into anything else, per the project's dev methodology.
    TICKERS = ["TSLA", "AAPL"]
    START = "2023-01-01"
    END = "2026-08-01"

    data = collect_stock_data(TICKERS, START, END, save_excel=True)

    for ticker, result in data.items():
        df = result["data"]
        print(f"\n{ticker}: {len(df)} rows")
        print(df.head())
        print("Quality report:", result["quality_report"])
        print("Clean files:", result["clean_paths"])
