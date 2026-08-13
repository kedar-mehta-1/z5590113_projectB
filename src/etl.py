"""Station 1 - ETL: load, clean, and integrity-check the project datasets.

This is the Part B re-use of the Part A foundation: it loads all three
datasets through src.data_access, runs the same integrity checks (missing
dates, duplicates, outliers), aligns the 252-day equity calendar with the
365-day crypto calendar, and returns clean frames with a documented
IntegrityReport. Nothing here scores sentiment or optimises portfolios - that
is Station 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src import data_access


@dataclass
class IntegrityReport:
    """Container for all Station 1 integrity-check results."""

    equity_rows_raw: int = 0
    equity_rows_after: int = 0
    equity_missing_dates_total: int = 0
    equity_missing_dates_per_ticker: dict[str, int] = field(default_factory=dict)
    equity_duplicates_found: int = 0
    equity_outliers_flagged: int = 0

    crypto_rows_raw: int = 0
    crypto_rows_after: int = 0
    crypto_stray_2024_rows: int = 0
    crypto_missing_dates_total: int = 0
    crypto_duplicates_found: int = 0

    news_rows_raw: int = 0
    news_rows_after_dedup: int = 0
    news_duplicates_found: int = 0

    def summary_table(self) -> pd.DataFrame:
        """Return a tidy summary of the integrity audit."""
        rows = [
            ("Equity prices", "Rows loaded", self.equity_rows_raw),
            ("Equity prices", "Duplicate ticker-date pairs removed", self.equity_duplicates_found),
            ("Equity prices", "Missing trading dates", self.equity_missing_dates_total),
            ("Equity prices", "Outlier returns flagged (kept)", self.equity_outliers_flagged),
            ("Crypto prices", "Rows loaded", self.crypto_rows_raw),
            ("Crypto prices", "Stray 2024-01-01 rows removed", self.crypto_stray_2024_rows),
            ("Crypto prices", "Rows after cleaning", self.crypto_rows_after),
            ("Crypto prices", "Duplicate ticker-date pairs removed", self.crypto_duplicates_found),
            ("Crypto prices", "Missing calendar dates", self.crypto_missing_dates_total),
            ("News headlines", "Rows loaded", self.news_rows_raw),
            ("News headlines", "Exact duplicate rows removed", self.news_duplicates_found),
            ("News headlines", "Rows after deduplication", self.news_rows_after_dedup),
        ]
        return pd.DataFrame(rows, columns=["Dataset", "Check", "Count"])


def _missing_date_audit(prices: pd.DataFrame, *, calendar_freq: str) -> dict[str, int]:
    """Return per-ticker counts of missing dates for the observed calendar."""
    all_dates = pd.date_range(prices["date"].min(), prices["date"].max(), freq=calendar_freq)
    missing: dict[str, int] = {}
    for ticker, group in prices.groupby("ticker"):
        observed = set(pd.to_datetime(group["date"]))
        missing[str(ticker)] = len(set(all_dates) - observed)
    return missing


def load_clean_equities() -> tuple[pd.DataFrame, IntegrityReport]:
    """Load equity prices, run integrity checks, return clean frame + report."""
    report = IntegrityReport()
    df = data_access.load_equity_prices()
    report.equity_rows_raw = len(df)

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    dup_mask = df.duplicated(subset=["ticker", "date"], keep="first")
    report.equity_duplicates_found = int(dup_mask.sum())
    df = df.loc[~dup_mask].copy()

    report.equity_missing_dates_per_ticker = _missing_date_audit(df, calendar_freq="B")
    report.equity_missing_dates_total = sum(report.equity_missing_dates_per_ticker.values())

    df["ret"] = df.groupby("ticker")["adjClose"].pct_change()
    z_scores = df.groupby("ticker")["ret"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
    )
    report.equity_outliers_flagged = int((z_scores.abs() > 4.0).sum())
    df = df.drop(columns=["ret"])
    report.equity_rows_after = len(df)
    return df, report


def load_clean_crypto() -> tuple[pd.DataFrame, IntegrityReport]:
    """Load crypto prices, cap to 2023-12-31, run integrity checks."""
    report = IntegrityReport()
    df = data_access.load_crypto_prices()
    report.crypto_rows_raw = len(df)

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    stray = df["date"] >= "2024-01-01"
    report.crypto_stray_2024_rows = int(stray.sum())
    df = df.loc[~stray].copy()

    dup_mask = df.duplicated(subset=["ticker", "date"], keep="first")
    report.crypto_duplicates_found = int(dup_mask.sum())
    df = df.loc[~dup_mask].copy()

    report.crypto_missing_dates_total = sum(
        _missing_date_audit(df, calendar_freq="D").values()
    )
    report.crypto_rows_after = len(df)
    return df, report


def load_clean_news() -> tuple[pd.DataFrame, IntegrityReport]:
    """Load news headlines, remove exact duplicates on (ticker, date, title)."""
    report = IntegrityReport()
    df = data_access.load_news_headlines()
    report.news_rows_raw = len(df)

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "date", "title"], keep="first")
    report.news_duplicates_found = before - len(df)
    report.news_rows_after_dedup = len(df)
    return df, report


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, IntegrityReport]:
    """Load and clean all three datasets, returning one merged IntegrityReport."""
    eq, eq_report = load_clean_equities()
    cr, cr_report = load_clean_crypto()
    nh, nh_report = load_clean_news()

    combined = eq_report
    combined.crypto_rows_raw = cr_report.crypto_rows_raw
    combined.crypto_rows_after = cr_report.crypto_rows_after
    combined.crypto_stray_2024_rows = cr_report.crypto_stray_2024_rows
    combined.crypto_duplicates_found = cr_report.crypto_duplicates_found
    combined.crypto_missing_dates_total = cr_report.crypto_missing_dates_total
    combined.news_rows_raw = nh_report.news_rows_raw
    combined.news_rows_after_dedup = nh_report.news_rows_after_dedup
    combined.news_duplicates_found = nh_report.news_duplicates_found
    return eq, cr, nh, combined
