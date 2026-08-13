"""Station 2 - Features: return features and headline text assembly.

Re-uses the Part A foundation for Part B: daily returns per panel (equities on
the 252-day calendar, crypto on its own 365-day calendar first), a
left-merge of crypto returns onto the equity trading calendar, and the daily
headline panel aligned to equity trading days. The sentiment model that scores
those headlines lives in src/sentiment.py (Station 3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(prices: pd.DataFrame, price_col: str = "adjClose") -> pd.DataFrame:
    """Compute simple daily returns per ticker from a long price panel.

    Returns a wide DataFrame (date x ticker) of simple daily returns built from
    adjusted close, so splits and dividends are accounted for.
    """
    wide = (
        prices.pivot_table(index="date", columns="ticker", values=price_col)
        .sort_index()
        .sort_index(axis=1)
    )
    returns = wide.pct_change()
    returns.index.name = "date"
    return returns


def merge_crypto_to_equity(
    equity_returns: pd.DataFrame,
    crypto_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Left-merge crypto returns onto the equity trading-day calendar.

    Crypto returns are computed on the 365-day calendar first, then reindexed
    to equity dates. This intentionally drops weekend-only crypto moves, which
    a fund trading on equity days could not act on.
    """
    aligned_crypto = crypto_returns.reindex(equity_returns.index)
    combined = pd.concat(
        [equity_returns, aligned_crypto.add_prefix("CR_")],
        axis=1,
    )
    return combined


def split_combined(combined: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (equity columns, crypto columns) from a combined wide panel."""
    cr_cols = [c for c in combined.columns if c.startswith("CR_")]
    eq_cols = [c for c in combined.columns if not c.startswith("CR_")]
    return eq_cols, cr_cols


def assemble_headline_panel(
    headlines: pd.DataFrame,
    equity_trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Assemble headlines into a daily panel per ticker and sector.

    Each headline is mapped to its equity trading day (the same day if it is a
    trading day, otherwise the next trading day). Raw headline text is
    preserved for Station 3 sentiment scoring - nothing is stripped.
    """
    trading_dates_sorted = pd.DatetimeIndex(sorted(equity_trading_dates))

    df = headlines.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    idx_array = df["date"].map(
        lambda d: int(trading_dates_sorted.searchsorted(d, side="left"))
    ).to_numpy()
    idx_array = np.clip(idx_array, 0, len(trading_dates_sorted) - 1)
    df["trading_date"] = pd.DatetimeIndex([trading_dates_sorted[i] for i in idx_array])

    df = df.dropna(subset=["trading_date"])
    df = df.sort_values(["trading_date", "ticker", "title"]).reset_index(drop=True)

    daily_counts = (
        df.groupby(["trading_date", "ticker", "sector"])
        .agg(headline_count=("title", "size"), unique_publishers=("publisher", "nunique"))
        .reset_index()
    )
    daily_text = (
        df.groupby(["trading_date", "ticker", "sector"])
        .agg(headlines=("title", list))
        .reset_index()
    )
    panel = daily_counts.merge(daily_text, on=["trading_date", "ticker", "sector"])
    panel = panel.rename(columns={"trading_date": "date"})
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel


def news_flow_spillover_index(
    headlines: pd.DataFrame,
    equity_trading_dates: pd.DatetimeIndex,
    window: int = 30,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute the Part A Cross-Sector News Flow Spillover Index.

    For each equity trading day, count headlines per sector, then take the
    rolling pairwise correlation of headline counts across the 10 sectors. The
    index is the mean off-diagonal correlation in the window: high positive
    values mean news is arriving systemically across sectors, low values mean
    sector-specific (idiosyncratic) news flow.

    Returns a DataFrame with columns [date, spillover_index, n_sectors].
    """
    if min_periods is None:
        min_periods = max(window // 2, 10)

    trading_dates_sorted = pd.DatetimeIndex(sorted(equity_trading_dates))
    df = headlines.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    idx_array = df["date"].map(
        lambda d: int(trading_dates_sorted.searchsorted(d, side="left"))
    ).to_numpy()
    idx_array = np.clip(idx_array, 0, len(trading_dates_sorted) - 1)
    df["trading_date"] = pd.DatetimeIndex([trading_dates_sorted[i] for i in idx_array])

    sector_daily = df.groupby(["trading_date", "sector"]).size().reset_index(name="headline_count")
    sector_wide = sector_daily.pivot_table(
        index="trading_date", columns="sector", values="headline_count", fill_value=0
    ).reindex(trading_dates_sorted, fill_value=0)

    corr_matrix = sector_wide.rolling(window=window, min_periods=min_periods).corr()

    results = []
    for date in trading_dates_sorted:
        try:
            day_corr = corr_matrix.loc[date]
        except KeyError:
            continue
        if isinstance(day_corr.index, pd.MultiIndex):
            continue
        mask = np.ones(day_corr.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        off_diag = day_corr.values[mask]
        off_diag = off_diag[np.isfinite(off_diag)]
        if len(off_diag) > 0:
            results.append({
                "date": date,
                "spillover_index": float(np.mean(off_diag)),
                "n_sectors": int(sector_wide.shape[1]),
            })
    return pd.DataFrame(results)


def cross_asset_lead_lag(
    combined_returns: pd.DataFrame,
    max_lag: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure lead-lag correlations between a crypto composite and equity returns.

    The crypto composite is the equal-weight average of BTC-USD and ETH-USD
    returns on the equity trading calendar. For each equity ticker and each lag
    k in [-max_lag, +max_lag], the correlation between the composite at t and
    the equity return at t+k is computed. Positive k means crypto leads.
    """
    eq_cols, cr_cols = split_combined(combined_returns)
    btc_col = "CR_BTC-USD" if "CR_BTC-USD" in cr_cols else cr_cols[0]
    eth_col = "CR_ETH-USD" if "CR_ETH-USD" in cr_cols else cr_cols[1]
    crypto_avg = combined_returns[[btc_col, eth_col]].mean(axis=1)
    crypto_avg.name = "crypto_avg"

    lags = range(-max_lag, max_lag + 1)
    records = []
    for ticker in eq_cols:
        eq_ret = combined_returns[ticker]
        row = {"ticker": ticker}
        for lag in lags:
            row[lag] = crypto_avg.shift(lag).corr(eq_ret)
        records.append(row)

    ll_df = pd.DataFrame(records).set_index("ticker")
    ll_df.columns.name = "lag"
    ll_df.index.name = "ticker"

    summary_records = []
    for ticker in eq_cols:
        row_data = ll_df.loc[ticker]
        abs_row = row_data.abs()
        summary_records.append({
            "ticker": ticker,
            "max_pos_lag": int(row_data.idxmax()),
            "max_pos_corr": round(float(row_data.max()), 4),
            "max_neg_lag": int(row_data.idxmin()),
            "max_neg_corr": round(float(row_data.min()), 4),
            "avg_abs_corr": round(float(abs_row.mean()), 4),
        })
    summary_df = (
        pd.DataFrame(summary_records)
        .sort_values("avg_abs_corr", ascending=False)
        .reset_index(drop=True)
    )
    return ll_df.reset_index(), summary_df
