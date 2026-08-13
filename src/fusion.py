"""Station 4 (extension) - fuse the sentiment signal into the funds.

Two look-ahead-safe fusion products sit on top of the walk-forward machinery
in portfolios.py:

    * Sentiment tilt fund. The base weights are the equity max-Sharpe fund;
      at every monthly decision date the weights are tilted toward names with
      a high cross-sectional sentiment signal. The tilt is long-only and
      renormalized, so it stays a real investable fund (no shorting, same 10%
      per-name cap logic after renormalization).
    * Sentiment momentum fund. A market-neutral long-short basket respecting
      the brief's 60% short cap: top-quintile names at 6% long each, bottom-
      quintile names at 6% short each, 40% cash. This is the "sentiment
      momentum factor" tradable as a fund.

Signal definition (used by both products):

    s_i(t) = mean of the last S=5 trading days of ticker i's lagged sentiment
    index, where the index is already lagged one trading day. Weights for
    decision day d use history strictly before d, so the newest sentiment
    feeding the signal is from day d-1 or earlier - no look-ahead.

    The raw signal is transformed to a cross-sectional z-score (winsorized at
    +-2) before tilting, so a date's tilt depends only on the relative
    sentiment of names that day, not on the absolute level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import portfolios

SIGNAL_LOOKBACK = 5
TILT_INTENSITY = 0.5
SHORT_CAP = 0.6
QUINTILE_SIZE = 10


def sentiment_signal(
    signal_wide: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    lookback: int = SIGNAL_LOOKBACK,
) -> pd.Series:
    """5-day mean lagged sentiment per ticker as of `as_of` (inclusive)."""
    window = signal_wide.loc[:as_of].tail(lookback)
    return window.mean(axis=0)


def cross_sectional_z(series: pd.Series, clip: float = 2.0) -> pd.Series:
    """Cross-sectional z-score of a signal, winsorized at +-clip."""
    mu = series.mean()
    sd = series.std(ddof=0)
    z = (series - mu) / (sd + 1e-9)
    return z.clip(-clip, clip)


def sentiment_tilt_transform(signal_wide: pd.DataFrame, intensity: float = TILT_INTENSITY):
    """Build a weight transform that tilts base weights toward high sentiment."""

    def transform(w: pd.Series, history: pd.DataFrame) -> pd.Series:
        as_of = history.index[-1]
        signal = sentiment_signal(signal_wide, as_of).reindex(w.index).fillna(0.0)
        z = cross_sectional_z(signal)
        tilted = w * (1.0 + intensity * z)
        tilted = np.maximum(tilted, 0.0)
        if tilted.sum() > 0:
            tilted = tilted / tilted.sum()
        return tilted

    return transform


def momentum_transform(signal_wide: pd.DataFrame, cap: float = SHORT_CAP):
    """Long-short transform: long top quintile, short bottom quintile."""

    def transform(w: pd.Series, history: pd.DataFrame) -> pd.Series:
        as_of = history.index[-1]
        signal = sentiment_signal(signal_wide, as_of).reindex(w.index).fillna(0.0)
        ranking = signal.rank(ascending=False)
        n = len(w)
        weights = np.zeros(n)
        top = ranking.to_numpy() <= QUINTILE_SIZE
        bot = ranking.to_numpy() > (n - QUINTILE_SIZE)
        weights[top] = cap / QUINTILE_SIZE
        weights[bot] = -cap / QUINTILE_SIZE
        return pd.Series(weights, index=w.index)

    return transform


def build_fusion_funds(
    equity_returns: pd.DataFrame,
    signal_wide: pd.DataFrame,
    *,
    min_training: int = portfolios.MIN_TRAINING,
    base_method: str = "max_sharpe",
) -> dict[str, portfolios.Backtest]:
    """Build the tilt fund and the momentum fund from the equity panel.

    Returns a dict keyed by fund name with the walk-forward Backtest objects.
    """
    tilt_bt = portfolios.walk_forward_backtest(
        equity_returns, base_method, "equity_sentiment_tilt",
        min_training=min_training,
        transform=sentiment_tilt_transform(signal_wide),
    )
    momentum_bt = portfolios.walk_forward_backtest(
        equity_returns, "equal_weight", "equity_sentiment_momentum",
        min_training=min_training,
        transform=momentum_transform(signal_wide),
    )
    return {"equity_sentiment_tilt": tilt_bt, "equity_sentiment_momentum": momentum_bt}


def fusion_comparison(
    base: portfolios.Backtest,
    tilted: portfolios.Backtest,
) -> pd.DataFrame:
    """Metrics table comparing a base fund with its sentiment-tilted version."""
    rows = []
    for name, bt in (("base", base), ("sentiment_tilt", tilted)):
        m = portfolios.performance_metrics(bt.returns)
        rows.append({
            "fund": name,
            "cagr": m["cagr"],
            "ann_vol": m["ann_vol"],
            "sharpe": m["sharpe"],
            "max_drawdown": m["max_drawdown"],
            "calmar": m["calmar"],
        })
    return pd.DataFrame(rows)
