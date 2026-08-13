"""Station 3/4 - fund metadata and fact-sheet outputs for AlphaStream.

Produces the investor-facing tables the report and the deployed app show:
snapshot metrics, month-by-month returns, fee-adjusted growth of $1, drawdown
series, and sector / top-holding breakdowns. All inputs are precomputed
walk-forward artifacts; this module never sees raw data.
"""
from __future__ import annotations

import pandas as pd

from . import portfolios

# ---------------------------------------------------------------------------
# Fund registry
# ---------------------------------------------------------------------------

MANAGEMENT_FEE = 0.0040          # 0.40% per annum, charged daily on NAV
SHORT_TERM_CAP_GAIN = 0.0        # no tax modelling; fees are the only drag

FUND_META = {
    "equity_equal_weight": {
        "family": "equity", "method": "equal_weight",
        "label": "Equity Equal Weight", "unit": "alpha_stream_eq_ew",
        "blurb": "Equal-weight exposure across the 50-name equity universe.",
    },
    "equity_min_variance": {
        "family": "equity", "method": "min_variance",
        "label": "Equity Minimum Variance", "unit": "alpha_stream_eq_mv",
        "blurb": "Lowest-volatility long-only equity portfolio.",
    },
    "equity_max_sharpe": {
        "family": "equity", "method": "max_sharpe",
        "label": "Equity Maximum Sharpe", "unit": "alpha_stream_eq_ms",
        "blurb": "Long-only equity portfolio maximising return per unit of risk.",
    },
    "equity_risk_parity": {
        "family": "equity", "method": "risk_parity",
        "label": "Equity Risk Parity", "unit": "alpha_stream_eq_rp",
        "blurb": "Equalises risk contribution across the equity universe.",
    },
    "equity_min_cvar": {
        "family": "equity", "method": "min_cvar",
        "label": "Equity Minimum CVaR", "unit": "alpha_stream_eq_cvar",
        "blurb": "Long-only equity portfolio minimising 95% expected shortfall.",
    },
    "crypto_equal_weight": {
        "family": "crypto", "method": "equal_weight",
        "label": "Crypto Equal Weight", "unit": "alpha_stream_cr_ew",
        "blurb": "Equal-weight exposure across the crypto panel.",
    },
    "crypto_min_variance": {
        "family": "crypto", "method": "min_variance",
        "label": "Crypto Minimum Variance", "unit": "alpha_stream_cr_mv",
        "blurb": "Lowest-volatility long-only crypto portfolio.",
    },
    "crypto_max_sharpe": {
        "family": "crypto", "method": "max_sharpe",
        "label": "Crypto Maximum Sharpe", "unit": "alpha_stream_cr_ms",
        "blurb": "Long-only crypto portfolio maximising return per unit of risk.",
    },
    "crypto_risk_parity": {
        "family": "crypto", "method": "risk_parity",
        "label": "Crypto Risk Parity", "unit": "alpha_stream_cr_rp",
        "blurb": "Equalises risk contribution across the crypto panel.",
    },
    "crypto_min_cvar": {
        "family": "crypto", "method": "min_cvar",
        "label": "Crypto Minimum CVaR", "unit": "alpha_stream_cr_cvar",
        "blurb": "Long-only crypto portfolio minimising 95% expected shortfall.",
    },
    "combined_equal_weight": {
        "family": "combined", "method": "equal_weight",
        "label": "Combined Equal Weight", "unit": "alpha_stream_cx_ew",
        "blurb": "Equal-weight across the full 60-name combined universe.",
    },
    "combined_min_variance": {
        "family": "combined", "method": "min_variance",
        "label": "Combined Minimum Variance", "unit": "alpha_stream_cx_mv",
        "blurb": "Lowest-volatility long-only portfolio over equities and crypto.",
    },
    "combined_max_sharpe": {
        "family": "combined", "method": "max_sharpe",
        "label": "Combined Maximum Sharpe", "unit": "alpha_stream_cx_ms",
        "blurb": "Flagship: maximises return per unit of risk over 60 names.",
    },
    "combined_risk_parity": {
        "family": "combined", "method": "risk_parity",
        "label": "Combined Risk Parity", "unit": "alpha_stream_cx_rp",
        "blurb": "Equalises risk contribution across equities and crypto.",
    },
    "combined_min_cvar": {
        "family": "combined", "method": "min_cvar",
        "label": "Combined Minimum CVaR", "unit": "alpha_stream_cx_cvar",
        "blurb": "Long-only portfolio minimising 95% expected shortfall.",
    },
    "equity_sentiment_tilt": {
        "family": "sentiment", "method": "sentiment_tilt",
        "label": "Equity Sentiment Tilt", "unit": "alpha_stream_eq_st",
        "blurb": "Equity max-Sharpe tilted toward high-sentiment names.",
    },
    "equity_sentiment_momentum": {
        "family": "sentiment", "method": "sentiment_momentum",
        "label": "Equity Sentiment Momentum", "unit": "alpha_stream_eq_sm",
        "blurb": "Long top-quintile / short bottom-quintile sentiment fund.",
    },
}


def fund_name(family: str, method: str) -> str:
    return f"{family}_{method}"


# ---------------------------------------------------------------------------
# Fact-sheet builders (all take precomputed artifacts only)
# ---------------------------------------------------------------------------

def growth_of_one(returns: pd.Series, fee: float = MANAGEMENT_FEE, start: float = 1.0) -> pd.Series:
    """Daily fee-adjusted growth of $1."""
    daily_fee = (1.0 + fee) ** (1 / portfolios.TRADING_DAYS) - 1.0
    net = returns - daily_fee
    return start * (1.0 + net).cumprod()


def snapshot(backtest: portfolios.Backtest, *, market_label: str = "Equal-weight") -> pd.DataFrame:
    """One-row snapshot of a fund: headline metrics plus 1y/3y/ITD rows."""
    metrics = portfolios.performance_metrics(backtest.returns)
    value = growth_of_one(backtest.returns)
    rows = [{
        "metric": "Total return (ITD)", "value": metrics["total_return"],
        "unit": "%", "format": "pct",
    }, {
        "metric": "CAGR", "value": metrics["cagr"], "unit": "%", "format": "pct",
    }, {
        "metric": "Annualised volatility", "value": metrics["ann_vol"],
        "unit": "%", "format": "pct",
    }, {
        "metric": "Sharpe ratio", "value": metrics["sharpe"], "unit": "", "format": "num",
    }, {
        "metric": "Max drawdown", "value": metrics["max_drawdown"], "unit": "%", "format": "pct",
    }, {
        "metric": "Calmar", "value": metrics["calmar"], "unit": "", "format": "num",
    }, {
        "metric": "Growth of $1 (net of fees)", "value": float(value.iloc[-1]),
        "unit": "$", "format": "num",
    }]
    return pd.DataFrame(rows)


def monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """Year x month matrix of gross monthly returns (blanks for empty months)."""
    monthly = returns.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
    out = monthly.to_frame("monthly").copy()
    out["year"] = out.index.year
    out["month"] = out.index.month
    piv = out.pivot_table(index="year", columns="month", values="monthly")
    piv.columns = [f"{m:02d}" for m in piv.columns]
    return piv


def sector_exposure(weights_row: pd.Series, sector_map: pd.DataFrame) -> pd.DataFrame:
    """Sector weights of a single weight row (ticker -> sector)."""
    w = weights_row.copy()
    mapping = sector_map.set_index("ticker")["sector"].to_dict()
    w.index = [mapping.get(t, "Other") for t in w.index]
    sector_w = w.groupby(level=0).sum()
    sector_w = sector_w.sort_values(ascending=False)
    return sector_w.to_frame("weight").reset_index().rename(columns={"index": "sector"})


def top_holdings(weights_row: pd.Series, n: int = 10) -> pd.DataFrame:
    """Top n holdings of a weight row with sector tags."""
    return weights_row.sort_values(ascending=False).head(n).to_frame("weight").reset_index()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Daily drawdown series from gross returns."""
    cum = (1.0 + returns).cumprod()
    return cum / cum.cummax() - 1.0


def asset_exposure(weights_row: pd.Series, family_map: dict[str, str]) -> pd.DataFrame:
    """Aggregate weights by asset family (equity vs crypto)."""
    grouped = weights_row.groupby(weights_row.index.map(lambda t: family_map.get(t, "other"))).sum()
    return grouped.to_frame("weight").reset_index().rename(columns={"index": "asset_class"})
