"""Station 3 - Out-of-sample portfolio construction and evaluation.

Walk-forward, out-of-sample backtests for the AlphaStream equity, crypto, and
combined fund families.

Conventions:

    * Rebalancing is monthly on the equity trading calendar (the combined
      panel's trading day). Weights are decided at the close of day t using
      strictly prior data (history rows with index < t) and are applied from
      day t onward until the next decision date - no look-ahead.
    * Each fund is long-only with a 10% per-name cap. The brief allows shorting
      up to 60% for a strategy fund; sentiment funds in fusion.py may use that,
      the core funds do not.
    * Every method is solved from the same expanding window so the method zoo
      is directly comparable. Training needs at least MIN_TRAINING days.
    * Risk model: shrinkage covariance (a diagonal ridge added to the sample
      covariance) so min-variance / max-Sharpe / risk-parity are stable on
      50-60 names with ~2 years of history.

Performance metrics are annualized on the equity calendar (252 trading days).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

TRADING_DAYS = 252
MIN_TRAINING = 504
COV_LOOKBACK = 504
CVAR_LOOKBACK = 252
MIN_VALID_OBS = 120
RIDGE_SHRINKAGE = 0.25

Method = Literal["equal_weight", "min_variance", "max_sharpe", "risk_parity", "min_cvar"]
FUND_FAMILIES = ("equity", "crypto", "combined")
METHODS = ("equal_weight", "min_variance", "max_sharpe", "risk_parity", "min_cvar")


def shrinkage_covariance(returns: pd.DataFrame, lookback: int = COV_LOOKBACK) -> np.ndarray:
    """Shrunk covariance of the last `lookback` daily returns (as matrix)."""
    window = returns.tail(lookback).to_numpy()
    if window.shape[0] < 60:
        raise ValueError(f"need at least 60 rows of history, got {window.shape[0]}")
    cov = np.cov(window, rowvar=False)
    avg_var = float(np.mean(np.diag(cov)))
    cov = cov + RIDGE_SHRINKAGE * avg_var * np.eye(cov.shape[0])
    return cov


def _bounds(n: int, cap: float) -> list[tuple[float, float]]:
    return [(0.0, cap)] * n


def _sum_constraint(n: int):
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones(n)}


def _sharpe_grad(w, mu, cov):
    num = mu @ w
    s2 = w @ cov @ w
    s = np.sqrt(s2)
    return mu / s - (num * cov @ w) / (s2 * s)


def equal_weight_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def min_variance_weights(returns: pd.DataFrame, cap: float = 0.10) -> np.ndarray:
    n = returns.shape[1]
    cov = shrinkage_covariance(returns)
    res = minimize(
        lambda w: w @ cov @ w, np.full(n, 1.0 / n),
        jac=lambda w: 2 * cov @ w,
        bounds=_bounds(n, cap), constraints=[_sum_constraint(n)],
        method="SLSQP", options={"maxiter": 300, "ftol": 1e-9},
    )
    return res.x


def max_sharpe_weights(returns: pd.DataFrame, cap: float = 0.10) -> np.ndarray:
    n = returns.shape[1]
    mu = returns.tail(COV_LOOKBACK).mean().to_numpy()
    cov = shrinkage_covariance(returns)

    def obj(w):
        s2 = w @ cov @ w
        if s2 <= 0:
            return 1e6
        return -(mu @ w) / np.sqrt(s2)

    res = minimize(
        obj, np.full(n, 1.0 / n),
        jac=lambda w: -_sharpe_grad(w, mu, cov),
        bounds=_bounds(n, cap), constraints=[_sum_constraint(n)],
        method="SLSQP", options={"maxiter": 400, "ftol": 1e-10},
    )
    return res.x


def risk_parity_weights(returns: pd.DataFrame, cap: float = 0.10) -> np.ndarray:
    """Risk parity via equal risk-contribution shares.

    Minimises the sum of squared deviations of each asset's risk-contribution
    *share* from 1/n. Shares are O(1), so the objective is well scaled (a
    level-based formulation is numerically flat at equal weight and SLSQP
    stalls on it - the exact pitfall the brief warns about).
    """
    n = returns.shape[1]
    cov = shrinkage_covariance(returns)
    inv_var = 1.0 / np.maximum(np.diag(cov), 1e-9)
    w0 = inv_var / inv_var.sum()

    def obj(w):
        s = cov @ w
        v = w @ s
        u = w * s / v - 1.0 / n
        return 0.5 * float(u @ u)

    def jac(w):
        s = cov @ w
        v = w @ s
        u = w * s / v - 1.0 / n
        return (u * s + cov @ (u * w) - 2 * s * ((u * w) @ s)) / v

    res = minimize(
        obj, w0, jac=jac,
        bounds=_bounds(n, cap), constraints=[_sum_constraint(n)],
        method="SLSQP", options={"maxiter": 800, "ftol": 1e-12},
    )
    return res.x


def min_cvar_weights(returns: pd.DataFrame, alpha: float = 0.95, cap: float = 0.10) -> np.ndarray:
    """Minimum-CVaR weights via a linear program (Rockafellar-Uryasev)."""
    n = returns.shape[1]
    window = returns.tail(CVAR_LOOKBACK).to_numpy()
    if window.shape[0] < 60:
        raise ValueError("need at least 60 rows of history for CVaR")
    n_obs = window.shape[0]

    c = np.concatenate([
        np.zeros(n),
        [1.0],
        np.full(n_obs, 1.0 / ((1 - alpha) * n_obs)),
    ])
    A_ub = np.hstack([-window, -np.ones((n_obs, 1)), -np.eye(n_obs)])
    b_ub = np.zeros(n_obs)
    A_eq = np.concatenate([np.ones(n), np.zeros(1 + n_obs)])[None, :]
    b_eq = [1.0]
    bounds = _bounds(n, cap) + [(None, None)] + [(0.0, None)] * n_obs

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"min-CVaR LP failed: {res.message}")
    return res.x[:n]


METHOD_FUNCTIONS = {
    "equal_weight": equal_weight_weights,
    "min_variance": min_variance_weights,
    "max_sharpe": max_sharpe_weights,
    "risk_parity": risk_parity_weights,
    "min_cvar": min_cvar_weights,
}


def compute_target_weights(history: pd.DataFrame, method: str, cap: float = 0.10) -> np.ndarray:
    """Target weights for one method from an expanding history frame."""
    if method == "equal_weight":
        return equal_weight_weights(history.shape[1])
    return METHOD_FUNCTIONS[method](history, cap=cap)


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

def monthly_decision_dates(returns: pd.DataFrame,
                           min_training: int = MIN_TRAINING) -> list[pd.Timestamp]:
    """Month-end trading days with at least `min_training` prior rows."""
    dates = returns.index
    keep = []
    for i in range(1, len(dates)):
        d = dates[i]
        if i < min_training:
            continue
        if d.month != dates[i - 1].month:
            keep.append(d)
    return keep


@dataclass
class Backtest:
    """Result of a walk-forward backtest for one fund."""

    fund: str
    method: str
    returns: pd.Series
    weights: pd.DataFrame
    decision_dates: list[pd.Timestamp]
    turnover: pd.Series
    cov_ridge: float


def walk_forward_backtest(
    returns: pd.DataFrame,
    method: str,
    fund: str,
    *,
    min_training: int = MIN_TRAINING,
    cap: float = 0.10,
    transform=None,
) -> Backtest:
    """Out-of-sample backtest for one method on one fund panel.

    Decision dates are month-ends with at least min_training prior days.
    Target weights are solved from strictly prior data; positions run from the
    decision date through the day before the next decision. If `transform` is
    given it is called as transform(w, history) after the base weights are
    solved and may overwrite the weights (used for sentiment tilts).
    """
    dates = monthly_decision_dates(returns, min_training)
    if not dates:
        raise ValueError("no decision dates: history too short for min_training")

    port_returns: list[pd.Series] = []
    weight_records = []
    turnover_vals: list[float] = []
    prev_w = None

    for i, d in enumerate(dates):
        history = returns.loc[returns.index < d]
        usable = history.tail(COV_LOOKBACK).notna().sum() >= MIN_VALID_OBS
        usable_cols = list(usable.index[usable])
        if not usable_cols:
            raise ValueError(f"no usable assets at decision date {d}")
        w = compute_target_weights(history[usable_cols], method, cap=cap)
        w = np.minimum(w, cap)
        if w.sum() > 0:
            w = w / w.sum()
        w = pd.Series(w, index=usable_cols).reindex(returns.columns, fill_value=0.0)
        if transform is not None:
            w = transform(w, history)
            w = w.astype(float)

        end = dates[i + 1] if i + 1 < len(dates) else returns.index[-1]
        interval = returns.loc[d:end]
        port_returns.append(interval @ w)
        weight_records.append(pd.DataFrame({d: w}).T)

        if prev_w is not None:
            turnover_vals.append(float((w - prev_w).abs().sum()))
        prev_w = w

    port = pd.concat(port_returns)
    port = port[~port.index.duplicated(keep="first")].sort_index()
    weights = pd.concat(weight_records).sort_index()
    return Backtest(
        fund=fund, method=method, returns=port, weights=weights,
        decision_dates=dates, turnover=pd.Series(turnover_vals, index=dates[1:]),
        cov_ridge=RIDGE_SHRINKAGE,
    )


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def performance_metrics(returns: pd.Series, trading_days: int = TRADING_DAYS) -> dict:
    """Annualized performance statistics for a daily returns series."""
    r = returns.dropna()
    if r.empty:
        raise ValueError("no returns to evaluate")
    total = float((1 + r).prod() - 1)
    years = len(r) / trading_days
    cagr = float((1 + total) ** (1 / years) - 1) if years > 0 else 0.0
    ann_vol = float(r.std(ddof=1) * np.sqrt(trading_days))
    sharpe = float(cagr / ann_vol) if ann_vol > 0 else 0.0
    cum = (1 + r).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1.0
    mdd = float(drawdown.min())
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0
    positive = float((r > 0).mean())
    return {
        "total_return": total,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": calmar,
        "positive_days": positive,
        "n_days": len(r),
        "start": r.index[0],
        "end": r.index[-1],
    }


def summary_table(backtests: list[Backtest]) -> pd.DataFrame:
    """Metrics table across a list of backtests (one row per fund-method)."""
    rows = []
    for bt in backtests:
        m = performance_metrics(bt.returns)
        rows.append({
            "fund": bt.fund,
            "method": bt.method,
            "cagr": m["cagr"],
            "ann_vol": m["ann_vol"],
            "sharpe": m["sharpe"],
            "max_drawdown": m["max_drawdown"],
            "calmar": m["calmar"],
            "turnover_avg": float(np.mean(bt.turnover)) if len(bt.turnover) else 0.0,
        })
    return pd.DataFrame(rows)
