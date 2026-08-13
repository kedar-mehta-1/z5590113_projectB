# Prompt Log - Funds: Walk-Forward Backtest and Optimisation Methods

## Prompt 1: Look-Ahead-Free Backtest Loop

**My prompt:**
"Write a walk_forward_backtest() in src/portfolios.py. Monthly decision dates.
For each date solve target weights from the estimation window, hold them for the
following month, and return a Backtest with returns, weights, and decision
dates. Document the first live date."

**AI output (key parts):**
```python
# AI estimated from an expanding window that included the decision day:
cov = returns.loc[:d].cov()
mu = returns.loc[:d].mean()

# and computed CAGR as:
cagr = returns.mean().mean() * 252
```

**What was wrong:**
1. Look-ahead: the estimation window included day d's own return, and one
   version used the whole panel. The weights were formed with information that
   would not have existed at the decision date.
2. The first live date was just the first month-end in the data, with no minimum
   estimation window, so early weights were solved from roughly 20 rows.
3. CAGR was computed as mean daily return times 252, which ignores compounding
   and overstates long-run growth.

**My correction:**
```python
# Estimation window is strictly prior data only:
cov = history.loc[history.index < d].cov()
mu = history.loc[history.index < d].mean()

# A date is eligible only with >= MIN_TRAINING (504) prior rows:
if n_prior < MIN_TRAINING:
    continue

# True CAGR from total return over actual elapsed years:
cagr = (1 + total_return) ** (1 / years) - 1
```

**Why the correction matters:** This is the core integrity guarantee of the
project. The first live backtest date becomes 2 January 2022 (504 prior rows
from the 2020 data), which is stated in the report and the app. Portfolio return
is the dot product of yesterday's weights with today's returns, so no leakage is
possible. A true CAGR keeps the report honest about growth of $1.

---

## Prompt 2: Risk-Parity / Min-Variance Solver Stall

**My prompt:**
"Implement risk_parity and min_cvar weight solvers with a 10% per-name cap.
Run the combined-family backtests and check the weights differ across methods."

**AI output (key parts):**
```python
# AI's risk parity targeted a volatility level per name:
#   minimize  sum((w_i * sigma_i - target) ** 2)
# and both solvers fed the raw daily-return covariance.
```

**What was wrong:**
1. The risk-parity weights came out numerically identical to equal weight: on a
   tiny daily covariance the level-based objective is nearly flat and SLSQP
   stalled at the starting point - the exact "solver silently stalls" warning
   in the brief. The funds were not actually different.
2. A flat 10% cap on the 10-name crypto panel forced every crypto method to
   equal weight, collapsing the crypto funds into one.
3. Raw covariance on correlated daily returns is near-singular and produced
   weights that jumped between rebalances.

**My correction:**
```python
# Risk parity as a risk-CONTRIBUTION-share objective with a 1/variance start:
def risk_parity_weights(cov):
    inv_var = 1 / np.diag(cov)
    w0 = inv_var / inv_var.sum()
    obj = lambda w: np.sum((w * (cov @ w) / (w @ cov @ w) - 1 / n) ** 2)
    ...

# Ridge shrinkage on the sample covariance before every solve:
cov = (1 - SHRINK) * sample_cov + SHRINK * np.diag(np.diag(sample_cov))
```

**Why the correction matters:** The brief explicitly warns that optimisers can
silently return the starting weights. The risk-contribution-share objective is
steep near the solution so SLSQP moves off the start, and the 1/variance start
is a sensible prior. Per-family caps (10% equity, 20% crypto so at least five of
ten names are held, 10% combined) keep the methods genuinely different. I added
a build-time assertion that the method weight vectors differ, so a silent stall
fails loudly next time instead of producing duplicate funds.
