# Prompt Log - Fusion: Folding FinSent Into the Equity Funds

## Prompt 1: Sentiment Tilt

**My prompt:**
"Add src/fusion.py: a sentiment tilt that reweights the equity max-Sharpe fund
by cross-sectional z-scores of the 5-day lagged sentiment, and a sentiment-
momentum long-short fund. Keep everything long-only where the base fund is
long-only, and lag the sentiment by at least one trading day. Produce a
before-vs-after table and figure."

**AI output (key parts):**
```python
# AI's first tilt re-used the signal inside the same bar it was deciding for:
z = zscore(sentiment.loc[d])
weights = base_weights * (1 + z)
```

**What was wrong:** Same-signal, same-period contamination. The tilt must use
sentiment as of day t-1 (or earlier) and then hold for the next month; using the
day-d sentiment while also deciding for day d lets the signal peek at the return
it is meant to predict. The momentum fund had a worse version: its long and
short legs were chosen from a ranking built over the full window including the
holding month (in-sample selection).

**My correction:**
```python
# Tilt: base weights reweighted by the LAGGED, winsorized z-score.
z = winsorize(zscore(mean5(sentiment.shift(LAG))), -2, +2)
w = base_weights * (1 + INTENSITY * z)
w = w / w.sum()          # renormalise, keep long-only

# Momentum: long top-10 at +6%, short bottom-10 at -6%, dollar-neutral,
# 40% cash, from the same lagged ranking. No forward information enters.
```

**Why the correction matters:** The whole point of the fusion is that it is a
test of whether news sentiment has tradeable persistence. If the construction
leaks the future, the test is meaningless. Both funds now obey the identical
decision-date rules as the rest of the shelf, so their results are directly
comparable with the base funds.

---

## Prompt 2: Before-vs-After Comparison and the Negative Result

**My prompt:**
"Produce the before-vs-after table and growth figures comparing each fusion fund
with its base, and capture the finding in the table and figures."

**AI output (key parts):**
```python
# AI first compared the tilt against Equal Weight and reported the tilt as
# "improved", even though the tilt's base is the Maximum-Sharpe fund.
```

**What was wrong:** Comparing the tilt to the wrong base made the negative
result look positive. The tilt sits on top of Maximum-Sharpe, and momentum sits
on top of Equal Weight; each must be measured against its own base.

**My correction:**
- Tilt vs Maximum-Sharpe base: Sharpe -0.226 to -0.154, CAGR -4.14% to -2.85%,
  max drawdown -25.74% to -24.40%. Still negative, but damage reduced.
- Momentum vs Equal-Weight base: -0.72% CAGR with Sharpe -0.094 while its base
  earned +5.97% with Sharpe 0.338. The only improvement is drawdown (-13.1% vs
  -20.3%), which follows from holding 40% cash and a dollar-neutral book.
- Both figures now compare each fusion fund to its true base.

**Why the correction matters:** An honest negative result - a daily signal
rebalanced monthly cannot beat buy-and-hold equal weight in this sample - is a
finding backed by the before-vs-after table and figures (noisy headline
sentiment, monthly rebalance too slow for a daily signal, weak lagged
persistence confirmed by the lead-lag diagnostics) rather than tuning the
fusion until it "wins". The marks are for evidenced original work, not a
winning number.
