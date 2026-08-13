"""Unit tests for the AlphaStream Part B modules.

These tests run offline on synthetic data (no hosted data, no network). The
smoke test in test_smoke.py covers the real data path.

    python -m pytest tests/ -q
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from src import fusion
from src import portfolios as P
from src import sentiment as S

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _panel(n_assets: int = 50, t: int = 700, seed: int = 7, late_cols: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=t, freq="B")
    rets = pd.DataFrame(rng.normal(0.0004, 0.012, size=(t, n_assets)),
                        index=dates, columns=[f"A{i}" for i in range(n_assets)])
    for c in range(late_cols):
        rets.iloc[:150, c] = np.nan
    return rets


# ---------------------------------------------------------------------------
# Solver sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", P.METHODS)
def test_weights_valid(method):
    rets = _panel()
    w = P.compute_target_weights(rets, method)
    assert len(w) == rets.shape[1]
    assert np.isclose(w.sum(), 1.0, atol=1e-4)
    assert w.min() >= -1e-6


def test_methods_differ_on_realistic_data():
    rng = np.random.default_rng(9)
    dates = pd.date_range("2021-01-01", periods=700, freq="B")
    vols = np.linspace(0.004, 0.030, 50)
    rets = pd.DataFrame([rng.normal(0.0003, v, size=700) for v in vols]).T
    rets.index = dates
    rets.columns = [f"A{i}" for i in range(50)]
    ws = {m: P.compute_target_weights(rets, m) for m in P.METHODS}
    for m in P.METHODS:
        if m == "equal_weight":
            continue
        assert np.abs(ws[m] - ws["equal_weight"]).max() > 1e-6, m


def test_risk_parity_not_stuck_at_equal_weight():
    # Varying per-asset volatility should move risk parity off equal weight.
    rng = np.random.default_rng(11)
    dates = pd.date_range("2021-01-01", periods=520, freq="B")
    vols = np.linspace(0.005, 0.04, 20)
    rets = pd.DataFrame([rng.normal(0.0, v, size=520) for v in vols]).T
    rets.index = dates
    w = P.compute_target_weights(rets, "risk_parity")
    assert np.abs(w - 1 / 20).max() > 0.005


def test_cap_respected():
    rets = _panel(n_assets=10)
    w = P.compute_target_weights(rets, "min_variance", cap=0.20)
    assert w.max() <= 0.20 + 1e-6


def test_no_lookahead_backtest_uses_prior_history():
    rets = _panel()
    bt = P.walk_forward_backtest(rets, "max_sharpe", "combined_max_sharpe")
    first_decision = bt.decision_dates[0]
    i = rets.index.get_loc(first_decision)
    assert i >= P.MIN_TRAINING - 1
    # returns cover exactly the interval from first decision to the last date
    assert bt.returns.index[0] == first_decision
    assert bt.returns.index[-1] == rets.index[-1]


def test_late_starting_assets_handled():
    rets = _panel(late_cols=2)
    bt = P.walk_forward_backtest(rets, "min_variance", "combined_min_variance")
    assert len(bt.returns) > 0
    assert np.isfinite(bt.returns).all()


def test_performance_metrics_annualised():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0005, 0.01, 252))
    m = P.performance_metrics(r)
    sample_std = r.std(ddof=1)
    total = float((1 + r).prod() - 1)
    years = len(r) / 252
    cagr = float((1 + total) ** (1 / years) - 1)
    assert m["ann_vol"] == pytest.approx(sample_std * np.sqrt(252), rel=1e-6)
    assert m["cagr"] == pytest.approx(cagr, rel=1e-6)
    assert m["sharpe"] == pytest.approx(cagr / (sample_std * np.sqrt(252)), rel=1e-6)
    assert m["max_drawdown"] <= 0.0


# ---------------------------------------------------------------------------
# Sentiment lexicon
# ---------------------------------------------------------------------------

def test_finsent_moves_finance_terms():
    fin = S.build_finsent()
    vader = S.build_vader()
    assert S.score_text(vader, "Apple beats earnings estimates") == 0.0
    assert S.score_text(fin, "Apple beats earnings estimates") > 0.2
    assert S.score_text(fin, "Company misses expectations") < -0.2
    assert S.score_text(fin, "Bank upgrades the stock") > 0.1


def test_finsent_reduces_false_neutrals():
    titles = pd.Series([
        "Apple beats earnings estimates",
        "Fed signals hawkish stance on inflation",
        "Company announces stock buyback",
        "Bank downgrades rival",
        "Firm cuts guidance after weak quarter",
    ])
    sc = S.score_unique_titles(titles, analyzers=("vader", "finsent"))
    neutral_vader = (sc["vader_compound"] == 0.0).mean()
    neutral_fin = (sc["finsent_compound"] == 0.0).mean()
    assert neutral_fin < neutral_vader


def test_scoring_deterministic():
    titles = pd.Series(["positive earnings beat here", "negative miss there"])
    a = S.score_unique_titles(titles, analyzers=("vader", "finsent"))
    b = S.score_unique_titles(titles, analyzers=("vader", "finsent"))
    pd.testing.assert_frame_equal(a, b)


def test_sector_index_no_news_neutral():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2022-01-03", periods=60, freq="B")
    sentiment = pd.DataFrame({
        "date": dates, "ticker": ["T1"] * 60, "sector": ["Tech"] * 60,
        "title": ["x"] * 60, "finsent_compound": rng.uniform(-0.5, 0.5, 60),
    })
    sector_map = pd.DataFrame({"ticker": ["T1", "T2"], "sector": ["Tech", "Tech"]})
    ticker_daily = S.ticker_daily_sentiment(sentiment, dates, score_col="finsent_compound", lag=1)
    index_long, coverage = S.sector_sentiment_index(ticker_daily, sector_map)
    idx = index_long.set_index("date")
    # first row must be neutral (lag) and coverage 0
    assert idx["sentiment"].iloc[0] == 0.0
    assert coverage["n_covered"].iloc[0] == 0
    # index is lagged: sentiment at day t reflects day t-1 scores
    sent_t2 = idx.loc[dates[2], "sentiment"]
    assert sent_t2 != idx.loc[dates[1], "sentiment"]


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def test_momentum_transform_dollar_neutral():
    rng = np.random.default_rng(5)
    dates = pd.date_range("2022-01-03", periods=700, freq="B")
    n = 50
    signal = pd.DataFrame(
        rng.normal(0, 0.1, size=(len(dates), n)),
        index=dates, columns=[f"E{i}" for i in range(n)],
    )
    transform = fusion.momentum_transform(signal)
    w = pd.Series(1.0 / n, index=signal.columns)
    history = pd.DataFrame(index=dates[:600])
    tw = transform(w, history)
    assert np.isclose(tw.sum(), 0.0, atol=1e-9)          # long 0.6 - short 0.6
    assert np.isclose(tw[tw > 0].sum(), 0.6, atol=1e-9)  # long cap 60%
    assert np.isclose(tw[tw < 0].sum(), -0.6, atol=1e-9)  # short cap 60%


def test_tilt_transform_lookahead_safe():
    # Signal available on a decision date must be built only from rows with
    # date <= history end (the day before the decision).
    rng = np.random.default_rng(5)
    dates = pd.date_range("2022-01-03", periods=100, freq="B")
    signal = pd.DataFrame(rng.normal(0, 0.1, size=(len(dates), 50)),
                          index=dates, columns=[f"E{i}" for i in range(50)])
    transform = fusion.sentiment_tilt_transform(signal)
    w = pd.Series(1.0 / 50, index=signal.columns)
    history = pd.DataFrame(index=dates[:60])
    tw = transform(w, history)
    assert np.isclose(tw.sum(), 1.0, atol=1e-6)
    assert tw.min() >= 0.0
