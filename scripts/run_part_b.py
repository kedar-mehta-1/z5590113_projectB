"""Reproduce all Part B results end-to-end. Run from the project root:

    .\\.venv\\Scripts\\python.exe scripts/run_part_b.py

Pipeline (Data Factory Floor):

    Station 1  ETL        load + clean + integrity-check the three datasets
    Station 2  Features   return panels, crypto-to-equity alignment, headline
                          panel, spillover index, lead-lag diagnostics
    Station 3  Sentiment  FinSent headline scoring -> sector sentiment index
    Station 3  Funds      walk-forward OOS backtests across equity / crypto /
                          combined x 5 methods
    Station 4  Fusion     sentiment tilt fund + sentiment momentum fund
    Fact sheets  snapshot, monthly returns, holdings for every fund
    Outputs      results/data (app-readable CSVs), results/tables, results/figures

The script is deterministic: given the same hosted data it produces the same
artifacts. Sentiment scoring is cached under results/data so rebuilds are fast.
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src import etl, factsheet, features, figures, fusion, portfolios, sentiment

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA_OUT = RESULTS / "data"
TABLES_OUT = RESULTS / "tables"
FIGURES_OUT = RESULTS / "figures"


def step(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def save_table(df: pd.DataFrame, name: str) -> None:
    path = TABLES_OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  wrote {path.relative_to(ROOT)} ({df.shape[0]} rows)")


def save_data(df: pd.DataFrame, name: str, index: bool = False) -> None:
    path = DATA_OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    print(f"  wrote {path.relative_to(ROOT)} ({df.shape[0]} rows)")


def main() -> None:
    t0 = time.time()

    # ------------------------------------------------------------------ S1
    step("Station 1 - ETL: load, clean, integrity check")
    eq, cr, nh, integrity = etl.load_all()
    print(integrity.summary_table().to_string(index=False))
    save_table(integrity.summary_table(), "integrity_report.csv")

    # ------------------------------------------------------------------ S2
    step("Station 2 - Features: returns, calendars, headlines")
    eq_ret = features.daily_returns(eq)
    cr_ret = features.daily_returns(cr)
    combined = features.merge_crypto_to_equity(eq_ret, cr_ret)
    eq_cols, cr_cols = features.split_combined(combined)
    trading_dates = combined.index

    sector_map = nh[["ticker", "sector"]].drop_duplicates().dropna()
    sectors = sorted(sector_map["sector"].unique())
    print(f"  equity trading days: {len(trading_dates)}  assets: {len(eq_cols)} eq / {len(cr_cols)} cr")  # noqa: E501
    print(f"  sectors: {', '.join(sectors)}")

    spillover = features.news_flow_spillover_index(nh, trading_dates)
    save_data(spillover, "spillover_index.csv")
    lead_lag, lead_lag_summary = features.cross_asset_lead_lag(combined)
    save_data(lead_lag, "lead_lag_heat.csv")
    save_data(lead_lag_summary, "lead_lag_summary.csv")

    # ------------------------------------------------------------------ S3
    step("Station 3 - Sentiment: FinSent scoring and sector index")
    sent_cache = DATA_OUT / "sentiment_title_scores.parquet"
    sent = sentiment.build_sentiment_index(
        nh, trading_dates, sector_map,
        cache_path=sent_cache,
    )
    index_long = sent["index_long"]
    market_long = sent["market_index"].copy()
    market_long["sector"] = "market"
    index_long_all = pd.concat([index_long, market_long], ignore_index=True)
    save_data(index_long_all, "sector_sentiment_index.csv")

    plain = sentiment.score_unique_titles(nh["title"], cache_path=sent_cache, analyzers=("vader",))
    neutral_counts = {
        "plain": plain["vader_compound"],
        "finsent": sent["scored"]
        .merge(plain[["title", "vader_compound"]], on="title", how="left")["finsent_compound"],
    }
    _ = neutral_counts

    # ------------------------------------------------------------------ S3
    step("Station 3 - Funds: walk-forward OOS backtests")
    # Per-name caps: equity 10% (50 names), crypto 20% (10 names -> at least 5
    # names held), combined 10% (60 names). A uniform 10% cap would force every
    # crypto method to equal weight (10 x 10% = 100%).
    family_config = {
        "equity": {"panel": combined[eq_cols], "cap": 0.10},
        "crypto": {"panel": combined[cr_cols], "cap": 0.20},
        "combined": {"panel": combined, "cap": 0.10},
    }
    backtests: dict[str, portfolios.Backtest] = {}
    for family, cfg in family_config.items():
        for method in portfolios.METHODS:
            fund = f"{family}_{method}"
            backtests[fund] = portfolios.walk_forward_backtest(
                cfg["panel"], method, fund, cap=cfg["cap"],
            )

    # ------------------------------------------------------------------ S4
    step("Station 4 - Fusion: sentiment tilt + momentum funds")
    signal_wide = sent["ticker_daily_lag"].reindex(columns=eq_cols)
    fusion_funds = fusion.build_fusion_funds(combined[eq_cols], signal_wide)
    backtests.update(fusion_funds)

    # ------------------------------------------------------------------ metrics
    step("Performance metrics and required app artifacts")
    SENTIMENT_FUNDS = {"equity_sentiment_tilt", "equity_sentiment_momentum"}

    def family_of(fund: str) -> str:
        return "sentiment" if fund in SENTIMENT_FUNDS else fund.split("_")[0]

    metric_rows = []
    for fund, bt in backtests.items():
        m = portfolios.performance_metrics(bt.returns)
        metric_rows.append({
            "fund": fund,
            "family": family_of(fund),
            "method": bt.method,
            "cagr": round(m["cagr"], 4),
            "ann_vol": round(m["ann_vol"], 4),
            "sharpe": round(m["sharpe"], 4),
            "max_drawdown": round(m["max_drawdown"], 4),
            "calmar": round(m["calmar"], 4),
            "total_return": round(m["total_return"], 4),
            "turnover_avg": round(float(np.mean(bt.turnover)) if len(bt.turnover) else 0.0, 4),
        })
    metrics_df = pd.DataFrame(metric_rows)
    save_table(metrics_df, "performance_metrics.csv")

    fund_returns = pd.DataFrame({fund: bt.returns for fund, bt in backtests.items()})
    fund_returns.index.name = "date"
    save_data(fund_returns.reset_index(), "fund_returns.csv")

    weight_rows = []
    for fund, bt in backtests.items():
        w = bt.weights.reset_index().melt(id_vars="index", var_name="ticker", value_name="weight")
        w = w.rename(columns={"index": "date"})
        w.insert(0, "fund", fund)
        weight_rows.append(w)
    fund_weights = pd.concat(weight_rows, ignore_index=True)
    save_data(fund_weights, "fund_weights.csv")

    last_dates = fund_weights.groupby("fund")["date"].transform("max")
    latest_weights = (
        fund_weights[fund_weights["date"] == last_dates]
        .loc[lambda d: d["weight"].abs() > 1e-8]
        .sort_values(["fund", "ticker"])
        .reset_index(drop=True)
    )
    save_data(latest_weights, "fund_latest_weights.csv")

    # ------------------------------------------------------------------ factsheets
    step("Fact sheets (snapshot, monthly returns, holdings)")
    for fund, bt in backtests.items():
        factsheet.snapshot(bt).to_csv(TABLES_OUT / f"snapshot_{fund}.csv", index=False)
        factsheet.monthly_returns(bt.returns).to_csv(TABLES_OUT / f"monthly_{fund}.csv")
        latest = bt.weights.iloc[-1]
        factsheet.top_holdings(latest, 10).to_csv(TABLES_OUT / f"holdings_{fund}.csv", index=False)

    # ------------------------------------------------------------------ figures
    step("Figures (all required exhibits + bonus)")
    sample_start = fund_returns.index.min().date()
    sample_end = fund_returns.index.max().date()
    sample = f"{sample_start} to {sample_end}"

    captions = []

    for family in ("equity", "crypto", "combined"):
        bts = [backtests[f"{family}_{m}"] for m in portfolios.METHODS]
        cap = figures.cumulative_growth(
            bts, FIGURES_OUT / f"growth_{family}.png",
            title=f"AlphaStream {family} funds - growth of $1",
            sample=sample, families=family,
        )
        captions.append(cap)

    figures.drawdown(
        backtests["combined_max_sharpe"], FIGURES_OUT / "drawdown_combined_max_sharpe.png",
        title="Combined Maximum Sharpe - drawdown",
        sample=sample,
    )
    figures.drawdown(
        backtests["combined_min_variance"], FIGURES_OUT / "drawdown_combined_min_variance.png",
        title="Combined Minimum Variance - drawdown",
        sample=sample,
    )

    figures.weights_over_time(
        [backtests["combined_max_sharpe"], backtests["combined_min_variance"],
         backtests["combined_risk_parity"]],
        FIGURES_OUT / "weights_combined.png",
        title="Combined funds - target weights over time",
        sample=sample, sector_map=sector_map,
    )

    figures.sharpe_bar(
        metrics_df, FIGURES_OUT / "sharpe_bar.png",
        title="AlphaStream funds - out-of-sample Sharpe ratio",
        sample=sample,
    )
    figures.risk_return_scatter(
        metrics_df, FIGURES_OUT / "risk_return.png",
        title="AlphaStream funds - return vs risk",
        sample=sample,
    )

    sent_sample = (
        f"{sent['index_wide'].index.min().date()} to "
        f"{sent['index_wide'].index.max().date()}"
    )
    cap = figures.sentiment_index(
        sent["index_wide"], sent["market_index"], sent["coverage"],
        FIGURES_OUT / "sentiment_index.png",
        title="AlphaStream news-sentiment index by sector",
        sample=sent_sample,
    )
    captions.append(cap)
    figures.sentiment_neutrality(
        plain["vader_compound"], sent["scored"]["finsent_compound"],
        FIGURES_OUT / "sentiment_neutrality.png",
        title="False-neutral headlines: plain VADER vs FinSent",
        sample=sample,
    )

    figures.fusion_compare(
        backtests["equity_max_sharpe"].returns,
        backtests["equity_sentiment_tilt"].returns,
        FIGURES_OUT / "fusion_tilt.png",
        title="Fusion - equity max Sharpe vs sentiment tilt",
        sample=sample,
    )
    figures.fusion_compare(
        backtests["equity_equal_weight"].returns,
        backtests["equity_sentiment_momentum"].returns,
        FIGURES_OUT / "fusion_momentum.png",
        title="Fusion - equity equal weight vs sentiment momentum",
        sample=sample,
        base_label="Base (equal weight)", tilt_label="Sentiment momentum",
    )

    cap_path = TABLES_OUT / "figure_captions.md"
    cap_path.write_text("\n\n".join(captions), encoding="utf-8")
    print(f"  wrote {cap_path.relative_to(ROOT)}")

    # ------------------------------------------------------------------ summary
    print("\n=== SUMMARY ===", flush=True)
    print(f"funds: {len(backtests)}")
    top = metrics_df.nlargest(3, "sharpe")[["fund", "sharpe", "cagr", "max_drawdown"]]
    print(top.to_string(index=False))
    print(f"\nfirst live backtest date: {fund_returns.index.min().date()}")
    print(f"total run time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
