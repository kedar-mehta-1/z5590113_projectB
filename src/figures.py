"""Station 3/4 - report and app figures in the AlphaStream design system.

Every figure is built from precomputed artifacts (Backtest objects, sentiment
panels, metrics tables) and saved as a self-contained PNG plus a caption string
for the Word report. No raw data is touched here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from . import design as d


def _fig(title: str = "", profile: str = "word_a4",
         figsize=(7.2, 4.0), constrained: bool = True) -> tuple[plt.Figure, plt.Axes]:
    d.apply_as_theme(profile)
    if constrained:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        with matplotlib.rc_context({"figure.constrained_layout.use": False}):
            fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title)
    return fig, ax


def _bboxes_overlap(a: list[float], b: list[float]) -> bool:
    """True if two [x0, y0, x1, y1] boxes intersect."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _circle_box_overlap(cx: float, cy: float, r: float, box: list[float]) -> bool:
    """True if a circle at (cx, cy) with radius r touches box [x0, y0, x1, y1]."""
    nx = min(max(cx, box[0]), box[2])
    ny = min(max(cy, box[1]), box[3])
    return (cx - nx) ** 2 + (cy - ny) ** 2 < r ** 2


def cumulative_growth(
    backtests: list,
    out_path: str | Path,
    *,
    title: str,
    sample: str,
    families: str = "combined",
) -> str:
    """Growth-of-$1 (net of fees) comparing methods for a fund family."""
    fig, ax = _fig(figsize=(8.2, 4.4))
    for bt in backtests:
        label = d.METHOD_LABELS.get(bt.method, bt.method)
        growth = (1 + bt.returns).cumprod()
        ax.plot(growth.index, growth.values, label=label, color=d.method_color(bt.method))
    ax.axhline(1.0, color=d.AS_COLORS["slate"], lw=0.8, ls="--", alpha=0.7)
    ax.set_ylabel("Growth of $1 (net of fees)")
    ax.set_xlabel("Trading date")
    ax.legend(ncol=2, loc="best")
    fig_path = d.save_figure(fig, out_path)
    cap = d.caption(
        title,
        sample=sample,
        units="Indexed to $1.00 at the first live backtest date",
        note=f"{families} fund family, monthly rebalance, long-only, 10% per-name cap.",
    )
    return f"{cap} Path: {fig_path}"


def drawdown(
    backtest,
    out_path: str | Path,
    *,
    title: str,
    sample: str,
) -> str:
    """Drawdown series for one fund (required exhibit)."""
    cum = (1 + backtest.returns).cumprod()
    dd = cum / cum.cummax() - 1.0
    fig, ax = _fig(figsize=(8.2, 3.6))
    ax.fill_between(dd.index, dd.values, 0, color=d.method_color(backtest.method), alpha=0.35, lw=0)
    ax.plot(dd.index, dd.values, color=d.method_color(backtest.method), lw=1.1)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Trading date")
    mdd = dd.min()
    ax.annotate(f"max drawdown {mdd:.1%}",
                xy=(dd.idxmin(), mdd), xytext=(dd.idxmin(), mdd * 0.45),
                fontsize=9, color=d.AS_COLORS["ink"],
                arrowprops=dict(arrowstyle="->", color=d.AS_COLORS["slate"], lw=0.8))
    fig_path = d.save_figure(fig, out_path)
    cap = d.caption(
        title,
        sample=sample,
        units="Percent of peak value (negative)",
        note=f"Daily drawdown of {backtest.fund} from gross returns, no fees.",
    )
    return f"{cap} Path: {fig_path}"


def weights_over_time(
    backtests: list,
    out_path: str | Path,
    *,
    title: str,
    sample: str,
    sector_map: pd.DataFrame | None = None,
) -> str:
    """Portfolio weights over time (stacked area) comparing methods.

    When sector_map is supplied, weights are aggregated to sector level so the
    chart is readable for a 50-60 name fund.
    """
    n = len(backtests)
    fig, axes = plt.subplots(1, n, figsize=(9.2, 4.2), sharex=True)
    d.apply_as_theme("word_a4")
    if n == 1:
        axes = [axes]
    for ax, bt in zip(axes, backtests):
        weights = bt.weights.T
        if sector_map is not None:
            mapping = sector_map.set_index("ticker")["sector"]
            weights.index = [mapping.get(t, "Other") for t in weights.index]
            weights = weights.groupby(level=0).sum()
            weights = weights.reindex(weights.sum(axis=1).sort_values(ascending=False).index)
        colors = d.AS_SECTOR_COLORS[: len(weights)]
        ax.stackplot(weights.columns, weights.to_numpy(), labels=weights.index,
                     colors=colors, alpha=0.85)
        ax.set_title(d.METHOD_LABELS.get(bt.method, bt.method), fontsize=10)
        ax.set_ylabel("Weight")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.tick_params(axis="x", labelsize=7)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(40)
            lbl.set_ha("right")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
    fig.suptitle(title, color=d.AS_COLORS["navy"], fontweight="bold", fontsize=13)
    fig_path = d.save_figure(fig, out_path)
    cap = d.caption(
        title,
        sample=sample,
        units="Target weight (fraction of NAV)",
        note="Weights are the month-end target weights from each live rebalance; "
             "interim drift is not shown. Sectors aggregated when sector_map is given.",
    )
    return f"{cap} Path: {fig_path}"


def sharpe_bar(
    metrics: pd.DataFrame,
    out_path: str | Path,
    *,
    title: str,
    sample: str,
) -> str:
    """Sharpe ratio bar chart across funds and methods."""
    plot_df = metrics.copy()
    plot_df["label"] = plot_df.apply(
        lambda r: (f"{d.FAMILY_LABELS.get(r['family'], r['family'])} / "
                   f"{d.METHOD_LABELS.get(r['method'], r['method'])}"),
        axis=1)
    plot_df = plot_df.sort_values("sharpe")
    colors = [d.method_color(m) for m in plot_df["method"]]
    fig, ax = _fig(figsize=(9.2, 4.6))
    bars = ax.barh(plot_df["label"], plot_df["sharpe"], color=colors, alpha=0.9, height=0.7)
    ax.axvline(0.0, color=d.AS_COLORS["ink"], lw=0.9)
    ax.set_xlabel("Out-of-sample Sharpe ratio (rf = 0)")
    for bar, val in zip(bars, plot_df["sharpe"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8.5, color=d.AS_COLORS["slate"])
    fig_path = d.save_figure(fig, out_path)
    cap = d.caption(
        title,
        sample=sample,
        units="Annualised Sharpe ratio, risk-free rate assumed 0",
        note="Bars coloured by optimisation method; label format is family / method.",
    )
    return f"{cap} Path: {fig_path}"


def risk_return_scatter(
    metrics: pd.DataFrame,
    out_path: str | Path,
    *,
    title: str,
    sample: str,
) -> str:
    """Return-vs-risk scatter with iso-Sharpe lines (bonus exhibit).

    The axes are pinned (no constrained layout, explicit position) and the
    figure is saved on a full canvas (no tight re-crop) so the saved pixels
    match the geometry the numbered labels were placed against.
    """
    fig, ax = _fig(figsize=(8.6, 7.0), constrained=False)
    fig.set_layout_engine(None)
    ax.set_position([0.10, 0.24, 0.87, 0.70])
    vol = np.linspace(0.05, 0.55, 100)
    for sharpe in (0.4, 0.8, 1.2, 1.6):
        ax.plot(vol, sharpe * vol, ls="--", lw=0.7, color=d.AS_COLORS["mist"])
        ax.text(vol[-1] * 0.99, sharpe * vol[-1], f"Sharpe {sharpe}", fontsize=7.5,
                color=d.AS_COLORS["slate"], ha="right", va="bottom")
    labels = []
    for i, r in enumerate(metrics.itertuples(), 1):
        if r.family == "sentiment":
            if r.fund.endswith("_momentum"):
                name = "Sentiment - Momentum"
            elif r.fund.endswith("_tilt"):
                name = "Sentiment - Tilt"
            else:
                name = r.fund
        else:
            fam = d.FAMILY_LABELS.get(r.family, r.family.title())
            name = f"{fam} - {d.METHOD_LABELS.get(r.method, r.method)}"
        labels.append(f"{i}. {name}")
    ax.set_xlabel("Annualised volatility")
    ax.set_ylabel("CAGR")
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=6,
               markerfacecolor=d.family_color(r.family),
               markeredgecolor=d.family_color(r.family))
        for r in metrics.itertuples()
    ]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=3, fontsize=7, frameon=False, columnspacing=1.2,
               handletextpad=0.5)
    for r in metrics.itertuples():
        ax.scatter(r.ann_vol, r.cagr, color=d.family_color(r.family), s=46, zorder=3)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pt = fig.dpi / 72.0
    marker_r = (float(np.sqrt(46) / 2) + 1.0) * pt
    axes_bbox = ax.get_window_extent(renderer=renderer)
    disp = ax.transData.transform(metrics[["ann_vol", "cagr"]].to_numpy())
    font = FontProperties(size=7.5, weight="bold")
    ring = []
    for dist in (9, 14, 20, 27, 35, 44, 55, 68, 84, 102, 124, 150):
        ring.extend([
            (0, dist), (0, -dist), (dist, 0), (-dist, 0),
            (dist, dist), (dist, -dist), (-dist, dist), (-dist, -dist),
        ])
    placed: list[list[float]] = []
    offsets: dict[int, tuple[float, float]] = {}
    for i, r in enumerate(metrics.itertuples(), 1):
        w, h, _ = renderer.get_text_width_height_descent(str(i), font, ismath=False)
        bw, bh = w + 6.0, h + 9.0
        xd, yd = disp[i - 1]
        chosen = None
        best = None
        best_score = float("inf")
        for odx, ody in ring:
            ox, oy = xd + odx * pt, yd + ody * pt
            box = [ox - bw / 2, oy - bh / 2, ox + bw / 2, oy + bh / 2]
            inside = axes_bbox.contains(box[0] - 2, box[1] - 2) and \
                axes_bbox.contains(box[2] + 2, box[3] + 2)
            label_hits = sum(_bboxes_overlap(box, other) for other in placed)
            marker_hits = sum(
                _circle_box_overlap(px, py, marker_r, box) for px, py in disp)
            if inside and label_hits == 0 and marker_hits == 0:
                chosen = (odx, ody, box)
                break
            score = label_hits * 100 + marker_hits * 100 + (0 if inside else 1000)
            if score < best_score:
                best_score = score
                best = (odx, ody, box)
        if chosen is None:
            chosen = best
        odx, ody, box = chosen
        offsets[i] = (odx, ody)
        placed.append(box)
    for i, r in enumerate(metrics.itertuples(), 1):
        odx, ody = offsets[i]
        ax.annotate(str(i), (r.ann_vol, r.cagr), fontsize=7.5, fontweight="bold",
                    textcoords="offset points", xytext=(odx, ody),
                    ha="center", va="center", zorder=5, color=d.AS_COLORS["ink"])
        if (odx ** 2 + ody ** 2) ** 0.5 * pt > 25:
            xd, yd = disp[i - 1]
            lx, ly = ax.transData.inverted().transform((xd + odx * pt, yd + ody * pt))
            ax.plot([r.ann_vol, lx], [r.cagr, ly], ls="-", lw=0.6,
                    color=d.AS_COLORS["slate"], alpha=0.85, zorder=2)
    fig_path = Path(out_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300,
                bbox_inches=Bbox.from_bounds(0, 0, *fig.get_size_inches()),
                facecolor=d.AS_COLORS["paper"])
    plt.close(fig)
    cap = d.caption(
        title,
        sample=sample,
        units="CAGR and annualised volatility per year",
        note="Each point is numbered and keyed in the legend; point colour is the asset "
             "family (equity cyan, crypto amber, combined navy, sentiment violet).",
    )
    return f"{cap} Path: {fig_path}"


_SECTOR_LABELS = {
    "Comm": "Comm",
    "Consumer": "Consumer",
    "Energy": "Energy",
    "Financials": "Financials",
    "Healthcare": "Healthcare",
    "Industrials": "Industrials",
    "Materials": "Materials",
    "RealEstate": "Real Estate",
    "Tech": "Tech",
    "Utilities": "Utilities",
}


def sentiment_index(
    index_wide: pd.DataFrame,
    market_index: pd.DataFrame,
    coverage: pd.DataFrame,
    out_path: str | Path,
    *,
    title: str,
    sample: str,
) -> str:
    """Sector news-sentiment index over time (required exhibit).

    Upper panel: each of the ten sector indices as a thin line, the equal-weight
    market index as the bold navy line, and a shaded min-max band that traces the
    daily cross-sector spread (the dispersion regime the report reads as the
    tradeable signal). Lower panel: the mean fraction of each sector's tickers
    with news that day, so sparse sectors are visible rather than hidden. The
    legend lives in its own row beneath both panels so no text overlaps the axes.
    """
    with matplotlib.rc_context({"figure.constrained_layout.use": False}):
        fig = plt.figure(figsize=(8.2, 7.0))
    fig.set_layout_engine("none")
    d.apply_as_theme("word_a4")

    x0 = index_wide.index.min()
    x1 = index_wide.index.max()

    ax1 = fig.add_axes([0.085, 0.38, 0.895, 0.54])
    ax2 = fig.add_axes([0.085, 0.16, 0.895, 0.15])

    market = market_index.set_index("date")["sentiment"]
    sector_min = index_wide.min(axis=1)
    sector_max = index_wide.max(axis=1)
    ax1.fill_between(index_wide.index, sector_min.values, sector_max.values,
                     color=d.AS_COLORS["cyan"], alpha=0.13, lw=0,
                     label="Sector range (min-max)")
    for i, sector in enumerate(index_wide.columns):
        ax1.plot(index_wide.index, index_wide[sector].values,
                 color=d.AS_SECTOR_COLORS[i % len(d.AS_SECTOR_COLORS)],
                 lw=0.8, alpha=0.7, label=_SECTOR_LABELS.get(sector, sector))
    ax1.plot(market.index, market.values, lw=2.6, color=d.AS_COLORS["navy"],
             label="Market index (equal-weight mean)", zorder=5)
    ax1.axhline(0.0, color=d.AS_COLORS["slate"], lw=0.8, ls="--")
    ax1.set_ylabel("Sentiment (compound, lagged)")
    ax1.tick_params(labelbottom=False)
    ax1.set_xlim(x0, x1)

    cov_piv = coverage.pivot_table(index="date", values="n_covered",
                                   aggfunc=lambda s: s.mean(skipna=True))
    mean_cov = cov_piv.iloc[:, 0] / coverage["n_total"].max()
    ax2.plot(mean_cov.index, mean_cov.values, color=d.AS_COLORS["teal"], lw=1.3)
    ax2.set_ylabel("News coverage")
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.set_xlabel("Trading date")
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.set_xlim(x0, x1)

    handles, labels = ax1.get_legend_handles_labels()
    order = [*range(1, len(handles) - 1), 0, len(handles) - 1]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               fontsize=7, frameon=False, columnspacing=1.1,
               handletextpad=0.5, handlelength=1.4)

    fig.canvas.draw()
    fig.savefig(out_path, dpi=300,
                bbox_inches=Bbox.from_bounds(0, 0, *fig.get_size_inches()),
                facecolor=d.AS_COLORS["paper"])
    plt.close(fig)
    fig_path = Path(out_path)
    mkt_mean = float(market.mean())
    mkt_sd = float(market.std())
    cap = d.caption(
        title,
        sample=sample,
        units="Mean FinSent compound score (-1 bearish to +1 bullish), lagged one trading day",
        note=f"Each thin line is one of the ten sector indices; the bold navy line is the "
             f"equal-weight market index (mean {mkt_mean:.3f}, sd {mkt_sd:.3f} over "
             f"{len(market):,} trading days). The shaded band traces the daily min-max "
             f"spread across sectors: the lines converge in crisis episodes and diverge in "
             f"calm periods, and that dispersion is the cross-sectional signal the report "
             f"reads. Sector indices equal-weight their tickers; ticker-days with no "
             f"headlines are neutral. Lower panel: mean fraction of each sector's tickers "
             f"with news that day.",
    )
    return f"{cap} Path: {fig_path}"


def fusion_compare(
    base: pd.Series,
    tilted: pd.Series,
    out_path: str | Path,
    *,
    title: str,
    sample: str,
    base_label: str = "Base (max Sharpe)",
    tilt_label: str = "Sentiment tilt",
) -> str:
    """Fusion before-vs-after cumulative-return figure."""
    fig, ax = _fig(figsize=(8.2, 4.4))
    ax.plot(base.index, (1 + base).cumprod(), label=base_label,
            color=d.AS_COLORS["slate"], lw=2.0)
    ax.plot(tilted.index, (1 + tilted).cumprod(), label=tilt_label,
            color=d.method_color("sentiment_tilt"), lw=2.2)
    ax.axhline(1.0, color=d.AS_COLORS["slate"], lw=0.8, ls="--", alpha=0.7)
    ax.set_ylabel("Growth of $1 (net of fees)")
    ax.set_xlabel("Trading date")
    ax.legend()
    fig_path = d.save_figure(fig, out_path)
    cap = d.caption(
        title,
        sample=sample,
        units="Indexed to $1.00 at the first live backtest date",
        note="Both series use identical decision dates and the same expanding estimation window.",
    )
    return f"{cap} Path: {fig_path}"


def sentiment_neutrality(
    plain: pd.Series,
    finsent: pd.Series,
    out_path: str | Path,
    *,
    title: str,
    sample: str,
) -> str:
    """Fraction of headlines scoring exactly neutral: plain VADER vs FinSent."""
    frac = {
        "Plain VADER": float((plain == 0.0).mean()),
        "FinSent extension": float((finsent == 0.0).mean()),
    }
    fig, ax = _fig(figsize=(5.6, 3.8))
    labels = list(frac)
    vals = [frac[k] for k in labels]
    colors = (d.AS_COLORS["slate"], d.AS_COLORS["cyan"])
    bars = ax.bar(labels, vals, color=colors, alpha=0.9, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.1%}",
                ha="center", fontsize=9.5, color=d.AS_COLORS["ink"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Fraction of headlines scoring 0")
    fig_path = d.save_figure(fig, out_path)
    cap = d.caption(
        title,
        sample=sample,
        units="Fraction of distinct headline titles scoring compound == 0",
        note="A zero score is a false neutral, not 'no information'; the finance "
             "lexicon is designed to move these.",
    )
    return f"{cap} Path: {fig_path}"
