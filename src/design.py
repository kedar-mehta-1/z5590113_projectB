"""AlphaStream design system - a custom visual identity for figures and the app.

The design system is original work rather than the provided course style: a
deep-navy base inspired by data-centre dark themes, an electric-cyan accent
(the "signal"), amber for warnings/highlights, and muted slate for secondary
information. All figures, tables, and the Streamlit app share this palette so
the product has one coherent look.

Palette rationale (color-blind safe):
    - navy / ink    : structure, text, primary panels
    - cyan          : the primary accent, "AlphaStream signal"
    - amber         : risk / highlight (drawdowns, warnings)
    - coral         : positive emphasis in diverging views
    - teal / green  : calm positive (used sparingly)
    - slate / mist  : grid lines, secondary text, muted series
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

AS_COLORS = {
    "navy": "#0B1E36",      # deep navy - primary brand
    "ink": "#101828",       # near-black text
    "slate": "#4A5B6E",     # secondary text / muted series
    "mist": "#D8E1EA",      # grid lines
    "paper": "#FFFFFF",     # background
    "cyan": "#17BEBB",      # primary accent - the AlphaStream signal
    "amber": "#F0A500",     # risk / highlight
    "coral": "#E4572E",     # positive emphasis (diverging)
    "teal": "#0F766E",      # calm positive
    "violet": "#6B5B95",    # auxiliary series
    "green": "#2E7D32",     # positive
    "red": "#B23A48",       # negative
    "gold": "#C99700",      # highlight
    "blue": "#0F5499",      # auxiliary series
    "orange": "#D56F3E",    # auxiliary series
    "pink": "#E95D8E",      # auxiliary series
}

# Method colours (consistent across every figure)
AS_METHOD_COLORS = {
    "equal_weight": AS_COLORS["slate"],
    "min_variance": AS_COLORS["teal"],
    "max_sharpe": AS_COLORS["coral"],
    "risk_parity": AS_COLORS["violet"],
    "min_cvar": AS_COLORS["amber"],
    "sentiment_tilt": AS_COLORS["cyan"],
    "sentiment_momentum": AS_COLORS["blue"],
    "regime": AS_COLORS["blue"],
}

# Family colours
AS_FAMILY_COLORS = {
    "equity": AS_COLORS["cyan"],
    "crypto": AS_COLORS["amber"],
    "combined": AS_COLORS["navy"],
    "sentiment": AS_COLORS["violet"],
}

AS_SECTOR_COLORS = [
    AS_COLORS["navy"], AS_COLORS["cyan"], AS_COLORS["teal"], AS_COLORS["amber"],
    AS_COLORS["coral"], AS_COLORS["violet"], AS_COLORS["blue"], AS_COLORS["gold"],
    AS_COLORS["orange"], AS_COLORS["pink"],
]

AS_COLOR_CYCLE = [
    AS_COLORS["navy"], AS_COLORS["cyan"], AS_COLORS["amber"], AS_COLORS["coral"],
    AS_COLORS["teal"], AS_COLORS["violet"], AS_COLORS["blue"], AS_COLORS["gold"],
    AS_COLORS["orange"], AS_COLORS["pink"],
]

# Display labels shared by figures, tables, and the app.
METHOD_LABELS = {
    "equal_weight": "Equal Weight",
    "min_variance": "Minimum Variance",
    "max_sharpe": "Maximum Sharpe",
    "risk_parity": "Risk Parity",
    "min_cvar": "Minimum CVaR",
    "sentiment_tilt": "Max Sharpe + Sentiment",
    "sentiment_momentum": "Sentiment Momentum",
    "regime": "Regime Aware",
}

FAMILY_LABELS = {
    "equity": "Equity only",
    "crypto": "Crypto only",
    "combined": "Combined",
    "sentiment": "Sentiment funds",
}

FUND_LABELS = {
    f"{family}_{method}": f"{FAMILY_LABELS[family]} - {METHOD_LABELS[method]}"
    for family in FAMILY_LABELS
    for method in METHOD_LABELS
}


# ---------------------------------------------------------------------------
# Matplotlib theme
# ---------------------------------------------------------------------------

def as_rcparams(profile: str = "word_a4") -> dict[str, object]:
    """Return matplotlib rcParams for the AlphaStream design system."""
    base: dict[str, object] = {
        "axes.edgecolor": AS_COLORS["slate"],
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.labelcolor": AS_COLORS["ink"],
        "axes.labelsize": 10.5,
        "axes.linewidth": 0.8,
        "axes.prop_cycle": cycler(color=AS_COLOR_CYCLE),
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.titlelocation": "left",
        "axes.titlecolor": AS_COLORS["navy"],
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "figure.autolayout": False,
        "figure.constrained_layout.use": True,
        "figure.dpi": 130,
        "figure.facecolor": AS_COLORS["paper"],
        "font.family": "DejaVu Sans",
        "grid.alpha": 0.3,
        "grid.color": AS_COLORS["mist"],
        "grid.linewidth": 0.6,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 1.9,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "savefig.facecolor": AS_COLORS["paper"],
        "savefig.pad_inches": 0.04,
        "xtick.color": AS_COLORS["slate"],
        "xtick.labelsize": 9.5,
        "ytick.color": AS_COLORS["slate"],
        "ytick.labelsize": 9.5,
    }
    if profile == "word_a4":
        base.update({"axes.labelsize": 11, "axes.titlesize": 13, "figure.dpi": 150,
                     "lines.linewidth": 2.0, "xtick.labelsize": 10, "ytick.labelsize": 10})
    elif profile == "app":
        base.update({"figure.dpi": 110, "axes.labelsize": 10, "axes.titlesize": 12,
                     "lines.linewidth": 1.8})
    return base


def apply_as_theme(profile: str = "word_a4") -> None:
    """Apply the AlphaStream theme globally for the current session."""
    mpl.rcParams.update(as_rcparams(profile))


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 300) -> Path:
    """Save a figure as a high-resolution PNG with the AlphaStream theme."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=AS_COLORS["paper"])
    plt.close(fig)
    return path


def method_color(method: str) -> str:
    return AS_METHOD_COLORS.get(method, AS_COLORS["slate"])


def family_color(family: str) -> str:
    return AS_FAMILY_COLORS.get(family, AS_COLORS["navy"])


# ---------------------------------------------------------------------------
# Caption helper (report exhibits)
# ---------------------------------------------------------------------------

def caption(title: str, *, sample: str, units: str = "", note: str = "",
            source: str = ("AlphaStream research pipeline, computed from "
                           "hosted project data")) -> str:
    """Build a self-contained exhibit caption string."""
    parts = [title, f"Sample period: {sample}."]
    if units:
        parts.append(f"Units: {units}.")
    if note:
        parts.append(note)
    if source:
        parts.append(f"Source: {source}.")
    return " ".join(parts)
