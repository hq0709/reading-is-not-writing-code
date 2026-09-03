"""Shared publication style for the evidence-locked paper figures."""

from pathlib import Path

import matplotlib.pyplot as plt


PAPER = Path(__file__).resolve().parents[1]
FIGURES = PAPER / "figures"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#707070"
LIGHT_GRAY = "#D9D9D9"

plt.rcParams.update(
    {
        "font.size": 9,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.pdf"
    fig.savefig(path)
    print(path)
