"""House style for every cf-transfer-v1 figure, transcribed from `journal/figs/house.py` of the MedVIGIL-paper
project (the journal figure standard): serif type matching the body face (Nimbus Roman / STIX / Times), a full
box frame with a light solid grid, a fixed Morandi categorical palette (eight muted hues ordered by lightness), dashed lines with white-edged
markers, boxed legends, "(a)  ..." panel titles, bold value call-outs, red dashed reference lines with inline
labels; PDF (Type 42 fonts) as the primary output, PNG at 300 dpi and SVG with editable text alongside.

Figure widths follow the journal layout: 7.2 in for a double-column figure (0.96 textwidth), 3.5 in for a
single-column figure; one-row figures are 2.7-2.9 in tall, two-row figures 3.9-5.3 in.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

RC = {
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.7, "axes.edgecolor": "#7a7a7a",
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.grid": True, "grid.color": "#e6e4e1", "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True,
    "xtick.color": "#444444", "ytick.color": "#444444", "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": True, "legend.framealpha": 1.0, "legend.edgecolor": "#8a8a8a", "legend.fancybox": False,
    "legend.borderpad": 0.4, "legend.handlelength": 2.2, "legend.handletextpad": 0.5,
    "lines.linewidth": 1.5, "lines.markersize": 5.5, "lines.markeredgewidth": 0.9, "lines.markeredgecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
}


def use_house_style():
    plt.rcParams.update(RC)


# Morandi palette: muted, dusty, low-saturation tones, ordered by lightness so series stay distinguishable
# without hue (colour-blind safe by lightness as well as hue). Eight chromatic slots plus warm neutrals.
ROSE = "#C08585"          # dusty rose          L ~ 62
SAGE = "#8FA68E"          # sage green          L ~ 65
HAZE = "#7D9AB2"          # greyish blue        L ~ 62
OAT = "#C6A87C"           # warm greige         L ~ 70
LILAC = "#9B8EA9"         # dusty lilac         L ~ 61
MUSTARD = "#C9B36A"       # muted mustard       L ~ 73
TERRACOTTA = "#B5766A"    # faded terracotta    L ~ 55
SLATE = "#6F7C8A"         # slate               L ~ 51
CHARCOAL, GREY, STONE = "#5C5B58", "#B8B4AE", "#D6D2CC"   # warm neutrals: ink, mid grey, light grey
RED, GREEN = TERRACOTTA, "#6E8B6B"                        # terracotta: reference lines only; moss: positive call-outs
BAND = "#EEF2EC"                                          # sage tint for shaded bands
SLOTS = [ROSE, SAGE, HAZE, OAT, LILAC, MUSTARD, TERRACOTTA, SLATE]   # fixed order for arms / tasks / series
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, MUTED, CHANCE = "#3a3a3a", "#6a6a6a", "#9a9a9a"
BLACK = CHARCOAL

# one colour and marker per dataset across every figure of the campaign (three lightness levels)
DATASET_COLOR = {"nih": ROSE, "chexpert": OAT, "coco": SLATE}
DATASET_MARKER = {"nih": "o", "chexpert": "s", "coco": "^"}
DATASET_LABEL = {"nih": "NIH ChestX-ray14", "chexpert": "CheXpert Plus", "coco": "COCO (control)"}

# figure widths of the journal layout (inches)
DOUBLE, SINGLE = 7.2, 3.5

# compact legend used for dense panels
LEG = dict(fontsize=6.3, handlelength=1.4, borderpad=0.35, labelspacing=0.3, handletextpad=0.4, columnspacing=1.0)

# warm sequential map used for share heat-maps: paper tint -> rose
HEAT = LinearSegmentedColormap.from_list("share", ["#F5F1EC", "#E3C9C5", ROSE, TERRACOTTA], N=256)


def panel_title(ax, letter, text):
    ax.set_title(f"({letter})  {text}", loc="center", fontsize=9, pad=5)


def series_handle(color, marker, label, ls="--"):
    return Line2D([], [], color=color, marker=marker, ls=ls, lw=1.5, ms=5.5, mec="white", mew=0.9, label=label)


def open_handle(color, marker, label):
    """Open marker (not owned) matching the filled series_handle."""
    return Line2D([], [], color=color, marker=marker, ls="none", ms=5.5, mfc="white", mec=color, mew=0.9, label=label)


def boxed_legend(fig_or_ax, handles, **kw):
    kw.setdefault("frameon", True); kw.setdefault("edgecolor", "#8a8a8a")
    return fig_or_ax.legend(handles=handles, **kw)


def value_label(ax, x, y, text, color, dx=0, dy=0, ha="center", va="bottom", size=8):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                fontsize=size, fontweight="bold", color=color)


def ref_line(ax, y=None, x=None, color=RED, label=None, ls="--", lw=1.0, where="right", pad=1.0):
    if y is not None:
        ax.axhline(y, color=color, ls=ls, lw=lw, zorder=2)
        if label:
            xr = ax.get_xlim()[1] if where == "right" else ax.get_xlim()[0]
            ax.annotate(label, (xr, y), xytext=(-3 if where == "right" else 3, pad), textcoords="offset points",
                        ha="right" if where == "right" else "left", va="bottom", fontsize=7.5, color=color)
    if x is not None:
        ax.axvline(x, color=color, ls=ls, lw=lw, zorder=2)
        if label:
            ax.annotate(label, (x, ax.get_ylim()[1]), xytext=(3, -3), textcoords="offset points",
                        ha="left", va="top", fontsize=7.5, color=color)


def chance_line(ax, y=None, x=None, label="chance", where="right"):
    """Neutral grey zero/chance line with a muted inline label (the house's chance-line convention)."""
    if y is not None:
        ax.axhline(y, color=CHANCE, lw=0.8, zorder=2)
        if label:
            xr = ax.get_xlim()[1] if where == "right" else ax.get_xlim()[0]
            ax.annotate(label, (xr, y), xytext=(-3 if where == "right" else 3, 2), textcoords="offset points",
                        ha="right" if where == "right" else "left", va="bottom", fontsize=6.2, color=MUTED)
    if x is not None:
        ax.axvline(x, color=CHANCE, lw=0.8, zorder=2)


def save(fig, base: Path, dpi: int = 300):
    """PDF is the paper format; PNG (300 dpi) and SVG (editable text) alongside."""
    base = Path(base); base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
