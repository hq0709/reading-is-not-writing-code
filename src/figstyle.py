"""Figure style, taken from published figures rather than from taste.

Two schools appear in top-venue papers and they do not mix well. The first, used by Darcet et al. (ICLR
2024) and Schaeffer et al. (NeurIPS 2023), is minimal: sans type, two spines, no grid, no legend, every
claim in the caption. The second, standard in CVPR and ICCV results figures, is the one this module now
implements, because it is what the reader of this paper asked for and because it suits a results figure
with several comparable series:

  type        a serif that matches the body text. Tinos is metrically Times, so a figure set in it sits
              beside the paper's TeX Gyre Termes without a visible seam. A sans figure in a Times paper
              always reads as pasted in.
  axes        a full box, thin, with a pale grid behind the data. The box gives the panel a boundary the
              eye can rest on when several panels are compared side by side.
  series      colour AND marker shape. Colour alone fails in print and for a colourblind reader; the
              published figure uses circle, square, triangle and diamond.
  emphasis    the series that carries the argument is drawn thicker and in the brightest hue. Everything
              else recedes.
  reference   an upper or lower bound is a grey dashed horizontal line with a small grey inline label,
              not an annotation with an arrow.
  legend      framed, inside the panel, in the corner the data leaves empty.
  palette     Okabe and Ito, which is colourblind safe and is close to the soft orange, rose, violet and
              sky blue the reference figure uses.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(_HERE, "fonts")
SERIF, SANS = "DejaVu Serif", "DejaVu Sans"
if os.path.isdir(FONT_DIR):
    for f in os.listdir(FONT_DIR):
        if f.lower().endswith((".ttf", ".otf")):
            try:
                fm.fontManager.addfont(os.path.join(FONT_DIR, f))
            except Exception:
                pass
    have = {f.name for f in fm.fontManager.ttflist}
    if "Tinos" in have:
        SERIF = "Tinos"
    if "Lato" in have:
        SANS = "Lato"
FAMILY = SERIF

# Okabe and Ito, colourblind safe, close to the reference figure's soft palette
ORANGE, ROSE, VIOLET, SKY = "#E69F00", "#CC79A7", "#8C77B5", "#4BA3DC"
GREEN, VERMILLION, BLUE = "#009E73", "#D55E00", "#0072B2"
INK, MUTE, GRID, HAIR = "#1a1a1a", "#7a7a7a", "#e4e6e9", "#c8ccd1"
GOOD, BAD = "#0072B2", "#D55E00"   # blue / vermillion
BG_GOOD, BG_BAD = "#eef4fa", "#fdf1e9"
POS, NEG = "#D55E00", "#a9b8c6"
MARKERS = ["o", "s", "^", "D", "v", "P"]

BASE = 7.0
LABEL = 8.0
TITLE = 8.4
ANNOT = 6.6


def use(scale: float = 1.0):
    plt.rcParams.update({
        "font.family": FAMILY, "font.serif": [SERIF], "font.size": BASE * scale,
        "mathtext.fontset": "custom", "mathtext.rm": SERIF, "mathtext.it": f"{SERIF}:italic",
        "mathtext.bf": f"{SERIF}:bold", "mathtext.cal": SERIF, "mathtext.tt": SANS,
        "mathtext.sf": SANS,
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.edgecolor": "#4a4a4a", "axes.linewidth": 0.7,
        "axes.labelsize": LABEL * scale, "axes.labelcolor": INK, "axes.labelpad": 3.0,
        "axes.titlesize": TITLE * scale, "axes.titleweight": "normal", "axes.titlepad": 4.5,
        "axes.titlecolor": INK, "axes.axisbelow": True,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.55, "grid.alpha": 1.0,
        "xtick.labelsize": BASE * scale, "ytick.labelsize": BASE * scale,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": HAIR,
        "legend.fancybox": False, "legend.fontsize": ANNOT * scale,
        "legend.handlelength": 1.8, "legend.handletextpad": 0.6, "legend.borderpad": 0.45,
        "legend.labelspacing": 0.35, "legend.borderaxespad": 0.5,
        "figure.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
        # round joins and caps matter more than they sound. These series zigzag by design, and a mitred
        # join throws a spike off the outside of every acute corner, which is what makes a thin line
        # read as ragged rather than as drawn.
        "lines.linewidth": 1.2, "lines.markersize": 3.2, "lines.markeredgewidth": 0.55,
        "lines.solid_joinstyle": "round", "lines.solid_capstyle": "round",
        "lines.dash_joinstyle": "round", "lines.dash_capstyle": "round",
        "lines.markeredgecolor": "white", "lines.antialiased": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def reference_line(ax, y, label, x=0.99, color=MUTE):
    """A bound drawn as a grey dashed rule with a small inline label, as in the reference figure."""
    ax.axhline(y, color=color, lw=0.8, ls=(0, (4, 2.5)), zorder=1)
    ax.text(x, y, label, transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=ANNOT - 0.4, color=color)


def series(ax, x, y, label, i, emphasis=False, color=None):
    """One series: its own colour and its own marker, a little heavier if it carries the argument.

    Solid throughout. Dashing every non-emphasised series was the earlier choice and it looked coarse:
    at 1.2 pt a 4-on-2-off dash is a row of blocks, and with four series the panel fills with texture that
    carries no information the marker shape does not already carry. Separation comes from hue and from
    marker shape; weight is reserved for saying which line is the argument.
    """
    cols = [ORANGE, ROSE, VIOLET, SKY, GREEN, VERMILLION]
    c = color or cols[i % len(cols)]
    return ax.plot(x, y, marker=MARKERS[i % len(MARKERS)], label=label, color=c, ls="-",
                   lw=1.25 if emphasis else 1.0, ms=3.6 if emphasis else 2.9,
                   markeredgecolor="white", markeredgewidth=0.55,
                   zorder=4 if emphasis else 3)


def mixed(ax, x, y, parts, size=None, ha="left", transform=None, **kw):
    """One line of text whose fragments carry different colours.

    Used instead of a legend when the categories are named in the sentence anyway: colouring the words
    themselves removes a box the reader would otherwise have to look up. Fragments are laid out by
    measuring each one with the live renderer, because matplotlib has no rich-text primitive.
    """
    size = size if size is not None else ANNOT
    tr = transform if transform is not None else ax.transAxes
    fig = ax.figure
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    inv = tr.inverted()

    widths = []
    for text, _c, weight in parts:
        t = ax.text(0, 0, text, transform=tr, fontsize=size, fontweight=weight, alpha=0)
        bb = t.get_window_extent(renderer=ren)
        widths.append(inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0])
        t.remove()

    total = sum(widths)
    cur = x if ha == "left" else x - total if ha == "right" else x - total / 2
    out = []
    for (text, colour, weight), w in zip(parts, widths):
        out.append(ax.text(cur, y, text, transform=tr, fontsize=size, color=colour,
                           fontweight=weight, ha="left", va="baseline", **kw))
        cur += w
    return out


def panel(ax, letter, title="", y=1.03, size=None):
    """The letter in bold and the panel's claim in regular, left-aligned above the axes.

    Reference figures put the letter outside the data area and set it apart by weight rather than by size
    or a box, so it reads as an index and not as a data label. The title carries the panel's claim, which
    keeps the caption for the method: a reader who looks only at the figure still gets the argument.
    """
    size = size if size is not None else TITLE
    t = ax.text(0.0, y, letter, transform=ax.transAxes, fontsize=size, fontweight="bold",
                va="bottom", ha="left", color=INK)
    if not title:
        return t
    fig = ax.figure
    fig.canvas.draw()
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    inv = ax.transAxes.inverted()
    w = inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]
    ax.text(w + 0.035, y, title, transform=ax.transAxes, fontsize=size, va="bottom", ha="left",
            color=INK)
    return t
