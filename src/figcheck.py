"""Automated overlap and clipping check for matplotlib figures.

Overlapping labels have been the recurring defect in this project's figures, and eyeballing a render at
screen size does not catch them: a collision that is obvious at 400 dpi on a monitor is invisible in a
thumbnail and fatal at 5.5 inches on paper. This does it arithmetically instead, on the drawn figure,
before it is saved.

    from figcheck import audit
    audit(fig, "fig_depth")      # prints findings, returns a list; raises nothing

Checks, in the order they matter:

    text-vs-text        two labels whose ink rectangles intersect
    text-out-of-axes    a label whose box leaves the axes it belongs to (silently clipped at save)
    text-out-of-figure  a label outside the canvas (lost entirely)
    legend-over-data    a legend frame covering plotted vertices, which is the collision no one sees
                        until the data changes
    tick-crowding       adjacent tick labels closer than a hair, which reads as a smear in print

Tolerances are in points so they mean the same thing at any dpi.
"""
from __future__ import annotations

import numpy as np
from matplotlib.text import Text
from matplotlib.legend import Legend

PAD = 0.5          # points of slack before two boxes count as touching
MIN_TICK_GAP = 1.5  # points between adjacent tick labels


def _boxes(fig, renderer):
    out = []
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip():
                out.append((t, ax, "text"))
        for lbl in (ax.xaxis.label, ax.yaxis.label, ax.title):
            if lbl.get_visible() and lbl.get_text().strip():
                out.append((lbl, ax, "axis-label"))
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip():
            out.append((t, None, "figure-text"))
    return out


def _bb(artist, renderer):
    try:
        return artist.get_window_extent(renderer=renderer)
    except Exception:
        return None


def _overlap(a, b, pad=PAD):
    """Intersection of two bboxes, shrunk by pad on each side so touching is not a collision."""
    x0 = max(a.x0, b.x0) + pad
    x1 = min(a.x1, b.x1) - pad
    y0 = max(a.y0, b.y0) + pad
    y1 = min(a.y1, b.y1) - pad
    return (x1 - x0) > 0 and (y1 - y0) > 0


def audit(fig, name="figure", verbose=True):
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    dpi_scale = 72.0 / fig.dpi           # device pixels -> points
    bad = []

    items = [(a, ax, kind, _bb(a, ren)) for a, ax, kind in _boxes(fig, ren)]
    items = [it for it in items if it[3] is not None and it[3].width > 0]

    # 1. label against label
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            ai, axi, ki, bi = items[i]
            aj, axj, kj, bj = items[j]
            if _overlap(bi, bj):
                w = (min(bi.x1, bj.x1) - max(bi.x0, bj.x0)) * dpi_scale
                h = (min(bi.y1, bj.y1) - max(bi.y0, bj.y0)) * dpi_scale
                bad.append(f"text/text  {ai.get_text()[:26]!r} x {aj.get_text()[:26]!r}"
                           f"  ({w:.1f}x{h:.1f} pt)")

    # 2. label leaving its axes, and label leaving the canvas
    figbb = fig.bbox
    for a, ax, kind, bb in items:
        if ax is not None and kind == "text":
            axbb = ax.get_window_extent(renderer=ren)
            if bb.x0 < axbb.x0 - 1 or bb.x1 > axbb.x1 + 1 or bb.y0 < axbb.y0 - 1 or bb.y1 > axbb.y1 + 1:
                # only a problem if the text was meant to sit inside; annotations outside are common,
                # so this is reported as a warning the author confirms rather than a hard failure
                if a.get_clip_on():
                    bad.append(f"clipped    {a.get_text()[:34]!r} leaves its axes and clipping is on")
        # Axis labels and figure-level text routinely sit outside the initial canvas; savefig with
        # bbox_inches="tight" expands to include them, so flagging those is noise. Only text placed
        # inside an axes can actually be lost.
        if kind == "text" and (bb.x0 < figbb.x0 - 1 or bb.x1 > figbb.x1 + 1
                               or bb.y0 < figbb.y0 - 1 or bb.y1 > figbb.y1 + 1):
            bad.append(f"off-canvas {a.get_text()[:34]!r} extends past the figure edge")

    # 3. a label sitting on top of plotted data. This is the collision that text-vs-text misses and it
    #    is the one that keeps happening: a label placed in what looked like empty space at design time
    #    and then buried when the data changed.
    for a, ax, kind, bb in items:
        if ax is None or kind != "text":
            continue
        hits = 0
        for ln in ax.lines:
            d = ln.get_xydata()
            if d is None or len(d) < 1:
                continue
            d = d[np.isfinite(d).all(axis=1)]
            if len(d) < 1:
                continue
            if len(d) > 1:
                t = np.linspace(0, 1, 24)[:, None]
                seg = (d[:-1][None] * (1 - t)[:, :, None] + d[1:][None] * t[:, :, None])
                d = np.vstack([d, seg.reshape(-1, 2)])
            p_ = ax.transData.transform(d)
            hits += int(np.sum((p_[:, 0] > bb.x0) & (p_[:, 0] < bb.x1) &
                               (p_[:, 1] > bb.y0) & (p_[:, 1] < bb.y1)))
        if hits > 2:
            bad.append(f"text/data  {a.get_text()[:34]!r} sits on {hits} plotted point(s)")

    # 4. legend over plotted data
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None or not leg.get_visible():
            continue
        lb = _bb(leg, ren)
        if lb is None:
            continue
        hits = 0
        for ln in ax.lines:
            d = ln.get_xydata()
            if d is None or len(d) == 0:
                continue
            d = d[np.isfinite(d).all(axis=1)]
            if len(d) < 1:
                continue
            # sample along the segments, not only at the vertices: a two-point line passing straight
            # through a legend has no vertex inside it, which is exactly the case the eye misses
            if len(d) > 1:
                t = np.linspace(0, 1, 24)[:, None]
                seg = (d[:-1][None] * (1 - t)[:, :, None] + d[1:][None] * t[:, :, None])
                d = np.vstack([d, seg.reshape(-1, 2)])
            pts = ax.transData.transform(d)
            hits += int(np.sum((pts[:, 0] > lb.x0) & (pts[:, 0] < lb.x1) &
                               (pts[:, 1] > lb.y0) & (pts[:, 1] < lb.y1)))
        for col in ax.collections:
            try:
                pts = col.get_offsets()
                if len(pts) == 0:
                    continue
                pts = ax.transData.transform(np.asarray(pts))
                hits += int(np.sum((pts[:, 0] > lb.x0) & (pts[:, 0] < lb.x1) &
                                   (pts[:, 1] > lb.y0) & (pts[:, 1] < lb.y1)))
            except Exception:
                pass
        if hits:
            bad.append(f"legend     covers {hits} plotted point(s)")

    # 5. tick labels running into each other
    for ax in fig.axes:
        for axis, nm in ((ax.xaxis, "x"), (ax.yaxis, "y")):
            bbs = [(_bb(t, ren), t.get_text()) for t in axis.get_ticklabels()
                   if t.get_visible() and t.get_text().strip()]
            bbs = [(b, s) for b, s in bbs if b is not None and b.width > 0]
            key = 0 if nm == "x" else 1
            bbs.sort(key=lambda z: (z[0].x0 if key == 0 else z[0].y0))
            for (b1, s1), (b2, s2) in zip(bbs, bbs[1:]):
                gap = ((b2.x0 - b1.x1) if key == 0 else (b2.y0 - b1.y1)) * dpi_scale
                if gap < MIN_TICK_GAP:
                    bad.append(f"ticks      {nm}: {s1!r} and {s2!r} are {gap:.1f} pt apart")

    if verbose:
        if bad:
            print(f"  [figcheck] {name}: {len(bad)} problem(s)")
            for b in dict.fromkeys(bad):
                print(f"      {b}")
        else:
            print(f"  [figcheck] {name}: clean")
    return bad
