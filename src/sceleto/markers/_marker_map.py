"""Hierarchy marker map: tree + per-cluster marker grid (band or dot).

One figure, two scopes selected by ``icls``:

- ``icls=None`` (default): FULL map — every node's top-N markers (rows) over the
  whole hierarchy, ordered depth-first (lineage) or breadth-first (level bands).
- ``icls="<id>"``: SINGLE path — only that path's per-level top-N markers (like
  :meth:`HierarchyRun.compare_markers`, but as an expression grid).

In both, columns are every icls leaf (tree on top), rows are colored by their
origin level (level0 red / level1 green / level2 black), and each cell encodes
*expression* (color, via the level colormap) and *fraction of expressing cells*:

- ``mode="band"`` (default): a filled rectangle whose HEIGHT is the fraction.
  Rows pack tight, so the full map stays reasonably short.
- ``mode="dot"``: a scatter dot whose AREA is the fraction.

Everything is raw — no dedup (unless asked), no minimum thickness, no size cap.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

LEVEL_CMAPS = ["Reds", "Greens", "Greys"]


def _consec_groups(keys):
    """Group consecutive equal keys -> list of (key, [indices])."""
    groups, start = [], 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[start]:
            groups.append((keys[start], list(range(start, i))))
            start = i
    return groups


def marker_map(
    hr,
    icls: Optional[str] = None,
    *,
    mode: str = "band",
    order: str = "dfs",
    n_markers: Optional[int] = None,
    dedup: bool = False,
    columns: Union[str, Sequence[str]] = "all",
    color_norm: str = "global",
    color_floor: float = 0.30,
    max_dot: float = 170.0,
    min_frac: float = 0.0,
    show_tree: bool = True,
    highlight_ref: bool = True,
    row_h: Optional[float] = None,
    band_gap: Optional[float] = None,
    band_w: float = 0.92,
    figsize=None,
    save: Optional[str] = None,
):
    """Hierarchy marker map (see module docstring).

    Parameters
    ----------
    hr
        A :class:`~sceleto.markers.HierarchyRun`.
    icls
        ``None`` (default) draws the FULL map over every node; a path id draws
        only that path's per-level top-N markers.
    mode
        ``"band"`` (default) → cell height = fraction; ``"dot"`` → dot area =
        fraction. Both use color = expression (the row's level colormap).
    order
        FULL map only. ``"dfs"`` (default) groups rows by lineage/branch;
        ``"bfs"`` stacks all level0, then level1, then level2.
    n_markers
        Top-N markers per node/level. ``None`` → the run's ``n_top_markers``.
    dedup
        SINGLE path only. Collapse the 3 levels' markers into a set (like
        ``compare_markers``). Default ``False`` keeps duplicates.
    columns
        ``"all"`` (default) or an explicit list of icls ids; always re-ordered by
        the hierarchy so the tree is planar.
    color_norm
        ``"global"`` (stored ``mean_norm``, atlas-wide) or ``"row"`` (per-gene
        max among shown columns).
    color_floor, max_dot, min_frac
        Color floor, max dot area (``mode="dot"``), and the fraction below which
        no glyph is drawn.
    show_tree
        Draw the hierarchy dendrogram on top.
    highlight_ref
        SINGLE path only: shade the reference column.
    row_h, band_gap, band_w
        Row spacing (default 0.30 band / 0.60 dot), gap at band/branch breaks,
        and band rectangle width.
    figsize, save
        Manual size; save path (PDF, dpi=300).

    Returns
    -------
    ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    def _cmap(name):
        return plt.get_cmap(name)

    def _solid(name, v=0.85):
        return _cmap(name)(v)

    if mode not in ("band", "dot"):
        raise ValueError("mode must be 'band' or 'dot'.")

    levels = hr.levels
    g0, g1, g2 = levels
    df = hr.icls_path_df.set_index("icls")
    full = hr.full_gene_lists
    if n_markers is None:
        n_markers = hr.params["n_top_markers"]
    # band packs rows tighter than dots, but rows must stay tall enough that gene
    # labels (fontsize ~6.5) don't collide vertically.
    if row_h is None:
        row_h = 0.30 if mode == "band" else 0.60
    if band_gap is None:
        band_gap = 0.35 if mode == "band" else 0.55

    def clu_at(c, lvl):
        return df.loc[c, lvl].split("@", 1)[1]

    full_map = icls is None
    ref_icls = None if full_map else str(icls)
    if not full_map and ref_icls not in df.index:
        raise ValueError(f"marker_map: icls {ref_icls!r} not found.")

    # --- columns: all leaves (or subset), hierarchy-ordered ---
    if isinstance(columns, str) and columns == "all":
        cols = list(df.index)
    else:
        cols = [str(c) for c in columns]
    cols = sorted(cols, key=lambda c: (int(clu_at(c, g0)), int(clu_at(c, g1)),
                                       int(clu_at(c, g2))))

    # --- rows (gene, level_index) + gap-before flags ---
    rows, gaps = [], []

    def _emit(level_i, cluster_id, gap):
        first = True
        for gene in full[f"{levels[level_i]}@{cluster_id}"][:n_markers]:
            rows.append((gene, level_i))
            gaps.append(gap if first else False)
            first = False

    if full_map:
        if order == "dfs":
            prev = (None, None, None)
            for c in cols:
                a, b, cc = clu_at(c, g0), clu_at(c, g1), clu_at(c, g2)
                if a != prev[0]:
                    _emit(0, a, gap=len(rows) > 0)
                if (a, b) != prev[:2]:
                    _emit(1, b, gap=False)
                if (a, b, cc) != prev:
                    _emit(2, cc, gap=False)
                prev = (a, b, cc)
        elif order == "bfs":
            for li, g in enumerate(levels):
                seen = []
                for c in cols:
                    k = clu_at(c, g)
                    if k not in seen:
                        seen.append(k)
                for ci, k in enumerate(seen):
                    _emit(li, k, gap=(li > 0 and ci == 0))
        else:
            raise ValueError("order must be 'bfs' or 'dfs'.")
    else:
        ref = df.loc[ref_icls]
        ref_clu = [ref[g0], ref[g1], ref[g2]]
        seen = set()
        for li, lvl_str in enumerate(ref_clu):
            first = True
            for gene in full[lvl_str][:n_markers]:
                if dedup and gene in seen:
                    continue
                seen.add(gene)
                rows.append((gene, li))
                gaps.append(li > 0 and first)     # gap between the 3 level blocks
                first = False

    # --- expression at the finest (leaf) resolution ---
    ctx2 = hr.contexts[g2]
    gidx = {gg: i for i, gg in enumerate(ctx2.genes)}
    leaf_row = {c: ctx2.group_to_idx.get(clu_at(c, g2)) for c in cols}

    def expr(c, gene):
        r, col = leaf_row[c], gidx.get(gene)
        if r is None or col is None:
            return 0.0, 0.0
        return float(ctx2.mean_norm[r, col]), float(ctx2.frac_expr[r, col])

    # ----- geometry -----
    ncol = len(cols)
    xs = list(range(ncol))
    gene_y, y = [], 0.0
    for i, (gene, li) in enumerate(rows):
        if i > 0 and gaps[i]:
            y -= band_gap
        gene_y.append(y)
        y -= row_h
    y_bot = gene_y[-1] if gene_y else 0.0

    dy = 1.25
    Y_PATH = 0.75
    Y_L2 = Y_PATH + dy
    Y_L1 = Y_L2 + dy
    Y_L0 = Y_L1 + dy
    Y_ROOT = Y_L0 + 0.75
    y_top = (Y_ROOT + 0.3) if show_tree else (Y_PATH + 0.4)

    if figsize is None:
        yspan = y_top - y_bot
        figsize = (2.8 + ncol * 0.30, 2.0 + yspan * 0.42)
    fig, ax = plt.subplots(figsize=figsize)

    # reference-column highlight (single-path only)
    if (not full_map) and highlight_ref and ref_icls in cols:
        rx = xs[cols.index(ref_icls)]
        ax.add_patch(Rectangle((rx - 0.5, y_bot - 0.35), 1.0,
                    (y_top + 0.1) - (y_bot - 0.35),
                    facecolor="#eef3f8", edgecolor="none", zorder=0))

    # ---------- hierarchy tree ----------
    lc, lw = "#9a9a9a", 1.1

    def bracket(px, py, children_x, child_y):
        bar_y = py - 0.46
        ax.plot([px, px], [py - 0.18, bar_y], color=lc, lw=lw, zorder=1)
        if len(children_x) > 1:
            ax.plot([min(children_x), max(children_x)], [bar_y, bar_y], color=lc, lw=lw, zorder=1)
        for cx in children_x:
            ax.plot([cx, cx], [bar_y, child_y + 0.18], color=lc, lw=lw, zorder=1)

    if show_tree:
        c_g0 = [clu_at(c, g0) for c in cols]
        c_g1 = [clu_at(c, g1) for c in cols]
        c_g2 = [clu_at(c, g2) for c in cols]
        l0g = _consec_groups(c_g0)
        l1g = _consec_groups(list(zip(c_g0, c_g1)))
        l0x = {k: float(np.mean(m)) for k, m in l0g}
        l1x = [(k, float(np.mean(m)), m) for k, m in l1g]
        for i in range(ncol):
            ax.text(i, Y_L2, c_g2[i], ha="center", va="center", fontsize=8, zorder=4)
            ax.plot([i, i], [Y_L2 - 0.18, Y_PATH + 0.34], color=lc, lw=lw, zorder=1)
        for (k0, k1), px, members in l1x:
            ax.text(px, Y_L1, k1, ha="center", va="center", fontsize=8, zorder=4)
            bracket(px, Y_L1, [float(m) for m in members], Y_L2)
        for k0, m0 in l0g:
            px = l0x[k0]
            ax.text(px, Y_L0, k0, ha="center", va="center", fontsize=8, zorder=4)
            child_x = [x for (kk0, kk1), x, mm in l1x if kk0 == k0]
            bracket(px, Y_L0, child_x, Y_L1)
        rootx = sorted(l0x.values())
        if len(rootx) > 1:
            ax.plot([min(rootx), max(rootx)], [Y_ROOT, Y_ROOT], color=lc, lw=lw, zorder=1)
        for x in rootx:
            ax.plot([x, x], [Y_ROOT, Y_L0 + 0.18], color=lc, lw=lw, zorder=1)
        for yy, name, cmapname in [(Y_L0, g0, LEVEL_CMAPS[0]), (Y_L1, g1, LEVEL_CMAPS[1]),
                                   (Y_L2, g2, LEVEL_CMAPS[2])]:
            ax.text(-1.0, yy, name, ha="right", va="center", fontsize=8.5,
                    color=_solid(cmapname, 0.9))
        ax.text(-1.0, Y_PATH, "path", ha="right", va="center", fontsize=8.5, color="black")

    # ---------- path (icls) circles ----------
    for x, c in zip(xs, cols):
        is_ref = (not full_map) and (c == ref_icls)
        ax.add_patch(Circle((x, Y_PATH), 0.33, facecolor="#c9c9c9", edgecolor="black",
                    lw=1.8 if is_ref else 0.8, zorder=3))
        ax.text(x, Y_PATH, c, ha="center", va="center", fontsize=7.5, color="black",
                fontweight="bold" if is_ref else "normal", zorder=4)

    # ---------- cells (band or dot) ----------
    for (gene, li), gy in zip(rows, gene_y):
        cmap = _cmap(LEVEL_CMAPS[li])
        vals = [expr(c, gene) for c in cols]
        mns = [v[0] for v in vals]
        if color_norm == "row":
            m = max(mns) or 1.0
            cvals = [v / m for v in mns]
        else:
            cvals = mns
        ax.text(-1.0, gy, gene, ha="right", va="center", fontsize=6.5,
                color=_solid(LEVEL_CMAPS[li], 0.95))
        for x, (mn, fr), cv in zip(xs, vals, cvals):
            if fr <= min_frac:
                continue
            face = cmap(color_floor + (1 - color_floor) * cv)
            if mode == "band":
                h = row_h * fr                       # raw: height = fraction, no floor
                ax.add_patch(Rectangle((x - band_w / 2, gy - h / 2), band_w, h,
                            facecolor=face, edgecolor="none", zorder=3))
            else:
                ax.scatter([x], [gy], s=8 + fr * max_dot, facecolor=face,
                           edgecolor="none", zorder=3)

    # ---------- legends ----------
    lx, bw = max(xs) + 1.7, 0.95
    ly = y_top
    lab = "fraction (band height)" if mode == "band" else "fraction"
    ax.text(lx, ly, lab, fontsize=8.5, va="top", ha="left")
    if mode == "band":
        ly -= 0.6
        for fr in [0.25, 0.5, 1.0]:
            h = row_h * fr
            ax.add_patch(Rectangle((lx, ly - h / 2), band_w, h, facecolor="0.5", edgecolor="none"))
            ax.text(lx + band_w + 0.2, ly, f"{int(fr * 100)}%", fontsize=8, va="center")
            ly -= 0.6
    else:
        ly -= 0.85
        for fr in [0.25, 0.5, 1.0]:
            ax.scatter([lx + 0.28], [ly], s=8 + fr * max_dot, facecolor="0.5", edgecolor="none")
            ax.text(lx + 0.95, ly, f"{int(fr * 100)}%", fontsize=8, va="center"); ly -= 0.85
    ly -= 0.55
    ax.text(lx, ly, "norm. expression", fontsize=8.5, va="top", ha="left"); ly -= 0.75
    nseg, cbar_gap = 24, 0.62
    for bi, name in enumerate([g0, g1, g2]):
        cy = ly - bi * cbar_gap
        cmap = _cmap(LEVEL_CMAPS[bi])
        for s in range(nseg):
            t = s / (nseg - 1)
            ax.add_patch(Rectangle((lx + s * bw / nseg, cy), bw / nseg, 0.24,
                        facecolor=cmap(color_floor + (1 - color_floor) * t), edgecolor="none", zorder=3))
        ax.add_patch(Rectangle((lx, cy), bw, 0.24, fill=False, edgecolor="black", lw=0.5, zorder=4))
        ax.text(lx + bw + 0.18, cy + 0.12, name, fontsize=8,
                color=_solid(LEVEL_CMAPS[bi], 0.9), va="center")
    cy_last = ly - 2 * cbar_gap
    ax.text(lx, cy_last - 0.12, "0", fontsize=7.5, ha="left", va="top")
    ax.text(lx + bw, cy_last - 0.12, "1", fontsize=7.5, ha="right", va="top")

    # ---------- title & frame ----------
    if full_map:
        title = "all-paths hierarchy marker map"
    else:
        title = f"path {ref_icls} markers  (rows colored by level)"
    ax.text(-1.0, y_top + 0.4, title, fontsize=11, ha="left", va="bottom")
    ax.set_xlim(-3.4, lx + bw + 1.6)
    ax.set_ylim(min(y_bot - 0.6, cy_last - 0.4), y_top + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()

    if save:
        fig.savefig(save, bbox_inches="tight", format="pdf", dpi=300)
    plt.close(fig)
    return fig, ax
