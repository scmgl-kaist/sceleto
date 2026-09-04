"""Hierarchy path-markers dotplot.

A ``compare_markers``-style dotplot with a hierarchy tree on top:

  - TOP: a dendrogram of the 3 leiden resolutions over ALL icls leaves
    (level 0 -> level 1 -> level 2 -> path circles). Sibling groups share a
    bracket; columns are ordered by the hierarchy so the tree is planar.
  - BOTTOM: dot matrix.
      rows   = the reference path's per-level top-N markers (like
               ``HierarchyRun.compare_markers``); duplicates are KEPT and each
               gene's row label is colored by its origin level
               (level 0 red / level 1 green / level 2 black).
      cols   = every icls leaf, aligned under the tree.
      dot color = normalized mean expression via the gene's own level colormap
      dot size  = fraction of expressing cells

Compared to :meth:`HierarchyRun.compare_markers` (a binary presence heatmap of
the de-duplicated marker union), this shows *expression* of every level's
markers across the whole atlas, so you can see whether a level's markers are
leaf-specific or shared.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

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


def path_markers_dotplot(
    hr,
    ref_icls: str,
    *,
    n_markers: Optional[int] = None,
    dedup: bool = False,
    columns: Union[str, Sequence[str]] = "all",
    color_norm: str = "global",
    color_floor: float = 0.30,
    max_dot: float = 170.0,
    min_frac: float = 0.0,
    highlight_ref: bool = True,
    show_tree: bool = True,
    row_h: float = 0.60,
    figsize=None,
    save: Optional[str] = None,
):
    """Draw the hierarchy path-markers dotplot for one reference ``icls`` path.

    Parameters
    ----------
    hr
        A :class:`~sceleto.markers.HierarchyRun` (from ``scl.markers.hierarchy``).
    ref_icls
        The reference icls id whose per-level top-N markers become the rows.
    n_markers
        Top-N markers gathered per level. ``None`` (default) uses the run's
        ``n_top_markers``.
    dedup
        If ``True`` collapse the 3 levels' markers into a set (like
        ``compare_markers``). Default ``False`` keeps duplicates so a gene that
        is a top marker at two levels appears as two differently-colored rows.
    columns
        ``"all"`` (default) uses every icls leaf; or pass an explicit list of
        icls ids. Columns are always re-ordered by the hierarchy so the tree is
        planar.
    color_norm
        ``"global"`` (default) uses the stored per-gene ``mean_norm`` (0..1 over
        all leaves at the finest level); ``"row"`` re-normalizes each gene by its
        max among the shown columns (max in-plot contrast).
    color_floor, max_dot, min_frac
        Dot color floor (so low values stay visible), max dot area, and the
        fraction threshold below which no dot is drawn.
    highlight_ref
        Shade the reference icls column.
    show_tree
        Draw the hierarchy dendrogram on top (default ``True``).
    row_h
        Vertical spacing between gene rows.
    figsize
        Manual ``(w, h)``; auto-sized otherwise.
    save
        If given, save to this path (PDF, dpi=300).

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

    levels = hr.levels                              # [g0, g1, g2]
    g0, g1, g2 = levels
    df = hr.icls_path_df.set_index("icls")
    full = hr.full_gene_lists
    if n_markers is None:
        n_markers = hr.params["n_top_markers"]

    ref_icls = str(ref_icls)
    if ref_icls not in df.index:
        raise ValueError(f"path_markers_dotplot: icls {ref_icls!r} not found.")
    ref = df.loc[ref_icls]
    ref_clu = [ref[g0], ref[g1], ref[g2]]

    def clu_at(icls, lvl):
        return df.loc[icls, lvl].split("@", 1)[1]

    # --- rows: per-level top-N markers of the reference path -----------------
    rows: List = []
    seen = set()
    for li, lvl_str in enumerate(ref_clu):
        for g in full[lvl_str][:n_markers]:
            if dedup and g in seen:
                continue
            seen.add(g)
            rows.append((g, li))

    # --- columns: all leaves (or a subset), hierarchy-ordered ---------------
    if isinstance(columns, str) and columns == "all":
        cols = list(df.index)
    else:
        cols = [str(c) for c in columns]
    cols = sorted(cols, key=lambda c: (int(clu_at(c, g0)), int(clu_at(c, g1)),
                                       int(clu_at(c, g2))))

    # --- expression at the finest (leaf) resolution -------------------------
    ctx2 = hr.contexts[g2]
    gidx = {g: i for i, g in enumerate(ctx2.genes)}
    leaf_row = {icls: ctx2.group_to_idx.get(clu_at(icls, g2)) for icls in cols}

    def expr(icls, gene):
        r, c = leaf_row[icls], gidx.get(gene)
        if r is None or c is None:
            return 0.0, 0.0
        return float(ctx2.mean_norm[r, c]), float(ctx2.frac_expr[r, c])

    # ----- geometry ---------------------------------------------------------
    ncol, ngene = len(cols), len(rows)
    xs = list(range(ncol))
    gene_y = [-(i * row_h) for i in range(ngene)]
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

    if highlight_ref and ref_icls in cols:
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
            ax.plot([min(children_x), max(children_x)], [bar_y, bar_y],
                    color=lc, lw=lw, zorder=1)
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

        for yy, name, cmapname in [(Y_L0, g0, LEVEL_CMAPS[0]),
                                   (Y_L1, g1, LEVEL_CMAPS[1]),
                                   (Y_L2, g2, LEVEL_CMAPS[2])]:
            ax.text(-1.0, yy, name, ha="right", va="center", fontsize=8.5,
                    color=_solid(cmapname, 0.9))
        ax.text(-1.0, Y_PATH, "path", ha="right", va="center", fontsize=8.5,
                color="black")

    # ---------- path (icls) circles ----------
    for x, icls in zip(xs, cols):
        ax.add_patch(Circle((x, Y_PATH), 0.33, facecolor="#c9c9c9",
                    edgecolor="black", lw=1.8 if icls == ref_icls else 0.8,
                    zorder=3))
        ax.text(x, Y_PATH, icls, ha="center", va="center", fontsize=7.5,
                color="black",
                fontweight="bold" if icls == ref_icls else "normal", zorder=4)

    # ---------- dot matrix ----------
    for (gene, li), gy in zip(rows, gene_y):
        cmap = _cmap(LEVEL_CMAPS[li])
        vals = [expr(icls, gene) for icls in cols]
        mns = [v[0] for v in vals]
        if color_norm == "row":
            m = max(mns) or 1.0
            cvals = [v / m for v in mns]
        else:
            cvals = mns
        ax.text(-1.0, gy, gene, ha="right", va="center", fontsize=8,
                color=_solid(LEVEL_CMAPS[li], 0.95))
        for x, (mn, fr), cv in zip(xs, vals, cvals):
            if fr <= min_frac:
                continue
            face = cmap(color_floor + (1 - color_floor) * cv)
            ax.scatter([x], [gy], s=8 + fr * max_dot, facecolor=face,
                       edgecolor="none", zorder=3)

    # ---------- legends (right side) ----------
    lx, bw = max(xs) + 1.7, 0.95
    ly = y_top
    ax.text(lx, ly, "fraction", fontsize=8.5, va="top", ha="left")
    ly -= 0.85
    for fr in [0.25, 0.5, 1.0]:
        ax.scatter([lx + 0.28], [ly], s=8 + fr * max_dot, facecolor="0.5",
                   edgecolor="none")
        ax.text(lx + 0.95, ly, f"{int(fr * 100)}%", fontsize=8, va="center")
        ly -= 0.85
    ly -= 0.55
    ax.text(lx, ly, "norm. expression", fontsize=8.5, va="top", ha="left")
    ly -= 0.75
    nseg, cbar_gap = 24, 0.62
    for bi, name in enumerate([g0, g1, g2]):
        cy = ly - bi * cbar_gap
        cmap = _cmap(LEVEL_CMAPS[bi])
        for s in range(nseg):
            t = s / (nseg - 1)
            ax.add_patch(Rectangle((lx + s * bw / nseg, cy), bw / nseg, 0.24,
                        facecolor=cmap(color_floor + (1 - color_floor) * t),
                        edgecolor="none", zorder=3))
        ax.add_patch(Rectangle((lx, cy), bw, 0.24, fill=False, edgecolor="black",
                    lw=0.5, zorder=4))
        ax.text(lx + bw + 0.18, cy + 0.12, name, fontsize=8,
                color=_solid(LEVEL_CMAPS[bi], 0.9), va="center")
    cy_last = ly - 2 * cbar_gap
    ax.text(lx, cy_last - 0.12, "0", fontsize=7.5, ha="left", va="top")
    ax.text(lx + bw, cy_last - 0.12, "1", fontsize=7.5, ha="right", va="top")

    # cosmetics
    ax.text(-1.0, y_top + 0.4, f"path {ref_icls} markers  (rows colored by level)",
            fontsize=11, ha="left", va="bottom")
    ax.set_xlim(-3.4, lx + bw + 1.6)
    ax.set_ylim(min(y_bot - 0.6, cy_last - 0.4), y_top + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()

    if save:
        fig.savefig(save, bbox_inches="tight", format="pdf", dpi=300)
    plt.close(fig)   # suppress inline auto-show; caller displays via `fig`
    return fig, ax


def hierarchy_markers_dotplot(
    hr,
    *,
    n_markers: Optional[int] = None,
    order: str = "dfs",
    style: str = "band",
    color_norm: str = "global",
    color_floor: float = 0.30,
    max_dot: float = 170.0,
    min_frac: float = 0.0,
    show_tree: bool = True,
    row_h: Optional[float] = None,
    band_gap: Optional[float] = None,
    figsize=None,
    save: Optional[str] = None,
):
    """FULL hierarchy marker map: EVERY node's top-N markers (rows) vs all leaves.

    Raw generalization of :func:`path_markers_dotplot` from one path to the whole
    hierarchy. Rows = each node's top-N markers (level0 red / level1 green /
    level2 black); columns = every icls leaf, tree on top. No dedup, no anchor.

    ``order`` controls the row order:

    - ``"dfs"`` (default): depth-first pre-order — each level0 cluster immediately
      followed by its level1 children and their level2 children, so rows are
      grouped by lineage/branch (levels interleave; gap separates top branches).
    - ``"bfs"``: breadth-first — all level0 clusters, then all level1, then all
      level2. Three level-bands (each band ~ a block/diagonal).

    ``style`` controls the per-cell glyph:

    - ``"band"`` (default): a filled rectangle spanning the column, whose HEIGHT
      encodes the fraction of expressing cells (color still encodes expression).
      Rows can be thin (``row_h`` defaults to 0.18), so the map is far shorter
      than the dot version. Raw — no minimum thickness, fraction maps linearly.
    - ``"dot"``: a scatter dot whose AREA encodes the fraction (``row_h`` 0.60).

    NOTE: this stacks all nodes' markers, so the figure is very tall (roughly
    ``(n_level0 + n_level1 + n_level2) * n_markers`` rows). Best saved as a large
    PDF / used as an atlas overview; keep ``n_markers`` small.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    def _cmap(name):
        return plt.get_cmap(name)

    def _solid(name, v=0.85):
        return _cmap(name)(v)

    if style not in ("band", "dot"):
        raise ValueError("style must be 'band' or 'dot'.")

    levels = hr.levels
    g0, g1, g2 = levels
    df = hr.icls_path_df.set_index("icls")
    full = hr.full_gene_lists
    if n_markers is None:
        n_markers = hr.params["n_top_markers"]
    # band packs rows tighter than dots (dots need room for their area)
    if row_h is None:
        row_h = 0.18 if style == "band" else 0.60
    if band_gap is None:
        band_gap = 0.35 if style == "band" else 0.55
    band_w = 0.92                                # column-fill width for band cells

    def clu_at(icls, lvl):
        return df.loc[icls, lvl].split("@", 1)[1]

    cols = sorted(df.index, key=lambda c: (int(clu_at(c, g0)), int(clu_at(c, g1)),
                                           int(clu_at(c, g2))))

    # rows: each node's top-N markers, ordered BFS (level bands) or DFS (branches)
    rows, gaps = [], []                         # (gene, level_index) ; gap_before flag

    def _emit(level_i, cluster_id, gap):
        first = True
        for gene in full[f"{levels[level_i]}@{cluster_id}"][:n_markers]:
            rows.append((gene, level_i))
            gaps.append(gap if first else False)
            first = False

    if order == "dfs":
        prev = (None, None, None)
        for icls in cols:
            a, b, c = clu_at(icls, g0), clu_at(icls, g1), clu_at(icls, g2)
            if a != prev[0]:
                _emit(0, a, gap=len(rows) > 0)          # gap before each new branch
            if (a, b) != prev[:2]:
                _emit(1, b, gap=False)
            if (a, b, c) != prev:
                _emit(2, c, gap=False)
            prev = (a, b, c)
    elif order == "bfs":
        for li, g in enumerate(levels):
            seen_clu = []
            for icls in cols:
                k = clu_at(icls, g)
                if k not in seen_clu:
                    seen_clu.append(k)
            for ci, k in enumerate(seen_clu):
                _emit(li, k, gap=(li > 0 and ci == 0))  # gap before each new band
    else:
        raise ValueError("order must be 'bfs' or 'dfs'.")

    ctx2 = hr.contexts[g2]
    gidx = {gg: i for i, gg in enumerate(ctx2.genes)}
    leaf_row = {icls: ctx2.group_to_idx.get(clu_at(icls, g2)) for icls in cols}

    def expr(icls, gene):
        r, c = leaf_row[icls], gidx.get(gene)
        if r is None or c is None:
            return 0.0, 0.0
        return float(ctx2.mean_norm[r, c]), float(ctx2.frac_expr[r, c])

    # ----- geometry (gaps at band/branch boundaries per `gaps`) -----
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

    for x, icls in zip(xs, cols):
        ax.add_patch(Circle((x, Y_PATH), 0.33, facecolor="#c9c9c9", edgecolor="black",
                    lw=0.8, zorder=3))
        ax.text(x, Y_PATH, icls, ha="center", va="center", fontsize=7.5, color="black", zorder=4)

    for (gene, li), gy in zip(rows, gene_y):
        cmap = _cmap(LEVEL_CMAPS[li])
        vals = [expr(icls, gene) for icls in cols]
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
            if style == "band":
                # rectangle spanning the column; height = fraction * row_h (raw,
                # no floor), centered on the row. color = expression.
                h = row_h * fr
                ax.add_patch(Rectangle((x - band_w / 2, gy - h / 2), band_w, h,
                            facecolor=face, edgecolor="none", zorder=3))
            else:
                ax.scatter([x], [gy], s=8 + fr * max_dot, facecolor=face,
                           edgecolor="none", zorder=3)

    lx, bw = max(xs) + 1.7, 0.95
    ly = y_top
    lab = "fraction (band height)" if style == "band" else "fraction"
    ax.text(lx, ly, lab, fontsize=8.5, va="top", ha="left")
    if style == "band":
        ly -= 0.6
        for fr in [0.25, 0.5, 1.0]:
            h = row_h * fr
            ax.add_patch(Rectangle((lx, ly - h / 2), band_w, h, facecolor="0.5",
                        edgecolor="none"))
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

    ax.text(-1.0, y_top + 0.4, "all-paths hierarchy marker map", fontsize=11, ha="left", va="bottom")
    ax.set_xlim(-3.4, lx + bw + 1.6)
    ax.set_ylim(min(y_bot - 0.6, cy_last - 0.4), y_top + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()

    if save:
        fig.savefig(save, bbox_inches="tight", format="pdf", dpi=300)
    plt.close(fig)
    return fig, ax
