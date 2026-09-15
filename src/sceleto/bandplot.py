"""Band plot: a compact dotplot alternative where each cell is a *band*.

Generalizes the "band" rendering of
:meth:`sceleto.markers.HierarchyRun.marker_map` into a standalone plot with the
same ``(adata, var_names, groupby)`` signature as :func:`sceleto.dotplot`.

Each (gene, group) cell is a filled rectangle:

- its **extent along the gene axis** (height by default) encodes the *fraction
  of cells expressing* the gene — like a partly-filled heatmap cell,
- its **color** encodes *mean expression* (raw log1p mean by default, or a
  per-gene max-scaled 0..1 with ``max_scale=True``).

Because bands pack tight (a cell simply fills part of its row), a band plot
stays much shorter than the equivalent dotplot while keeping both the fraction
and the expression readouts.

Default orientation matches ``marker_map``: **genes on the y-axis, groups on the
x-axis**.  Pass ``swap_axes=True`` for the ``scanpy.pl.dotplot`` orientation
(genes on x, groups on y); then the fraction is encoded as band *width*.

Usage
-----
>>> import sceleto as scl
>>> scl.bandplot(adata, ['CD3D', 'CD8A', 'MS4A1'], 'leiden')
>>> scl.bandplot(adata, {'T': ['CD3D', 'CD8A'], 'B': ['MS4A1']}, 'leiden')
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse

from .dotplot import _resolve_var_names, _check_log1p_normalized


def _levels(series: pd.Series, requested):
    """Ordered string levels of *series* (categorical order if available)."""
    if requested is not None:
        return [str(x) for x in requested]
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [str(x) for x in series.cat.categories]
    return [str(x) for x in pd.unique(series.astype(str))]


def _truncate_cmap(cmap, floor: float, n: int = 256):
    """Return a colormap that starts at ``cmap(floor)`` instead of ``cmap(0)``.

    A small ``floor`` keeps a low-but-present cell from rendering near-white
    (invisible) while the band's extent already carries the fraction signal.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    base = plt.get_cmap(cmap)
    if floor <= 0:
        return base
    return LinearSegmentedColormap.from_list(
        f"{base.name}_floor", base(np.linspace(floor, 1.0, n))
    )


def bandplot(
    adata,
    var_names: Union[Sequence[str], Mapping[str, Sequence]],
    groupby: str,
    *,
    max_scale: bool = False,
    groups: Optional[Sequence[str]] = None,
    swap_axes: bool = False,
    use_raw: bool = True,
    cmap: str = "OrRd",
    color_floor: float = 0.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    band_w: float = 0.85,
    band_max: float = 0.6,
    min_frac: float = 0.0,
    group_gap: float = 0.0,
    group_rotation: Optional[float] = None,
    label_genes: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    save: Optional[str] = None,
    show: bool = True,
):
    """Band plot — a compact dotplot alternative (see module docstring).

    Parameters
    ----------
    adata
        AnnData with log1p-normalized expression.
    var_names
        Gene list, or ``{group_label: [gene, ...]}`` mapping.  A mapping splits
        the gene axis into bracketed blocks (like ``scanpy.pl.dotplot``).  Tuple
        entries ``(gene, score)`` are accepted; the score is ignored.
    groupby
        Column in ``adata.obs`` to group cells by (the group axis).
    max_scale
        Color scaling of mean expression.  ``False`` (default) → raw log1p group
        mean with automatic ``vmin``/``vmax``.  ``True`` → per-gene
        max-normalized group mean (``group_mean / max_group(group_mean)``), so
        ``vmax=1`` is always the highest-expressing group for that gene.
    groups
        Ordered subset of ``groupby`` levels to show.  ``None`` shows all
        (categorical order if available).
    swap_axes
        ``False`` (default): genes on y-axis, groups on x-axis (``marker_map``
        orientation); fraction = band **height**.  ``True``: genes on x-axis,
        groups on y-axis (``scanpy.pl.dotplot`` orientation); fraction = band
        **width**.
    use_raw
        Read expression from ``adata.raw.X`` (default) else ``adata.X``.
    cmap
        Matplotlib colormap for mean expression (default ``"OrRd"``).
    color_floor
        Start the colormap at this fraction (0..1) instead of 0, so faint-but-
        present cells stay visible.  Default ``0.0`` (plain colormap).
    vmin, vmax
        Color limits for raw mean (``max_scale=False``).  ``None`` auto-selects
        ``0`` .. data max.  Ignored when ``max_scale=True`` (fixed 0..1).
    band_w
        Fixed band extent (0..1) along the *group* axis — the band's width in the
        default orientation.
    band_max
        Band extent (0..1) along the *gene* axis at fraction = 1 — how tall a
        fully-expressed band is relative to its row spacing (default ``0.6`` →
        flat wide bars with small gaps; lower toward ``0`` for thinner bars,
        raise toward ``1`` for chunkier cells that nearly touch).
    min_frac
        Fractions at or below this draw no band (default ``0.0``).
    group_gap
        Extra spacing inserted between bracketed gene blocks (mapping input).
        Default ``0.0`` (blocks are contiguous; the brackets alone separate them).
    group_rotation
        Rotation (degrees) of the group-axis tick labels.  ``None`` auto-picks
        (0 for short labels, 90 for long ones).
    label_genes
        Draw per-gene labels along the gene axis (default ``True``).
    figsize
        Manual ``(width, height)`` in inches; auto-sized otherwise.
    save
        Path to save the figure (PDF, dpi=300).
    show
        Call ``plt.show()``.

    Returns
    -------
    ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    # ── expression source & genes ─────────────────────────────────────
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True but adata.raw is None.")
        src, src_names = adata.raw, list(adata.raw.var_names)
    else:
        src, src_names = adata, list(adata.var_names)

    available = set(src_names)
    var_group_dict, flat_genes = _resolve_var_names(var_names, available)
    if not flat_genes:
        raise ValueError("sceleto.bandplot: none of the provided genes are present.")
    # unique genes, first-occurrence order; keep mapping structure for brackets
    genes = list(dict.fromkeys(flat_genes))

    if groupby not in adata.obs:
        raise ValueError(f"sceleto.bandplot: obs column {groupby!r} not found.")
    group_levels = _levels(adata.obs[groupby], groups)

    # ── per (group, gene) mean & fraction ─────────────────────────────
    gi = np.array([src_names.index(g) for g in genes])
    X = src.X[:, gi]
    _check_log1p_normalized(X, "adata.raw.X" if use_raw else "adata.X")
    labels = adata.obs[groupby].astype(str).to_numpy()

    K, G = len(group_levels), len(genes)
    mean_mat = np.zeros((K, G), dtype=float)
    frac_mat = np.zeros((K, G), dtype=float)
    for j, grp in enumerate(group_levels):
        m = labels == grp
        if not m.any():
            continue
        sub = X[m]
        if sparse.issparse(sub):
            mean_mat[j] = np.asarray(sub.mean(0)).ravel()
            frac_mat[j] = np.asarray((sub > 0).sum(0)).ravel() / sub.shape[0]
        else:
            sub = np.asarray(sub)
            mean_mat[j] = sub.mean(0)
            frac_mat[j] = (sub > 0).mean(0)

    # ── color scaling ─────────────────────────────────────────────────
    if max_scale:
        gene_max = mean_mat.max(axis=0)
        gene_max[gene_max == 0] = 1.0
        color_mat = mean_mat / gene_max[np.newaxis, :]
        cvmin, cvmax = 0.0, 1.0
        cbar_title = "Max-scaled\nmean"
    else:
        color_mat = mean_mat
        cvmin = 0.0 if vmin is None else float(vmin)
        cvmax = (float(mean_mat.max()) if mean_mat.size else 1.0) if vmax is None else float(vmax)
        if cvmax <= cvmin:
            cvmax = cvmin + 1e-9
        cbar_title = "Mean\nexpression"

    norm = Normalize(vmin=cvmin, vmax=cvmax)
    eff_cmap = _truncate_cmap(cmap, color_floor)

    # ── gene-axis positions (with bracket-block gaps) ─────────────────
    if var_group_dict is not None:
        # keep only blocks whose (unique) genes survived, in mapping order
        blocks = []
        placed = set()
        for label, gs in var_group_dict.items():
            members = [g for g in dict.fromkeys(gs) if g in available and g not in placed]
            if members:
                blocks.append((str(label), members))
                placed.update(members)
    else:
        blocks = [(None, genes)]

    gpos, gene_order, brackets = [], [], []
    pos = 0.0
    for bi, (label, members) in enumerate(blocks):
        if bi > 0:
            pos += group_gap
        start = pos
        for g in members:
            gene_order.append(g)
            gpos.append(pos)
            pos += 1.0
        if label is not None:
            brackets.append((label, start, pos - 1.0))
    gpos = np.asarray(gpos)
    gcol = {g: k for k, g in enumerate(genes)}          # gene -> column in matrices
    order_idx = [gcol[g] for g in gene_order]           # matrix cols in draw order
    spos = np.arange(K, dtype=float)                    # group-axis positions

    # ── figure ─────────────────────────────────────────────────────────
    if group_rotation is None:
        group_rotation = 90 if max((len(str(g)) for g in group_levels), default=0) > 3 else 0

    GENE_FS, GRP_FS, BRACKET_FS = 10.5, 11.5, 10.5
    LEG_W_IN = 2.2                       # inches reserved for the right-hand legend
    _CHAR_IN = 0.095                     # rough inches/char, only for the initial size
    max_gene_len = max((len(str(g)) for g in gene_order), default=1)
    max_grp_len = max((len(str(g)) for g in group_levels), default=1)

    # left labels are genes (default) or groups (swapped); estimate their width so
    # the figure is wide enough for the grid, the labels+brackets, and the legend.
    if not swap_axes:
        left_len, n_cols, n_rows = (max_gene_len if label_genes else 0), K, len(gene_order)
    else:
        left_len, n_cols, n_rows = max_grp_len, len(gene_order), K
    m_left_est = _CHAR_IN * left_len + 0.14
    if brackets and not swap_axes:
        m_left_est += 0.16 + BRACKET_FS / 72.0 * 1.6 + 0.10

    if figsize is None:
        grid_w = 0.32 * n_cols + 0.4
        row_h = 0.30 if swap_axes else 0.22
        figsize = (m_left_est + grid_w + LEG_W_IN, max(3.2, row_h * n_rows + 1.2))
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_position([0, 0, 1, 1])        # axes fills the canvas → du_x = xrange/fw exactly

    def _face(cval):
        return eff_cmap(norm(cval))

    # ── draw bands ─────────────────────────────────────────────────────
    # Band extent: `band_max*frac` along the GENE axis, `band_w` along the GROUP
    # axis.  In the default orientation (genes=y) that is height×width; when
    # swapped (genes=x) it is width×height.
    for gi_draw, col in enumerate(order_idx):          # gene axis index
        gy = gpos[gi_draw]
        for j in range(K):                             # group axis index
            fr = frac_mat[j, col]
            if fr <= min_frac:
                continue
            face = _face(color_mat[j, col])
            gene_ext = band_max * fr
            if not swap_axes:  # genes on y (top→down), groups on x
                cx, cy = spos[j], -gy
                ax.add_patch(Rectangle((cx - band_w / 2, cy - gene_ext / 2),
                                       band_w, gene_ext, facecolor=face,
                                       edgecolor="none", zorder=3))
            else:              # genes on x, groups on y (top→down)
                cx, cy = gy, -spos[j]
                ax.add_patch(Rectangle((cx - gene_ext / 2, cy - band_w / 2),
                                       gene_ext, band_w, facecolor=face,
                                       edgecolor="none", zorder=3))

    # ── manual labels (axis off; everything is placed in data coords and the
    #    limits host the legend, like marker_map) ──
    left_texts, bottom_texts = [], []
    if not swap_axes:
        # genes on y (left labels), groups on x (bottom labels)
        base_x0, base_x1 = -0.7, K - 0.3
        base_y0, base_y1 = -gpos.max() - 0.6, 0.6
        if label_genes:
            for g, gy in zip(gene_order, gpos):
                left_texts.append(ax.text(-0.65, -gy, g, ha="right", va="center",
                                          fontsize=GENE_FS))
        for j, grp in enumerate(group_levels):
            bottom_texts.append(ax.text(spos[j], base_y0 + 0.15, grp, ha="center",
                                        va="top", rotation=group_rotation, fontsize=GRP_FS))
    else:
        # genes on x (bottom labels), groups on y (left labels)
        base_x0, base_x1 = -0.7, gpos.max() + 0.6
        base_y0, base_y1 = -spos.max() - 0.7, 0.7
        for j, grp in enumerate(group_levels):
            left_texts.append(ax.text(-0.65, -spos[j], grp, ha="right", va="center",
                                      fontsize=GRP_FS))
        if label_genes:
            for g, gx in zip(gene_order, gpos):
                bottom_texts.append(ax.text(gx, base_y0 + 0.15, g, ha="center",
                                            va="top", rotation=90, fontsize=GENE_FS))
    ax.axis("off")

    # Measure the label footprints in INCHES. A glyph's pixel size is fixed by the
    # font, independent of the axis limits, so this is stable no matter how the
    # grid is scaled afterwards (the earlier data-space measurement broke because
    # the legend fixpoint changed the x scale after measuring). Fall back to a
    # char-count estimate if the renderer is unavailable.
    fw, fh = fig.get_size_inches()
    dpi = fig.get_dpi()
    ax.set_xlim(base_x0 - 3.0, base_x1 + 5.0)
    ax.set_ylim(base_y0 - 3.0, base_y1 + 2.0)
    left_w_in = _CHAR_IN * left_len
    bottom_h_in = 0.16 if (group_rotation == 0 and not swap_axes) else \
        _CHAR_IN * (max_grp_len if not swap_axes else max_gene_len)
    try:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        if left_texts:
            left_w_in = max(t.get_window_extent(r).width for t in left_texts) / dpi
        if bottom_texts:
            bottom_h_in = max(t.get_window_extent(r).height for t in bottom_texts) / dpi
    except Exception:  # pragma: no cover - keep the char-count estimates
        pass

    # ── solve the data<->inch scale per axis (aspect="auto", x and y differ) ──
    # x: reserve a fixed left margin (labels + optional vertical brackets) and a
    # fixed LEG_W_IN legend column; a fixpoint gives du_x = data-units per inch.
    has_left_brackets = bool(brackets) and not swap_axes
    m_left_in = left_w_in + 0.14
    if has_left_brackets:
        m_left_in += 0.16 + BRACKET_FS / 72.0 * 1.6 + 0.10
    du_x = (base_x1 + 0.65) / max(fw - LEG_W_IN - m_left_in, 0.6)
    # y: reserve the bottom label band (+ a top bracket band when swapped)
    top_in = 0.55 if (brackets and swap_axes) else 0.12
    du_y = (base_y1 - base_y0) / max(fh - top_in - bottom_h_in - 0.14, 0.6)

    def IX(inch):
        return inch * du_x

    def IY(inch):
        return inch * du_y

    lim_left = -0.65 - m_left_in * du_x
    lim_right = base_x1 + LEG_W_IN * du_x
    lim_bottom = base_y0 - (bottom_h_in + 0.14) * du_y
    lim_top = base_y1 + top_in * du_y

    # ── brackets for gene blocks (mapping input): one shared line position so
    #    the brackets form a single column (never a staircase) ──
    if has_left_brackets:                     # vertical brackets, left of gene labels
        edge = -0.65 - left_w_in * du_x       # measured gene-label left edge (data-x)
        bx = edge - IX(0.16)
        for label, lo, hi in brackets:
            mid = (lo + hi) / 2.0
            ax.plot([bx, bx], [-hi - 0.35, -lo + 0.35], color="0.4", lw=1.0, zorder=2)
            ax.text(bx - IX(0.07), -mid, label, rotation=90, ha="right",
                    va="center", fontsize=BRACKET_FS)
    elif brackets:                            # horizontal brackets, above the plot
        by = base_y1 + IY(0.26)
        for label, lo, hi in brackets:
            mid = (lo + hi) / 2.0
            ax.plot([lo - 0.35, hi + 0.35], [by, by], color="0.4", lw=1.0, zorder=2)
            ax.text(mid, by + IY(0.06), label, ha="center", va="bottom", fontsize=BRACKET_FS)

    # ── legend: fraction key + expression colorbar (sized in inches via IX/IY) ──
    unit = "height" if not swap_axes else "width"
    lx = base_x1 + IX(0.45)
    ly = base_y1
    ax.text(lx, ly, f"fraction\n(band {unit})", fontsize=9, va="top", ha="left")
    ly -= IY(0.6)
    for fr in (0.25, 0.5, 1.0):
        ext = band_max * fr                       # matches the plot bands' extent
        if not swap_axes:
            ax.add_patch(Rectangle((lx, ly - ext / 2), IX(0.34), ext,
                                   facecolor="0.55", edgecolor="none"))
        else:
            ax.add_patch(Rectangle((lx, ly - IY(0.09)), ext, IY(0.18),
                                   facecolor="0.55", edgecolor="none"))
        ax.text(lx + IX(0.62), ly, f"{int(fr * 100)}%", fontsize=8.5, va="center")
        ly -= IY(0.34)

    # manual vertical colorbar (respects color_floor)
    ly -= IY(0.30)
    ax.text(lx, ly, cbar_title, fontsize=9, va="top", ha="left")
    ly -= IY(0.55)
    cbar_h, cbar_w = IY(1.1), IX(0.34)
    cb_top = ly
    grad = np.linspace(0.0, 1.0, 256).reshape(-1, 1)   # 0 at bottom, 1 at top
    ax.imshow(grad, extent=(lx, lx + cbar_w, cb_top - cbar_h, cb_top),
              origin="lower", aspect="auto", cmap=eff_cmap, interpolation="bilinear",
              zorder=3)
    ax.add_patch(Rectangle((lx, cb_top - cbar_h), cbar_w, cbar_h, fill=False,
                           edgecolor="0.3", lw=0.6))
    ax.text(lx + cbar_w + IX(0.1), cb_top, f"{cvmax:.2g}", fontsize=8,
            va="center", ha="left")
    ax.text(lx + cbar_w + IX(0.1), cb_top - cbar_h, f"{cvmin:.2g}", fontsize=8,
            va="center", ha="left")

    # final limits: everything is inside these bounds (axes fills the canvas, so
    # no tight_layout); the legend column width is already baked into lim_right.
    lim_bottom = min(lim_bottom, cb_top - cbar_h - IY(0.2))
    ax.set_xlim(lim_left, lim_right)
    ax.set_ylim(lim_bottom, lim_top)
    ax.set_aspect("auto")

    if save:
        fig.savefig(save, bbox_inches="tight", format="pdf", dpi=300)
    if show:
        plt.show()
    return fig, ax
