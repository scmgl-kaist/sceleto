"""General-purpose dotplot for gene expression across groups.

Usage
-----
>>> import sceleto as sl
>>> sl.dotplot(adata, ['CD3D', 'CD8A', 'MS4A1'], 'leiden')

With marker results:
>>> mk = sl.markers.simple(adata, 'leiden')
>>> sl.dotplot(adata, mk.markers, 'leiden')
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.colorbar
import matplotlib.gridspec as gridspec
from scipy import sparse


# ── Data computation ────────────────────────────────────────────────


def _compute_group_stats(
    adata,
    groupby: str,
    genes: List[str],
    *,
    use_raw: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Compute mean expression and fraction expressing per (group, gene).

    Returns (mean, frac_expr, groups, genes) where mean and frac_expr
    are arrays of shape (n_groups, n_genes).
    """
    from ._expr import resolve_expression
    X, var_names, _ = resolve_expression(adata)

    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr()

    # resolve gene indices
    var_to_idx = {g: i for i, g in enumerate(var_names)}
    valid_genes = [g for g in genes if g in var_to_idx]
    gene_cols = np.array([var_to_idx[g] for g in valid_genes])
    X_sub = X[:, gene_cols]

    groups = sorted(adata.obs[groupby].astype(str).unique())
    obs_labels = adata.obs[groupby].astype(str).to_numpy()

    n_groups = len(groups)
    n_genes = len(valid_genes)
    mean = np.zeros((n_groups, n_genes), dtype=np.float32)
    frac = np.zeros((n_groups, n_genes), dtype=np.float32)

    for i, g in enumerate(groups):
        mask = obs_labels == g
        n_cells = int(mask.sum())
        if n_cells == 0:
            continue
        X_g = X_sub[mask]
        mean[i] = np.asarray(X_g.mean(axis=0)).ravel()
        frac[i] = X_g.getnnz(axis=0) / n_cells

    return mean, frac, groups, valid_genes


# ── Layout constants (adapted from scanpy BasePlot / DotPlot) ──────

_CELL_HEIGHT = 0.35           # inches per group-axis item
_CELL_WIDTH = 0.35            # inches per gene-axis item
_MIN_FIGURE_HEIGHT = 2.5      # minimum so legends always fit
_LEGENDS_WIDTH = 2.0          # inches for right-side legend panel
_X_PADDING = 0.8              # x-axis padding (dot-center to border)
_Y_PADDING = 1.0              # y-axis padding

_DEFAULT_LARGEST_DOT = 200.0
_DEFAULT_SMALLEST_DOT = 0.0
_DEFAULT_SIZE_EXPONENT = 1.5


# ── Legend helpers (adapted from scanpy DotPlot._plot_legend) ───────


def _plot_size_legend(ax, dot_max, dot_min, largest_dot, smallest_dot,
                      size_exponent):
    """Render fraction-of-cells size legend (scanpy style)."""
    diff = dot_max - dot_min
    if diff > 0.6:
        step = 0.2
    elif diff > 0.3:
        step = 0.1
    else:
        step = 0.05

    size_range = np.arange(dot_max, dot_min, -step)[::-1]
    if dot_min != 0 or dot_max != 1:
        dot_range = dot_max - dot_min
        size_values = (size_range - dot_min) / dot_range
    else:
        size_values = size_range

    sizes = size_values ** size_exponent
    sizes = sizes * (largest_dot - smallest_dot) + smallest_dot

    ax.scatter(
        np.arange(len(sizes)) + 0.5,
        np.repeat(0, len(sizes)),
        s=sizes, color="gray", zorder=100,
    )
    ax.set_xticks(np.arange(len(sizes)) + 0.5)
    labels = [f"{int(np.round(x * 100))}" for x in size_range]
    ax.set_xticklabels(labels, fontsize="small")
    ax.tick_params(axis="y", left=False, labelleft=False, labelright=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(visible=False)

    ymax = ax.get_ylim()[1]
    ax.set_ylim(-1.05 - largest_dot * 0.003, 4)
    ax.set_title("Fraction of cells\nin group (%)", y=ymax + 0.45, size="small")
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin - 0.15, xmax + 0.5)


def _plot_legends(fig, legend_spec, height, dot_max, dot_min,
                  norm, cmap_obj, largest_dot, smallest_dot,
                  size_exponent,
                  show_size_legend=True, show_colorbar=True):
    """Plot size legend + colorbar in the right panel (scanpy style).

    Uses a 4-row sub-gridspec: [spacer | size legend | spacer | colorbar].
    """
    min_h = max(_MIN_FIGURE_HEIGHT, height)

    cbar_h = min_h * 0.08
    size_h = min_h * 0.27
    spacer_h = min_h * 0.15   # tighter gap between size legend & colorbar
    top_h = max(height - size_h - cbar_h - spacer_h, 0)

    leg_gs = legend_spec.subgridspec(
        4, 1, height_ratios=[top_h, size_h, spacer_h, cbar_h],
    )

    # hide the container axis
    container = fig.add_subplot(legend_spec)
    container.set_axis_off()

    # ── size legend ──
    if show_size_legend:
        size_ax = fig.add_subplot(leg_gs[1])
        _plot_size_legend(size_ax, dot_max, dot_min,
                          largest_dot, smallest_dot,
                          size_exponent)

    # ── colorbar (narrower: ~50% of legend panel width) ──
    if show_colorbar:
        cbar_inner = leg_gs[3].subgridspec(
            1, 2, width_ratios=[1, 1], wspace=0,
        )
        cbar_ax = fig.add_subplot(cbar_inner[0, 0])
        mappable = ScalarMappable(norm=norm, cmap=cmap_obj)
        cb = matplotlib.colorbar.Colorbar(
            cbar_ax, mappable=mappable, orientation="horizontal",
        )
        cb.ax.xaxis.set_tick_params(labelsize="small")
        cb.ax.set_title("Mean expression\n(normalized)", fontsize="small")


# ── Main dotplot function ──────────────────────────────────────────


def dotplot(
    adata,
    genes: Union[Sequence[str], Dict[str, list]],
    groupby: str,
    *,
    groups: Optional[Sequence[str]] = None,
    n_top: Optional[int] = None,
    transpose: bool = False,
    use_raw: bool = True,
    min_frac: float = 0.01,
    cmap: str = "OrRd",
    largest_dot: float = _DEFAULT_LARGEST_DOT,
    smallest_dot: float = _DEFAULT_SMALLEST_DOT,
    size_exponent: float = _DEFAULT_SIZE_EXPONENT,
    grid: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    save: Optional[str] = None,
    show: bool = True,
    # deprecated
    figscale: Optional[float] = None,
):
    """Dotplot of gene expression across groups.

    Size encodes fraction of cells expressing the gene.
    Color encodes mean expression, normalized per gene (value / max).

    Layout adapts automatically to the number of genes and groups,
    allocating a fixed physical size per dot cell.

    Parameters
    ----------
    adata
        AnnData object.
    genes
        Gene names as a list, or marker dict ``{group: [(gene, score), ...]}``.
    groupby
        Column in ``adata.obs`` to group cells by.
    groups
        Subset of groups to show.  If ``None``, show all.
    n_top
        When *genes* is a dict, keep only the top-n per group.
    transpose
        If ``True``, genes on x-axis (top) and groups on y-axis.
        Default puts genes on y-axis and groups on x-axis.
    use_raw
        Use ``adata.raw`` for expression values.
    min_frac
        Fractions below this are set to zero (dot hidden).
    cmap
        Matplotlib colormap name for mean expression.
    largest_dot
        Dot size in points² for fraction = 1.0.
    smallest_dot
        Dot size in points² for fraction = 0.0.
    size_exponent
        Exponent applied to fraction before size scaling.  Values > 1
        increase contrast between small and large dots.
    grid
        Show background grid lines.
    figsize
        Manual ``(width, height)`` in inches.  Overrides auto-sizing.
    save
        Path to save figure (PDF, dpi=300).
    show
        Whether to call ``plt.show()``.

    Returns
    -------
    fig, ax
    """
    # back-compat
    if figscale is not None:
        largest_dot = figscale * 400

    # ── 1. Resolve gene input ───────────────────────────────────────
    if isinstance(genes, dict):
        ordered_genes: List[str] = []
        for gk in sorted(genes.keys()):
            items = genes[gk]
            for item in items[:n_top] if n_top else items:
                g = item[0] if isinstance(item, tuple) else str(item)
                ordered_genes.append(g)
    else:
        ordered_genes = list(genes)

    # ── 2. Compute stats (our way) ─────────────────────────────────
    mean, frac, all_groups, valid_genes = _compute_group_stats(
        adata, groupby, ordered_genes, use_raw=use_raw,
    )

    if groups is not None:
        gset = set(groups)
        idx = [i for i, g in enumerate(all_groups) if g in gset]
        all_groups = [all_groups[i] for i in idx]
        mean = mean[idx]
        frac = frac[idx]

    # keep only valid genes, preserve input order
    valid_set = set(valid_genes)
    display_genes = [g for g in ordered_genes if g in valid_set]
    gidx = {g: i for i, g in enumerate(valid_genes)}
    cols = [gidx[g] for g in display_genes]

    # ── 3. Build DataFrames (groups × genes) ───────────────────────
    color_df = pd.DataFrame(mean[:, cols], index=all_groups, columns=display_genes)
    size_df = pd.DataFrame(frac[:, cols], index=all_groups, columns=display_genes)

    # ── 4. Normalize color: value / max per gene ───────────────────
    gmax = color_df.max(axis=0)
    gmax[gmax == 0] = 1
    color_df = color_df / gmax

    # ── 5. Apply min_frac ──────────────────────────────────────────
    size_df[size_df < min_frac] = 0

    # ── 6. Orient for display ──────────────────────────────────────
    #   default  : y = genes, x = groups  →  transpose the DFs
    #   transpose: y = groups, x = genes  →  keep as-is
    if not transpose:
        color_df = color_df.T
        size_df = size_df.T

    n_rows, n_cols = color_df.shape

    # ── 7. Figure size (scanpy cell-unit approach) ─────────────────
    # Match scanpy's swap_axes behavior: taller cells for the gene axis,
    # narrower cells for the group axis.
    if figsize is not None:
        fig_w, fig_h = figsize
    else:
        if not transpose:
            # genes on y → taller cells; groups on x → narrower cells
            mainplot_h = n_rows * _CELL_WIDTH    # 0.37 per gene
            mainplot_w = n_cols * _CELL_HEIGHT   # 0.35 per group
        else:
            # groups on y → shorter cells; genes on x → wider cells
            mainplot_h = n_rows * _CELL_HEIGHT   # 0.35 per group
            mainplot_w = n_cols * _CELL_WIDTH    # 0.37 per gene
        fig_h = max(_MIN_FIGURE_HEIGHT, mainplot_h + 1.0)
        fig_w = mainplot_w + _LEGENDS_WIDTH

    # ── 8. Figure + GridSpec ───────────────────────────────────────
    # Apply style matching sc.settings.set_figure_params + sns ticks
    import seaborn as sns
    import matplotlib as mpl
    sns.set_style("ticks")
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 14,
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans",
                            "Bitstream Vera Sans", "sans-serif"],
        "xtick.labelsize": 14, "ytick.labelsize": 14,
        "axes.labelsize": 14, "axes.titlesize": 14,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "savefig.dpi": 150,
    })

    fig = plt.figure(figsize=(fig_w, fig_h))

    mainplot_w_frac = fig_w - _LEGENDS_WIDTH
    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[mainplot_w_frac, _LEGENDS_WIDTH],
        wspace=1.2 / fig_w,
        figure=fig,
    )
    ax = fig.add_subplot(gs[0, 0])

    # ── 9. Plot dots (scanpy-style 0.5-offset grid) ───────────────
    y, x = np.indices(color_df.shape)
    y = y.flatten() + 0.5
    x = x.flatten() + 0.5

    frac_flat = size_df.values.flatten()
    mean_flat = color_df.values.flatten()

    # dot_max / dot_min
    dot_max = np.ceil(np.max(frac_flat) * 10) / 10 if np.max(frac_flat) > 0 else 1.0
    dot_min = 0.0

    # rescale frac to 0–1 within [dot_min, dot_max], then apply exponent
    if dot_min != 0 or dot_max != 1:
        frac_normed = np.clip(frac_flat, dot_min, dot_max)
        frac_normed = (frac_normed - dot_min) / (dot_max - dot_min)
    else:
        frac_normed = frac_flat.copy()

    size = frac_normed ** size_exponent
    size = size * (largest_dot - smallest_dot) + smallest_dot
    size[frac_flat == 0] = 0  # hide zeros

    # color
    cmap_obj = colormaps.get_cmap(cmap)
    norm = Normalize(vmin=0, vmax=1)
    color = cmap_obj(norm(mean_flat))

    ax.scatter(x, y, s=size, c=color)

    # ── 10. Axis formatting ────────────────────────────────────────
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(color_df.index)

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(color_df.columns, rotation=90, ha="center")

    ax.tick_params(axis="both", labelsize="small")
    ax.grid(visible=False)

    # padding (scanpy default: x=0.8, y=1.0; adjusted for 0.5-offset)
    ax.set_ylim(n_rows + (_Y_PADDING - 0.5), -(_Y_PADDING - 0.5))
    ax.set_xlim(-(_X_PADDING - 0.5), n_cols + (_X_PADDING - 0.5))

    # gene labels always at bottom (x-axis default position)

    if grid:
        ax.grid(visible=True, color="gray", linewidth=0.1)
        ax.set_axisbelow(True)

    # ── 11. Legends (right panel, scanpy style) ────────────────────
    _plot_legends(
        fig, gs[0, 1], fig_h,
        dot_max, dot_min, norm, cmap_obj,
        largest_dot, smallest_dot, size_exponent,
    )

    # ── 12. Save / show ────────────────────────────────────────────
    if save:
        plt.savefig(save, bbox_inches="tight", format="pdf", dpi=300)
    if show:
        plt.show()

    return fig, ax
