"""General-purpose dotplot for gene expression across groups.

Usage
-----
>>> import sceleto as sl
>>> sl.dotplot(adata, ['CD3D', 'CD8A', 'MS4A1'], 'leiden')

With marker results:
>>> mk = sl.markers.simple(adata, 'leiden')
>>> sl.dotplot(adata, mk.mks, 'leiden')
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse


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


def _resolve_genes(
    genes: Union[Sequence[str], Dict[str, list]],
) -> List[str]:
    """Convert gene input to a flat list with spacer separators.

    Accepts:
    - list of gene names: ['CD3D', 'CD8A']
    - dict from marker output {cluster: [(gene, score), ...]}: uses gene names
    - dict {cluster: [gene, ...]}: flat gene lists
    """
    if isinstance(genes, dict):
        gene_list = []
        for group in sorted(genes.keys()):
            items = genes[group]
            for item in items:
                if isinstance(item, tuple):
                    gene_list.append(item[0])
                else:
                    gene_list.append(str(item))
            gene_list.append(" ")  # spacer between groups
        return gene_list

    return list(genes)


def dotplot_size_legend(
    figscale: float = 0.25,
    fracs: Sequence[float] = (0.25, 0.50, 0.75, 1.00),
    fontsize: int = 10,
    save: Optional[str] = None,
    show: bool = True,
):
    """Standalone size legend for dotplot (fraction expressing).

    Use this alongside dotplot() to show what dot sizes mean.

    Parameters
    ----------
    figscale
        Should match the figscale used in dotplot().
    fracs
        Fraction values to show in the legend.
    fontsize
        Font size for labels.
    save
        Path to save figure as PDF.
    show
        Whether to call plt.show().

    Returns
    -------
    fig, ax
    """
    dot_scale = figscale * 400
    n = len(fracs)
    fig, ax = plt.subplots(figsize=(0.9 * n + 1.5, 0.8))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_axis_off()

    for i, f in enumerate(fracs):
        ax.scatter([i], [0.1], s=f * dot_scale, c="grey", edgecolors="none")
        ax.text(i, -0.25, f"{int(f*100)}", fontsize=fontsize, ha="center", va="top")

    ax.set_title("Fraction\nexpressing (%)", fontsize=fontsize, pad=10)

    if save:
        plt.savefig(save, bbox_inches="tight", format="pdf", dpi=300)
    if show:
        plt.show()

    return fig, ax


def dotplot(
    adata,
    genes: Union[Sequence[str], Dict[str, list]],
    groupby: str,
    *,
    groups: Optional[Sequence[str]] = None,
    n_top: Optional[int] = None,
    transpose: bool = False,
    use_raw: bool = True,
    fontsize: int = 10,
    figscale: float = 0.25,
    min_frac: float = 0.01,
    cmap: str = "OrRd",
    size_legend: bool = True,
    save: Optional[str] = None,
    show: bool = True,
):
    """Dotplot of gene expression across groups.

    Size = fraction of cells expressing the gene.
    Color = normalized mean expression (per-gene, 0-1).

    Parameters
    ----------
    adata
        AnnData object.
    genes
        Gene names as a list, or marker dict {group: [(gene, score), ...]}.
    groupby
        Column in adata.obs to group cells by.
    groups
        Subset of groups to show. If None, show all.
    n_top
        When genes is a dict, limit to top-n per group.
    transpose
        If True, genes on x-axis and groups on y-axis.
    use_raw
        If True, use adata.raw for expression values.
    fontsize
        Font size for tick labels.
    figscale
        Scaling factor for figure and dot sizes.
    min_frac
        Minimum fraction expressing to show a dot (below → size 0).
    cmap
        Colormap for mean expression.
    save
        Path to save figure as PDF.
    show
        Whether to call plt.show().

    Returns
    -------
    fig, ax
    """
    # resolve genes
    gene_list = _resolve_genes(genes)
    if n_top is not None and isinstance(genes, dict):
        gene_list = []
        for group_key in sorted(genes.keys()):
            items = genes[group_key]
            for item in items[:n_top]:
                gene_list.append(item[0] if isinstance(item, tuple) else str(item))
            gene_list.append(" ")

    # separate real genes from spacers
    real_genes = [g for g in gene_list if g.strip()]

    # compute stats
    mean, frac, all_groups, valid_genes = _compute_group_stats(
        adata, groupby, real_genes, use_raw=use_raw,
    )

    if groups is not None:
        group_idx = [i for i, g in enumerate(all_groups) if g in set(groups)]
        all_groups = [all_groups[i] for i in group_idx]
        mean = mean[group_idx]
        frac = frac[group_idx]

    gene_to_col = {g: i for i, g in enumerate(valid_genes)}
    n_groups = len(all_groups)

    # build scatter data
    sizes = []
    colors = []
    pos_major = []
    pos_minor = []
    ticks_major = []
    ticklabels_major = []

    major_idx = 0
    iterate = gene_list[::-1] if not transpose else gene_list

    for gene in iterate:
        if gene.strip() and gene in gene_to_col:
            major_idx += 1
            col = gene_to_col[gene]

            frac_vals = frac[:, col].copy()
            frac_vals[frac_vals < min_frac] = 0
            mean_vals = mean[:, col].copy()
            max_mean = np.max(mean_vals)
            norm_vals = (mean_vals / max_mean) if max_mean > 0 else mean_vals

            sizes.extend(frac_vals)
            colors.extend(norm_vals)
            ticks_major.append(major_idx)
            ticklabels_major.append(gene)
        else:
            major_idx += 0.5
            sizes.extend([0] * n_groups)
            colors.extend([0] * n_groups)

        if transpose:
            pos_major.extend([major_idx] * n_groups)
            pos_minor.extend(list(range(n_groups))[::-1])
        else:
            pos_minor.extend(list(range(n_groups)))
            pos_major.extend([major_idx] * n_groups)

    sizes = np.array(sizes)
    colors = np.array(colors)

    # figure — figscale controls both spacing and dot size
    dot_scale = figscale * 400

    if transpose:
        fw = len(gene_list) * figscale * 0.45
        fh = n_groups * figscale * 0.55
    else:
        fw = n_groups * figscale * 0.65
        fh = len(gene_list) * figscale * 0.55

    fig, ax = plt.subplots(figsize=(fw, fh))

    if transpose:
        sc = ax.scatter(
            pos_major, pos_minor,
            s=sizes * dot_scale, c=colors, cmap=cmap, vmax=1.0,
        )
        ax.xaxis.tick_top()
        ax.set_xticks(ticks_major)
        ax.set_xticklabels(ticklabels_major, rotation=90, fontsize=fontsize)
        ax.set_yticks(range(n_groups))
        ax.set_yticklabels(all_groups[::-1], fontsize=fontsize)
    else:
        sc = ax.scatter(
            pos_minor, pos_major,
            s=sizes * dot_scale, c=colors, cmap=cmap, vmax=1.0,
        )
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(all_groups, rotation=90, fontsize=fontsize)
        ax.set_yticks(ticks_major)
        ax.set_yticklabels(ticklabels_major, fontsize=fontsize)

    if pos_minor:
        ax.set_xlim(min(pos_minor) - 0.5, max(pos_minor) + 0.5)
    if pos_major:
        ax.set_ylim(min(pos_major) - 0.5, max(pos_major) + 0.5)
    ax.grid(False)

    # Colorbar — thin vertical bar on the right (matching dotplot height)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cbar_size = "2%" if transpose else "5%"
    cax = divider.append_axes("right", size=cbar_size, pad=0.05)
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label("Mean expr\n(normalized)", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    # Size legend — below the dotplot
    if size_legend:
        legend_fracs = [0.25, 0.50, 0.75, 1.00]
        leg_inches = 0.8

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_tight = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        actual_bottom = ax_tight.y0

        orig_w, orig_h = fig.get_size_inches()
        new_h = orig_h + leg_inches
        fig.set_size_inches(orig_w, new_h)

        for a in fig.axes:
            pos = a.get_position()
            a.set_position([pos.x0,
                            (pos.y0 * orig_h + leg_inches) / new_h,
                            pos.width,
                            pos.height * orig_h / new_h])

        actual_bottom_new = (actual_bottom * orig_h + leg_inches) / new_h
        bbox = ax.get_position()
        leg_h = (leg_inches * 0.55) / new_h
        leg_y = actual_bottom_new - 0.04 - leg_h

        # Align legend width to first/last group dot positions
        first_group_data = 0  # first group x in data coords
        last_group_data = n_groups - 1  # last group x in data coords
        trans = ax.transData + fig.transFigure.inverted()
        x_left = trans.transform((first_group_data, 0))[0]
        x_right = trans.transform((last_group_data, 0))[0]
        leg_x = x_left
        leg_w = x_right - x_left

        leg_ax = fig.add_axes([leg_x, leg_y, leg_w, leg_h])
        leg_ax.set_axis_off()
        n_leg = len(legend_fracs)
        leg_ax.set_xlim(-0.8, n_leg - 0.2)
        leg_ax.set_ylim(-1.0, 1.0)
        for i, f in enumerate(legend_fracs):
            leg_ax.scatter([i], [0.2], s=f * dot_scale, c="grey", edgecolors="none")
            leg_ax.text(i, -0.5, f"{int(f*100)}", fontsize=fontsize,
                        ha="center", va="top")
        leg_ax.text((n_leg - 1) / 2, 1.0, "Fraction\nexpressing (%)",
                    fontsize=fontsize, ha="center", va="bottom")

    if save:
        plt.savefig(save, bbox_inches="tight", format="pdf", dpi=300)
    if show:
        plt.show()

    return fig, ax
