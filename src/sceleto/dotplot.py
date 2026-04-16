"""Dotplot wrapper around ``scanpy.pl.dotplot`` with per-gene max-normalized color.

Usage
-----
>>> import sceleto as scl
>>> scl.dotplot(adata, ['CD3D', 'CD8A'], 'leiden')

With a marker dict (bracket-grouped x-axis labels):
>>> mk = scl.markers.simple(adata, 'leiden')
>>> scl.dotplot(adata, mk.markers, 'leiden')
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from scipy import sparse


_LAYER_NAME = "_scl_scaled"


# ── helpers ─────────────────────────────────────────────────────────


def _resolve_var_names(var_names, available: set):
    """Return ``(var_group_dict_or_None, flat_gene_list)``.

    - If *var_names* is a mapping, keep the mapping structure so scanpy
      renders bracket-grouped x-axis labels.  Tuple entries ``(gene, score)``
      are accepted.
    - Genes absent from *available* are dropped; empty groups are dropped.
    """
    if isinstance(var_names, Mapping):
        clean: dict = {}
        for k, items in var_names.items():
            names = []
            for it in items:
                g = it[0] if isinstance(it, tuple) else str(it)
                if g in available:
                    names.append(g)
            if names:
                clean[k] = names
        flat = [g for gs in clean.values() for g in gs]
        return clean, flat
    flat = [g for g in var_names if g in available]
    return None, flat


def _add_scaled_layer(adata, groupby: str, layer_name: str = _LAYER_NAME):
    """Attach a per-gene max-normalized layer to *adata* (in-place).

    For each gene g, ``gene_max[g] = max over groups of (group mean of X[:, g])``.
    The layer stores ``X[:, g] / gene_max[g]``.  Because mean is linear, the
    per-group mean of the layer equals ``group_mean / gene_max`` — i.e. the
    ``x / max`` normalization used in ``sceleto.markers``.
    """
    X = adata.X
    labels = adata.obs[groupby].astype(str).to_numpy()
    groups_u = np.unique(labels)

    gene_max = np.zeros(adata.n_vars, dtype=np.float64)
    for g in groups_u:
        mask = labels == g
        if not mask.any():
            continue
        mean_g = np.asarray(X[mask].mean(axis=0)).ravel()
        gene_max = np.maximum(gene_max, mean_g)

    gene_max[gene_max == 0] = 1.0
    inv = 1.0 / gene_max

    if sparse.issparse(X):
        adata.layers[layer_name] = X @ sparse.diags(inv)
    else:
        adata.layers[layer_name] = np.asarray(X) * inv[np.newaxis, :]


# ── main API ────────────────────────────────────────────────────────


def dotplot(
    adata,
    var_names: Union[Sequence[str], Mapping[str, Sequence]],
    groupby: str,
    *,
    groups: Optional[Sequence[str]] = None,
    transpose: bool = False,
    use_raw: bool = False,
    cmap: str = "OrRd",
    figsize: Optional[Tuple[float, float]] = None,
    save: Optional[str] = None,
    show: bool = True,
    **kwargs,
):
    """Dotplot with per-gene max-normalized color, built on ``scanpy.pl.dotplot``.

    Size encodes fraction of cells expressing the gene (scanpy default).
    Color encodes ``group_mean(gene) / max_group(group_mean(gene))`` per gene,
    so ``vmax=1`` always corresponds to the highest-expressing group.

    Parameters
    ----------
    adata
        AnnData with log1p-normalized ``adata.X``.  Raw is not supported.
    var_names
        Gene list or ``{bracket_name: [gene, ...]}`` / ``{bracket_name: [(gene, score), ...]}``
        mapping.  Mappings render as bracket-grouped x-axis labels via scanpy.
    groupby
        Column in ``adata.obs`` to group cells by.
    groups
        Subset of groups to display.  ``None`` shows all.
    transpose
        If ``True``, genes on x-axis, groups on y-axis.  Default puts genes on y.
    use_raw
        Must be ``False``.  Passing ``True`` raises ``ValueError``.
    cmap
        Matplotlib colormap for color scale (default ``OrRd``).
    figsize
        Manual ``(width, height)`` in inches.
    save
        Path to save figure (PDF, dpi=300).
    show
        Whether to call ``plt.show()``.
    **kwargs
        Forwarded to ``scanpy.pl.dotplot``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if use_raw:
        raise ValueError(
            "sceleto.dotplot: use_raw=True is not supported. "
            "Pass log1p-normalized values via adata.X directly."
        )

    available = set(adata.var_names)
    var_group_dict, flat_genes = _resolve_var_names(var_names, available)
    if not flat_genes:
        raise ValueError(
            "sceleto.dotplot: none of the provided genes are in adata.var_names."
        )

    # subset copy: only required genes, optionally restrict cells to `groups`
    ad = adata[:, flat_genes].copy()
    if groups is not None:
        mask = ad.obs[groupby].astype(str).isin([str(g) for g in groups])
        ad = ad[mask].copy()

    # per-gene max-normalized layer
    _add_scaled_layer(ad, groupby, layer_name=_LAYER_NAME)

    # dict → bracket-grouped x-axis via scanpy; else flat list
    sc_var = var_group_dict if var_group_dict is not None else flat_genes

    # Use DotPlot class API directly: module-level sc.pl.dotplot does not
    # expose dot_edge_* in scanpy 1.12; those live on DotPlot.style().
    dp = sc.pl.DotPlot(
        ad,
        sc_var,
        groupby,
        use_raw=False,
        layer=_LAYER_NAME,
        vmin=0,
        vmax=1,
        figsize=figsize,
        **kwargs,
    )
    dp.style(cmap=cmap, dot_edge_color="none", dot_edge_lw=0)
    dp.legend(colorbar_title="Max-scaled\nmean")
    if not transpose:
        dp.swap_axes()

    dp.make_figure()
    fig = dp.fig

    if save:
        fig.savefig(save, bbox_inches="tight", format="pdf", dpi=300)
    if show:
        plt.show()
    return fig
