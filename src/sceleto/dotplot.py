"""Dotplot wrapper around ``scanpy.pl.dotplot`` with per-gene max-normalized color.

Usage
-----
>>> import sceleto as scl
>>> scl.dotplot(adata, ['CD3D', 'CD8A'], 'leiden')

For marker outputs, prefer the convenience method:
>>> mk = scl.markers.simple(adata, 'leiden')
>>> mk.plot()
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
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
    - Groups with no valid genes are silently dropped.
    - Genes absent from *available* are dropped.
    """
    if isinstance(var_names, Mapping):
        clean: dict = {}
        for k, items in var_names.items():
            names = []
            for it in items:
                g = it[0] if isinstance(it, tuple) else str(it)
                if g in available:
                    names.append(g)
            if names:  # groups with no valid genes are silently dropped
                clean[k] = names
        # deduplicate while preserving order (same gene can appear in multiple groups)
        seen: set = set()
        flat = []
        for g in (g for gs in clean.values() for g in gs):
            if g not in seen:
                flat.append(g)
                seen.add(g)
        return clean, flat
    flat = [g for g in var_names if g in available]
    return None, flat


def _check_log1p_normalized(X, label: str = "adata.X"):
    """Check if *X* looks like log1p-normalized data.

    - Negative values → ``ValueError`` (definitive failure).
    - Max > 30 → ``UserWarning`` only (heuristic; proceed anyway).
    """
    x_sub = X[: min(500, X.shape[0])]
    min_val = float(np.asarray(x_sub.min() if sparse.issparse(x_sub) else x_sub.min()))
    if min_val < 0:
        raise ValueError(
            f"{label} has negative values — looks like scaled data."
        )
    max_val = float(np.asarray(x_sub.max() if sparse.issparse(x_sub) else x_sub.max()))
    if max_val > 30:
        warnings.warn(
            f"{label} max = {max_val:.1f}; may not be log1p-normalized.",
            UserWarning,
            stacklevel=3,
        )


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
    swap_axes: bool = False,
    use_raw: bool = True,
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

    Follows scanpy's default axis orientation: genes on x-axis, groups on y-axis.
    Pass ``swap_axes=True`` to put genes on y-axis, groups on x-axis.

    Parameters
    ----------
    adata
        AnnData with log1p-normalized expression.
    var_names
        Gene list or ``{bracket_name: [gene, ...]}`` / ``{bracket_name: [(gene, score), ...]}``
        mapping.  Mappings render as bracket-grouped labels via scanpy.
    groupby
        Column in ``adata.obs`` to group cells by.
    groups
        Subset of groups to display.  ``None`` shows all.
    swap_axes
        If ``True``, genes on y-axis, groups on x-axis (swaps scanpy default).
    use_raw
        If ``True`` (default), read from ``adata.raw.X``.
        If ``False``, read from ``adata.X``.
        Both sources are checked for log1p normalization.
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
    """
    # ── select expression source ─────────────────────────────────────
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True but adata.raw is None.")
        src_var_names = list(adata.raw.var_names)
    else:
        src_var_names = list(adata.var_names)

    available = set(src_var_names)
    var_group_dict, flat_genes = _resolve_var_names(var_names, available)
    if not flat_genes:
        raise ValueError("sceleto.dotplot: none of the provided genes are in var_names.")

    # ── filter cells ─────────────────────────────────────────────────
    if groups is not None:
        cell_mask = adata.obs[groupby].astype(str).isin([str(g) for g in groups]).values
        adata_c = adata[cell_mask]
    else:
        adata_c = adata

    # ── build working AnnData ─────────────────────────────────────────
    if use_raw:
        gene_idx = np.array([src_var_names.index(g) for g in flat_genes])
        X_work = adata_c.raw.X[:, gene_idx]
        _check_log1p_normalized(X_work, "adata.raw.X")
        X_copy = X_work.copy() if sparse.issparse(X_work) else np.asarray(X_work)
        ad = anndata.AnnData(
            X=X_copy,
            obs=adata_c.obs[[groupby]].copy(),
            var=pd.DataFrame(index=pd.Index(flat_genes)),
        )
    else:
        ad = adata_c[:, flat_genes].copy()
        _check_log1p_normalized(ad.X, "adata.X")

    # ── per-gene max-normalized layer ─────────────────────────────────
    _add_scaled_layer(ad, groupby, layer_name=_LAYER_NAME)

    # dict → bracket-grouped x-axis via scanpy; else flat list
    sc_var = var_group_dict if var_group_dict is not None else flat_genes

    # Block kwargs that conflict with sceleto's normalization logic
    _BLOCKED = {
        "layer", "standard_scale",         # normalization
        "vmin", "vmax", "vcenter", "norm",  # color range
        "var_group_positions", "var_group_labels",  # bracket structure
        "dot_color_df", "dot_size_df",      # bypass sceleto logic entirely
    }
    bad = _BLOCKED & set(kwargs)
    if bad:
        raise ValueError(f"sceleto.dotplot: {sorted(bad)} cannot be set.")

    # Split kwargs: .style() params must not go to the constructor
    _STYLE_KEYS = {
        "color_on", "dot_max", "dot_min", "smallest_dot", "largest_dot",
        "size_exponent", "grid", "x_padding", "y_padding",
    }
    style_kwargs = {k: v for k, v in kwargs.items() if k in _STYLE_KEYS}
    dp_kwargs = {k: v for k, v in kwargs.items() if k not in _STYLE_KEYS}

    # Use DotPlot class API directly: module-level sc.pl.dotplot does not
    # expose dot_edge_* in scanpy 1.12; those live on DotPlot.style().
    # ── compact figsize ───────────────────────────────────────────────
    # Passing figsize directly sets min_figure_height = figsize[1], causing
    # legend to scale with plot size. Instead let scanpy auto-calculate by
    # overriding per-cell size on the instance (read in make_figure()).
    dp = sc.pl.DotPlot(
        ad,
        sc_var,
        groupby,
        use_raw=False,
        layer=_LAYER_NAME,
        vmin=0,
        vmax=1,
        figsize=figsize,
        **dp_kwargs,
    )
    if figsize is None:
        dp.DEFAULT_CATEGORY_HEIGHT = 0.27
        dp.DEFAULT_CATEGORY_WIDTH = 0.29
    style_kwargs.setdefault("x_padding", 0.6)
    style_kwargs.setdefault("y_padding", 0.6)
    dp.style(cmap=cmap, dot_edge_color="none", dot_edge_lw=0, **style_kwargs)
    dp.legend(colorbar_title="Max-scaled\nmean")
    if swap_axes:
        dp.swap_axes()

    dp.make_figure()

    if save:
        dp.fig.savefig(save, bbox_inches="tight", format="pdf", dpi=300)
    if show:
        plt.show()
