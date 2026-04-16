"""Correlation-based gene network module.

Pipeline
--------
1. compute_corr        — GOI vs all genes (Pearson r + p-value)
2. build_corr_matrix   — multiple AnnData → wide merged table
3. select_top_genes    — top-N per condition
4. build_feature_matrix — gene × conditions corr matrix
5. build_gene_network  — Euclidean kNN → networkx Graph
6. plot_network        — spring layout, optional condition coloring
7. plot_clustermap     — seaborn hierarchical heatmap
"""

from __future__ import annotations

from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from anndata import AnnData
from scipy.spatial.distance import pdist, squareform
from scipy.stats import t as student_t


# ─────────────────────────────────────────────────────────────────────────────
# 1. compute_corr
# ─────────────────────────────────────────────────────────────────────────────

def compute_corr(
    adata: AnnData,
    gene: str,
    label: Optional[str] = None,
    layer: Optional[str] = None,
    chunk_size: int = 4096,
) -> pd.DataFrame:
    """Pearson correlation of *gene* against all genes in *adata*.

    Parameters
    ----------
    adata
        Input AnnData.
    gene
        Gene of interest; must be in ``adata.var_names``.
    label
        Column prefix for output.  Falls back to ``adata.uns["label"]``
        then ``"sample"``.
    layer
        Layer to use instead of ``adata.X``.
    chunk_size
        Genes processed per chunk (memory control).

    Returns
    -------
    pd.DataFrame
        Columns: ``gene``, ``{label}_corr``, ``{label}_pval``.
    """
    if gene not in adata.var_names:
        raise ValueError(f"{gene!r} not found in adata.var_names")

    lbl = label or adata.uns.get("label", "sample")
    X = adata.layers[layer] if layer is not None else adata.X
    if not (sp.issparse(X) or isinstance(X, np.ndarray)):
        X = np.asarray(X)

    gene_names = np.array(adata.var_names)
    goi_idx = int(np.where(gene_names == gene)[0][0])
    n_cells = adata.n_obs

    # GOI vector
    goi = X[:, goi_idx]
    goi = goi.toarray().ravel() if sp.issparse(goi) else np.asarray(goi).ravel()
    goi_center = goi - goi.mean()
    goi_ss = np.dot(goi_center, goi_center)
    goi_std = np.sqrt(goi_ss / (n_cells - 1)) if n_cells > 1 else 0.0

    corrs = np.full(adata.n_vars, np.nan, dtype=float)

    for start in range(0, adata.n_vars, chunk_size):
        end = min(start + chunk_size, adata.n_vars)
        block = X[:, start:end]
        block = block.toarray() if sp.issparse(block) else np.asarray(block)

        block_center = block - block.mean(axis=0)
        cov = (goi_center[:, None] * block_center).sum(axis=0) / (n_cells - 1)
        block_std = block_center.std(axis=0, ddof=1)

        denom = goi_std * block_std
        good = denom > 0
        corrs[start:end][good] = cov[good] / denom[good]

    corrs[goi_idx] = 1.0

    # t-stat based p-values
    dfree = n_cells - 2
    pvals = np.full_like(corrs, np.nan, dtype=float)
    valid = np.isfinite(corrs) & (np.abs(corrs) < 1.0) & (dfree > 0)
    r = corrs[valid]
    tstat = r * np.sqrt(dfree / (1 - r * r))
    pvals[valid] = 2 * student_t.sf(np.abs(tstat), dfree)
    pvals[goi_idx] = 0.0

    return pd.DataFrame({
        "gene": gene_names,
        f"{lbl}_corr": corrs,
        f"{lbl}_pval": pvals,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 2. build_corr_matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_corr_matrix(
    adatas: dict[str, AnnData],
    gene: str,
    layer: Optional[str] = None,
    chunk_size: int = 4096,
) -> pd.DataFrame:
    """Compute per-condition correlation for *gene* across multiple AnnData objects.

    Parameters
    ----------
    adatas
        ``{label: AnnData}`` mapping.  The key is used as the column prefix.
    gene
        Gene of interest.
    layer
        Layer to use instead of ``adata.X``.
    chunk_size
        Passed to :func:`compute_corr`.

    Returns
    -------
    pd.DataFrame
        Wide table: ``gene`` + ``{label}_corr`` + ``{label}_pval`` per condition.
    """
    merged: Optional[pd.DataFrame] = None
    for lbl, adata in adatas.items():
        df = compute_corr(adata, gene, label=lbl, layer=layer, chunk_size=chunk_size)
        merged = df if merged is None else merged.merge(df, on="gene", how="outer")
    return merged.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. select_top_genes
# ─────────────────────────────────────────────────────────────────────────────

def select_top_genes(
    corr_df: pd.DataFrame,
    top_n: int = 10,
    conditions: Optional[list[str]] = None,
    exclude_goi: bool = True,
) -> pd.DataFrame:
    """Select the top *top_n* positively correlated genes per condition.

    Parameters
    ----------
    corr_df
        Wide table from :func:`build_corr_matrix`.
    top_n
        Number of top genes to keep per condition.
    conditions
        Subset of condition labels (column prefix, i.e. without ``_corr``).
        If None, all ``*_corr`` columns are used.
    exclude_goi
        Drop rank-1 gene (GOI itself, corr = 1.0).

    Returns
    -------
    pd.DataFrame
        Long-form: ``condition``, ``gene``, ``corr``, ``pval``.
    """
    corr_cols = [c for c in corr_df.columns if c.endswith("_corr")]
    if conditions is not None:
        corr_cols = [c for c in corr_cols if c[:-5] in conditions]

    records = []
    for col in corr_cols:
        lbl = col[:-5]
        pval_col = f"{lbl}_pval"
        keep_cols = ["gene", col] + ([pval_col] if pval_col in corr_df.columns else [])
        sub = corr_df[keep_cols].dropna(subset=[col]).sort_values(col, ascending=False)
        if exclude_goi:
            sub = sub.iloc[1:]
        sub = sub.head(top_n)
        for _, row in sub.iterrows():
            records.append({
                "condition": lbl,
                "gene": row["gene"],
                "corr": row[col],
                "pval": row[pval_col] if pval_col in sub.columns else None,
            })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 4. build_feature_matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    top_genes_df: pd.DataFrame,
    corr_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a gene × conditions correlation matrix for network construction.

    Parameters
    ----------
    top_genes_df
        Long-form output of :func:`select_top_genes`.
    corr_df
        Wide table from :func:`build_corr_matrix`.

    Returns
    -------
    pd.DataFrame
        Index = unique genes, columns = condition labels, values = corr
        (NaN filled with 0.0).
    """
    unique_genes = sorted(top_genes_df["gene"].unique())
    corr_cols = [c for c in corr_df.columns if c.endswith("_corr")]

    sub = (
        corr_df.loc[corr_df["gene"].isin(unique_genes), ["gene"] + corr_cols]
        .copy()
        .set_index("gene")
    )
    sub.columns = [c[:-5] for c in corr_cols]  # strip "_corr"
    return sub.reindex(unique_genes).fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. build_gene_network
# ─────────────────────────────────────────────────────────────────────────────

def build_gene_network(
    feature_matrix: pd.DataFrame,
    k: int = 5,
    metric: str = "euclidean",
) -> nx.Graph:
    """Build a k-NN gene network from a feature matrix.

    Parameters
    ----------
    feature_matrix
        Gene × conditions matrix (output of :func:`build_feature_matrix`).
    k
        Number of nearest neighbours per gene.
    metric
        Distance metric passed to ``scipy.spatial.distance.pdist``.

    Returns
    -------
    networkx.Graph
        Nodes = gene names; edge attributes: ``dist``, ``weight``.
    """
    genes = list(feature_matrix.index)
    dist_mat = squareform(pdist(feature_matrix.values, metric=metric))

    G = nx.Graph()
    G.add_nodes_from(genes)
    n = len(genes)

    for i in range(n):
        dists = dist_mat[i].copy()
        dists[i] = np.inf
        for j in np.argsort(dists)[:k]:
            G.add_edge(
                genes[i], genes[j],
                dist=dist_mat[i, j],
                weight=1.0 / (dist_mat[i, j] + 1e-6),
            )
    return G


# ─────────────────────────────────────────────────────────────────────────────
# 6. plot_network
# ─────────────────────────────────────────────────────────────────────────────

def plot_network(
    G: nx.Graph,
    feature_matrix: Optional[pd.DataFrame] = None,
    condition: Optional[str] = None,
    pos: Optional[dict] = None,
    seed: int = 3,
    figsize: tuple[int, int] = (15, 15),
    node_size_range: tuple[int, int] = (50, 600),
    cmap: str = "coolwarm",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Draw a gene network with optional per-condition node coloring.

    Parameters
    ----------
    G
        networkx Graph from :func:`build_gene_network`.
    feature_matrix
        Gene × conditions matrix.  Required when *condition* is set.
    condition
        Column in *feature_matrix* to use for node color/size.
    pos
        Pre-computed layout positions.  If None, spring layout is computed.
    seed
        Random seed for spring layout.
    figsize
    node_size_range
        ``(min_size, max_size)`` when coloring by condition.
    cmap
        Colormap name for condition coloring.
    ax
        Existing Axes to draw on.

    Returns
    -------
    matplotlib Figure
    """
    if pos is None:
        pos = nx.spring_layout(G, weight="weight", seed=seed)

    edges = list(G.edges(data=True))
    max_dist = max((d["dist"] for (_, _, d) in edges), default=1.0)
    edge_widths = [(1 - d["dist"] / max_dist) * 3 for (_, _, d) in edges]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if feature_matrix is not None and condition is not None:
        scores = np.array([feature_matrix.loc[n, condition] for n in G.nodes])
        unique_sorted = np.unique(scores)
        vmin = scores.min()
        vmax = unique_sorted[-2] if len(unique_sorted) >= 2 else unique_sorted[-1]
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        cm = plt.get_cmap(cmap)
        node_colors = cm(norm(scores))
        s01 = mpl.colors.Normalize(vmin=scores.min(), vmax=scores.max())(scores)
        min_s, max_s = node_size_range
        node_sizes = min_s + s01 * (max_s - min_s)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.001, label=f"{condition} corr")
    else:
        node_colors = "#f0f0f0"
        node_sizes = 300

    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color=node_colors,
        edgecolors="#525050", linewidths=0.6, ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos, width=edge_widths, edge_color="gray", alpha=0.6, ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)

    title = "Gene network"
    if condition:
        title += f"\nNode color = {condition} correlation"
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. plot_clustermap
# ─────────────────────────────────────────────────────────────────────────────

def plot_clustermap(
    feature_matrix: pd.DataFrame,
    figsize: tuple[int, int] = (15, 35),
    cmap: str = "coolwarm",
    max_genes: int = 96,
) -> sns.matrix.ClusterGrid:
    """Hierarchically clustered heatmap of the feature matrix.

    Parameters
    ----------
    feature_matrix
        Gene × conditions matrix.
    figsize
    cmap
    max_genes
        If more genes than this, keep top *max_genes* by mean |corr|.

    Returns
    -------
    seaborn ClusterGrid
    """
    mat = feature_matrix.copy()
    if mat.shape[0] > max_genes:
        mat = (
            mat.assign(_mean_abs=mat.abs().mean(axis=1))
            .sort_values("_mean_abs", ascending=False)
            .drop(columns="_mean_abs")
            .head(max_genes)
        )

    g = sns.clustermap(
        mat,
        cmap=cmap,
        center=0,
        linewidths=0.1,
        dendrogram_ratio=(0.1, 0.05),
        colors_ratio=0.003,
        figsize=figsize,
        yticklabels=True,
        xticklabels=True,
        cbar_pos=(0.02, 0.95, 0.01, 0.05),
    )
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=6)
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8,
    )
    g.ax_heatmap.set_xlabel("Condition")
    g.ax_heatmap.set_ylabel("Gene")
    return g
