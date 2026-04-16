"""Batch-aware marker validation for candidate marker genes.

Provides:
- Welch's t-test across batch-level means (lightweight statistical check)
- Per-(gene, group) batch mean inspection utility
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _compute_batch_group_mean(
    adata,
    groupby: str,
    batch_key: str,
    candidate_genes: List[str],
    *,
    use_raw: bool = True,
    min_cells: int = 5,
):
    """Compute mean expression per (group, batch) for candidate genes.

    Returns
    -------
    mean : ndarray (n_groups, n_batches, n_genes) — NaN where n_cells < min_cells
    n_cells : ndarray (n_groups, n_batches)
    groups, batches : list of str
    genes : ndarray of str
    group_to_idx : dict
    """
    from scipy import sparse

    from sceleto._expr import resolve_expression
    X, var_names, _ = resolve_expression(adata)
    all_genes = var_names.to_numpy()

    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr()

    gene_to_col = {g: i for i, g in enumerate(all_genes)}
    candidate_genes = [g for g in candidate_genes if g in gene_to_col]
    gene_cols = np.array([gene_to_col[g] for g in candidate_genes])
    X_sub = X[:, gene_cols]

    groups = sorted(adata.obs[groupby].astype(str).unique().tolist())
    batches = sorted(adata.obs[batch_key].astype(str).unique().tolist())
    group_to_idx = {g: i for i, g in enumerate(groups)}

    obs_groups = adata.obs[groupby].astype(str).to_numpy()
    obs_batches = adata.obs[batch_key].astype(str).to_numpy()

    n_groups = len(groups)
    n_batches = len(batches)
    n_genes = len(candidate_genes)

    mean = np.full((n_groups, n_batches, n_genes), np.nan, dtype=np.float32)
    n_cells_mat = np.zeros((n_groups, n_batches), dtype=np.int32)

    for g_name, g_idx in group_to_idx.items():
        g_mask = obs_groups == g_name
        for b_idx, b_name in enumerate(batches):
            mask = g_mask & (obs_batches == b_name)
            nc = int(mask.sum())
            n_cells_mat[g_idx, b_idx] = nc
            if nc < min_cells:
                continue
            mean[g_idx, b_idx] = np.asarray(X_sub[mask].mean(axis=0)).ravel()

    return mean, n_cells_mat, groups, batches, np.array(candidate_genes), group_to_idx


def compute_batch_ttest(
    adata,
    ctx,
    edge_gene_df: pd.DataFrame,
    batch_key: str,
    *,
    use_raw: bool = True,
    min_cells: int = 5,
    min_batches: int = 3,
    eps: float = 1e-3,
) -> pd.DataFrame:
    """Welch's t-test between batch-level means of start and end groups.

    For each (edge, gene) row in *edge_gene_df*, treat each valid batch's
    mean expression as one observation.  Runs a Welch's t-test
    (``scipy.stats.ttest_ind(equal_var=False)``) when both start and end have
    ≥ *min_batches* valid batches.

    Parameters
    ----------
    adata : AnnData
    ctx : MarkerContext
    edge_gene_df : DataFrame
        Output of :func:`compute_fc_delta`.
    batch_key : str
        Column in ``adata.obs``.
    min_cells : int
        Minimum cells per (group, batch) to treat as valid.
    min_batches : int
        Minimum valid batches required on *both* sides to run the test.
    eps : float
        Added to means before testing.

    Returns
    -------
    DataFrame — copy of *edge_gene_df* with added columns:
        ``ttest_pval``, ``ttest_n_start``, ``ttest_n_end``.
        Rows where the condition is not met have NaN in those columns.
    """
    from scipy.stats import ttest_ind

    candidate_genes = edge_gene_df["gene"].unique().tolist()

    mean, _, groups, batches, genes, group_to_idx = _compute_batch_group_mean(
        adata, ctx.groupby, batch_key, candidate_genes,
        use_raw=use_raw, min_cells=min_cells,
    )
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    n_rows = len(edge_gene_df)
    out_pval = np.full(n_rows, np.nan, dtype=np.float64)
    out_n_start = np.zeros(n_rows, dtype=np.int32)
    out_n_end = np.zeros(n_rows, dtype=np.int32)

    for row_idx, (_, row) in enumerate(edge_gene_df.iterrows()):
        start, end, gene = str(row["start"]), str(row["end"]), str(row["gene"])

        if start not in group_to_idx or end not in group_to_idx or gene not in gene_to_idx:
            continue

        si = group_to_idx[start]
        ei = group_to_idx[end]
        gi = gene_to_idx[gene]

        start_means = mean[si, :, gi]
        end_means = mean[ei, :, gi]

        start_valid = start_means[~np.isnan(start_means)]
        end_valid = end_means[~np.isnan(end_means)]

        n_s = len(start_valid)
        n_e = len(end_valid)

        out_n_start[row_idx] = n_s
        out_n_end[row_idx] = n_e

        if n_s < min_batches or n_e < min_batches:
            continue

        _, pval = ttest_ind(end_valid + eps, start_valid + eps, equal_var=False)
        out_pval[row_idx] = float(pval)

    result = edge_gene_df.copy()
    result["ttest_pval"] = out_pval
    result["ttest_n_start"] = out_n_start
    result["ttest_n_end"] = out_n_end
    return result


def filter_edge_gene_df_by_ttest(
    adata,
    ctx,
    edge_gene_df: pd.DataFrame,
    batch_key: str,
    *,
    use_raw: bool = True,
    min_cells: int = 5,
    min_batches: int = 3,
    alpha: float = 0.05,
    eps: float = 1e-3,
) -> pd.DataFrame:
    """Filter edge-gene candidates by Welch's t-test across batch means.

    Keeps a row if:
    - t-test was not run (either side has < *min_batches* valid batches) → can't test, keep
    - t-test pval < *alpha* → statistically significant, keep
    Drops rows where pval ≥ *alpha* (test ran but not significant).

    Parameters
    ----------
    edge_gene_df : DataFrame
        Output of :func:`compute_fc_delta` (FC-filtered candidates).
    alpha : float
        Significance threshold. Default 0.05.

    Returns
    -------
    Filtered DataFrame (subset of *edge_gene_df*, index reset).
    """
    if edge_gene_df.empty:
        return edge_gene_df

    ttest_df = compute_batch_ttest(
        adata, ctx, edge_gene_df, batch_key,
        use_raw=use_raw, min_cells=min_cells,
        min_batches=min_batches, eps=eps,
    )
    # keep if: ttest not run (NaN) OR pval < alpha
    keep = ttest_df["ttest_pval"].isna() | (ttest_df["ttest_pval"] < alpha)
    return edge_gene_df[keep].reset_index(drop=True)


def aggregate_ttest_stats(
    ttest_edge_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate t-test stats from edge-level to (group, gene) level.

    For each (group=end, gene):

    * ``ttest_pval`` — minimum p-value across edges (most significant).
    * ``ttest_n_start_min``, ``ttest_n_end_min`` — minimum batch counts
      (worst-case data availability).

    Returns
    -------
    DataFrame with columns: ``group``, ``gene``, ``ttest_pval``,
    ``ttest_n_start_min``, ``ttest_n_end_min``.
    """
    df = ttest_edge_df.dropna(subset=["ttest_pval"]).copy()
    if df.empty:
        return pd.DataFrame(
            columns=["group", "gene", "ttest_pval", "ttest_n_start_min", "ttest_n_end_min"],
        )

    agg = (
        df.groupby(["end", "gene"])
        .agg(
            ttest_pval=("ttest_pval", "min"),
            ttest_n_start_min=("ttest_n_start", "min"),
            ttest_n_end_min=("ttest_n_end", "min"),
        )
        .reset_index()
        .rename(columns={"end": "group"})
    )
    return agg


def get_batch_mean_detail(
    adata,
    ctx,
    edge_gene_df: pd.DataFrame,
    batch_key: str,
    gene: str,
    group: str,
    *,
    use_raw: bool = True,
    min_cells: int = 5,
) -> pd.DataFrame:
    """Return per-batch mean expression for a specific (gene, group).

    For each edge where *group* is the high side ("end") for *gene*,
    returns the batch-level mean expression of start and end groups.

    Returns
    -------
    DataFrame with columns:
        ``edge_start``, ``edge_end``, ``batch``,
        ``mean_start``, ``mean_end``, ``n_cells_start``, ``n_cells_end``.
    """
    mask = (edge_gene_df["gene"].astype(str) == gene) & (edge_gene_df["end"].astype(str) == group)
    edges = edge_gene_df.loc[mask]

    if edges.empty:
        return pd.DataFrame(
            columns=["edge_start", "edge_end", "batch",
                     "mean_start", "mean_end", "n_cells_start", "n_cells_end"],
        )

    mean, n_cells_mat, groups, batches, genes, group_to_idx = _compute_batch_group_mean(
        adata, ctx.groupby, batch_key, [gene],
        use_raw=use_raw, min_cells=min_cells,
    )
    gi = 0

    rows = []
    for _, edge_row in edges.iterrows():
        start = str(edge_row["start"])
        end = str(edge_row["end"])

        if start not in group_to_idx or end not in group_to_idx:
            continue

        si = group_to_idx[start]
        ei = group_to_idx[end]

        for bi, b_name in enumerate(batches):
            m_s = mean[si, bi, gi]
            m_e = mean[ei, bi, gi]
            rows.append({
                "edge_start": start,
                "edge_end": end,
                "batch": b_name,
                "mean_start": float(m_s) if not np.isnan(m_s) else np.nan,
                "mean_end": float(m_e) if not np.isnan(m_e) else np.nan,
                "n_cells_start": int(n_cells_mat[si, bi]),
                "n_cells_end": int(n_cells_mat[ei, bi]),
            })

    return pd.DataFrame(rows)
