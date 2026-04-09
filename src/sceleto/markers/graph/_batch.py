"""Batch-aware FC validation for candidate marker genes.

After the main marker algorithm identifies candidate genes per edge,
this module checks whether those markers hold across batch combinations.

For edge A→B with batches {x1,x2,x3} in A and {x1,x2,x4} in B,
we compute FC for all pairs: A_x1→B_x1, A_x1→B_x2, A_x1→B_x4, ...
Only candidate genes are evaluated (not all genes) for efficiency.
"""

from __future__ import annotations

from typing import List, Optional

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


def compute_batch_edge_fc(
    adata,
    ctx,
    edge_gene_df: pd.DataFrame,
    batch_key: str,
    *,
    use_raw: bool = True,
    eps: float = 1e-3,
    min_cells: int = 5,
    fc_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Compute batch-pair FC for candidate marker genes on each edge.

    For each (start→end, gene) in *edge_gene_df* (start=low, end=high):

    * Find batches with ≥ *min_cells* in start and end groups.
    * Compute ``FC = mean(end_bj) / mean(start_bi)`` for all batch pairs
      ``(bi, bj)``.
    * Summarise with ``n_batch_pairs``, ``min_batch_fc``,
      ``median_batch_fc``, and ``frac_batch_pass``.

    Parameters
    ----------
    adata : AnnData
    ctx : MarkerContext
    edge_gene_df : DataFrame
        Output of :func:`compute_fc_delta`.
    batch_key : str
        Column in ``adata.obs``.
    use_raw : bool
    eps : float
    min_cells : int
        Minimum cells per (group, batch) to include that combination.
    fc_threshold : float, optional
        Threshold for *frac_batch_pass*.  Defaults to ``thres_fc``
        (median FC in *edge_gene_df* if not given).

    Returns
    -------
    DataFrame — copy of *edge_gene_df* with added columns:
        ``n_batch_pairs``, ``min_batch_fc``, ``median_batch_fc``,
        ``frac_batch_pass``.
    """
    candidate_genes = edge_gene_df["gene"].unique().tolist()

    mean, _, groups, batches, genes, group_to_idx = _compute_batch_group_mean(
        adata, ctx.groupby, batch_key, candidate_genes,
        use_raw=use_raw, min_cells=min_cells,
    )
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    if fc_threshold is None:
        fc_threshold = float(edge_gene_df["fc"].median())

    n_rows = len(edge_gene_df)
    out_n_pairs = np.zeros(n_rows, dtype=np.int32)
    out_min_fc = np.full(n_rows, np.nan, dtype=np.float32)
    out_median_fc = np.full(n_rows, np.nan, dtype=np.float32)
    out_frac_pass = np.full(n_rows, np.nan, dtype=np.float32)

    for row_idx, (_, row) in enumerate(edge_gene_df.iterrows()):
        start, end, gene = str(row["start"]), str(row["end"]), str(row["gene"])

        if start not in group_to_idx or end not in group_to_idx or gene not in gene_to_idx:
            continue

        si = group_to_idx[start]
        ei = group_to_idx[end]
        gi = gene_to_idx[gene]

        start_means = mean[si, :, gi]
        end_means = mean[ei, :, gi]

        start_valid = np.where(~np.isnan(start_means))[0]
        end_valid = np.where(~np.isnan(end_means))[0]

        if len(start_valid) == 0 or len(end_valid) == 0:
            continue

        # Vectorised cross-product of batch FCs
        sm = start_means[start_valid]  # (S,)
        em = end_means[end_valid]      # (E,)
        fcs = (em[None, :] + eps) / (sm[:, None] + eps)  # (S, E)
        fcs = fcs.ravel()

        out_n_pairs[row_idx] = len(fcs)
        out_min_fc[row_idx] = float(np.min(fcs))
        out_median_fc[row_idx] = float(np.median(fcs))
        out_frac_pass[row_idx] = float(np.mean(fcs >= fc_threshold))

    result = edge_gene_df.copy()
    result["n_batch_pairs"] = out_n_pairs
    result["min_batch_fc"] = out_min_fc
    result["median_batch_fc"] = out_median_fc
    result["frac_batch_pass"] = out_frac_pass
    return result


def aggregate_batch_stats(
    batch_edge_fc_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate batch FC stats from edge-level to (group, gene) level.

    For each (group, gene) where *group* is the "end" (high expression side):

    * ``batch_min_fc`` — min of *min_batch_fc* across edges (worst case).
    * ``batch_median_fc`` — mean of *median_batch_fc* across edges.
    * ``batch_frac_pass`` — min of *frac_batch_pass* across edges.

    Returns
    -------
    DataFrame with columns: ``group``, ``gene``, ``batch_min_fc``,
    ``batch_median_fc``, ``batch_frac_pass``.
    """
    df = batch_edge_fc_df.dropna(subset=["min_batch_fc"]).copy()
    if df.empty:
        return pd.DataFrame(
            columns=["group", "gene", "batch_min_fc", "batch_median_fc", "batch_frac_pass"],
        )

    agg = (
        df.groupby(["end", "gene"])
        .agg(
            batch_min_fc=("min_batch_fc", "min"),
            batch_median_fc=("median_batch_fc", "mean"),
            batch_frac_pass=("frac_batch_pass", "min"),
        )
        .reset_index()
        .rename(columns={"end": "group"})
    )
    return agg


def fill_propagated_batch_stats(
    adata,
    ctx,
    specific_ranking_df: pd.DataFrame,
    batch_stats_df: pd.DataFrame,
    batch_key: str,
    *,
    use_raw: bool = True,
    eps: float = 1e-3,
    min_cells: int = 5,
    fc_threshold: float = 3.0,
) -> pd.DataFrame:
    """Fill batch stats for (group, gene) pairs that lack edges.

    Genes labelled +1 via level-2/3 propagation may not appear as the
    "end" of any row in *edge_gene_df*.  For each such missing pair we
    build **synthetic edges** from every PAGA neighbor where the group
    has higher ``mean_norm``, then compute batch-pair FC on those.

    Returns
    -------
    Updated *batch_stats_df* with the missing pairs filled in.
    """
    if batch_stats_df.empty:
        existing: set = set()
    else:
        existing = set(zip(
            batch_stats_df["group"].astype(str),
            batch_stats_df["gene"].astype(str),
        ))

    all_pairs = set(zip(
        specific_ranking_df["group"].astype(str),
        specific_ranking_df["gene"].astype(str),
    ))
    missing = all_pairs - existing

    if not missing:
        return batch_stats_df

    # Build neighbor map from PAGA edges
    neighbors: dict = {}
    for a, b in ctx.undirected_edges:
        a_s, b_s = str(a), str(b)
        neighbors.setdefault(a_s, set()).add(b_s)
        neighbors.setdefault(b_s, set()).add(a_s)

    ctx_gene_to_idx = {str(g): i for i, g in enumerate(ctx.genes)}

    # Build synthetic edge rows: (start=low_neighbor, end=group, gene, fc, delta)
    synthetic_rows = []
    for group, gene in missing:
        gene_ci = ctx_gene_to_idx.get(gene)
        group_ci = ctx.group_to_idx.get(group)
        if gene_ci is None or group_ci is None:
            continue

        group_expr = float(ctx.mean_norm[group_ci, gene_ci])

        for nbr in neighbors.get(group, []):
            nbr_ci = ctx.group_to_idx.get(nbr)
            if nbr_ci is None:
                continue
            nbr_expr = float(ctx.mean_norm[nbr_ci, gene_ci])
            if group_expr <= nbr_expr:
                continue  # group is not the high side vs this neighbor
            fc = (group_expr + eps) / (nbr_expr + eps)
            delta = group_expr - nbr_expr
            synthetic_rows.append((nbr, group, gene, fc, delta))

    if not synthetic_rows:
        return batch_stats_df

    synthetic_edge_df = pd.DataFrame(
        synthetic_rows, columns=["start", "end", "gene", "fc", "delta"],
    )

    batch_synthetic = compute_batch_edge_fc(
        adata, ctx, synthetic_edge_df, batch_key,
        use_raw=use_raw, eps=eps, min_cells=min_cells,
        fc_threshold=fc_threshold,
    )

    extra_stats = aggregate_batch_stats(batch_synthetic)
    if extra_stats.empty:
        return batch_stats_df

    return pd.concat([batch_stats_df, extra_stats], ignore_index=True)


def get_batch_pair_detail(
    adata,
    ctx,
    edge_gene_df: pd.DataFrame,
    batch_key: str,
    gene: str,
    group: str,
    *,
    use_raw: bool = True,
    eps: float = 1e-3,
    min_cells: int = 5,
    fc_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Return per-batch-pair FC table for a specific (gene, group).

    Finds every edge in *edge_gene_df* where *group* is the high side
    ("end") for *gene*, then expands all batch-pair combinations.

    Parameters
    ----------
    adata : AnnData
    ctx : MarkerContext
    edge_gene_df : DataFrame
        Output of :func:`compute_fc_delta`.
    batch_key : str
    gene, group : str
        The marker gene and its high-expression cluster.
    use_raw, eps, min_cells : see :func:`compute_batch_edge_fc`.
    fc_threshold : float, optional
        Threshold for the ``pass`` column.  Defaults to ``thres_fc``
        (median FC in *edge_gene_df*).

    Returns
    -------
    DataFrame with columns:
        ``edge_start``, ``edge_end``, ``batch_start``, ``batch_end``,
        ``mean_start``, ``mean_end``, ``fc``, ``n_cells_start``,
        ``n_cells_end``, ``pass``.
    """
    # Find edges where this gene is a candidate and group is the high side
    mask = (edge_gene_df["gene"].astype(str) == gene) & (edge_gene_df["end"].astype(str) == group)
    edges = edge_gene_df.loc[mask]

    if edges.empty:
        return pd.DataFrame(
            columns=[
                "edge_start", "edge_end", "batch_start", "batch_end",
                "mean_start", "mean_end", "fc",
                "n_cells_start", "n_cells_end", "pass",
            ],
        )

    if fc_threshold is None:
        fc_threshold = float(edge_gene_df["fc"].median())

    mean, n_cells_mat, groups, batches, genes, group_to_idx = _compute_batch_group_mean(
        adata, ctx.groupby, batch_key, [gene],
        use_raw=use_raw, min_cells=min_cells,
    )
    # gene is index 0 since we passed a single gene
    gi = 0

    rows = []
    for _, edge_row in edges.iterrows():
        start = str(edge_row["start"])
        end = str(edge_row["end"])

        if start not in group_to_idx or end not in group_to_idx:
            continue

        si = group_to_idx[start]
        ei = group_to_idx[end]

        for bi, b_start in enumerate(batches):
            if np.isnan(mean[si, bi, gi]):
                continue
            for bj, b_end in enumerate(batches):
                if np.isnan(mean[ei, bj, gi]):
                    continue
                m_s = float(mean[si, bi, gi])
                m_e = float(mean[ei, bj, gi])
                fc_val = (m_e + eps) / (m_s + eps)
                rows.append({
                    "edge_start": start,
                    "edge_end": end,
                    "batch_start": b_start,
                    "batch_end": b_end,
                    "mean_start": m_s,
                    "mean_end": m_e,
                    "fc": fc_val,
                    "n_cells_start": int(n_cells_mat[si, bi]),
                    "n_cells_end": int(n_cells_mat[ei, bj]),
                    "pass": fc_val >= fc_threshold,
                })

    return pd.DataFrame(rows)


