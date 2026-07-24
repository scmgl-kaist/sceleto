"""Fast one-vs-rest Wilcoxon AUC differential expression — a ``presto`` port.

CellChat's ``identifyOverExpressedGenes`` uses ``presto::wilcoxauc`` (Korsunsky
et al.), a vectorized rank-sum test. Reproducing it — rather than delegating to
scanpy's ``rank_genes_groups`` — is what makes the over-expressed gene set (and
hence the downstream L–R pair set) match CellChat: presto's rank-based p-values
and its ``logFC`` definition differ materially from scanpy's, selecting ~10×
fewer genes on the same data.

For each gene and each cell group (one-vs-rest), :func:`wilcoxauc` computes, from
the log-normalized expression:

- ``avgExpr``  — mean expression **in** the group (log space)
- ``logFC``    — ``mean_in − mean_out`` (difference of log-space means)
- ``auc``      — Mann–Whitney U / (n_in · n_out) = rank-based effect size
- ``pval``     — two-sided Wilcoxon rank-sum normal approximation with tie
                 correction (identical statistic to presto)
- ``padj``     — Benjamini–Hochberg FDR
- ``pct_in`` / ``pct_out`` — % of cells with nonzero expression (0–100)

Ranking is done once per gene over all cells (shared across groups), so the cost
is ``O(G · N log N)`` for the sort plus ``O(G · K)`` for the per-group sums —
the same vectorized strategy as presto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm


def _rankdata_columns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average ranks per column plus per-column tie-correction term.

    Returns
    -------
    ranks : (N, G) float array
        Average ranks (1..N) of each cell within each gene column.
    tie_term : (G,) float array
        ``sum(t^3 - t)`` over tie groups per column, for the rank-sum variance.
    """
    N, G = X.shape
    ranks = np.empty((N, G), dtype=float)
    tie_term = np.zeros(G, dtype=float)
    order = np.argsort(X, axis=0, kind="mergesort")
    for j in range(G):
        col = X[:, j]
        o = order[:, j]
        sorted_col = col[o]
        # average ranks with tie handling
        r = np.empty(N, dtype=float)
        i = 0
        pos = 1
        while i < N:
            k = i
            while k + 1 < N and sorted_col[k + 1] == sorted_col[i]:
                k += 1
            avg = (pos + (pos + (k - i))) / 2.0  # mean of consecutive ranks
            r[i:k + 1] = avg
            t = k - i + 1
            if t > 1:
                tie_term[j] += t ** 3 - t
            pos += t
            i = k + 1
        ranks[o, j] = r
    return ranks, tie_term


def wilcoxauc(
    X,
    groups,
    *,
    var_names,
) -> pd.DataFrame:
    """One-vs-rest Wilcoxon rank-sum + AUC per gene per group (presto port).

    Parameters
    ----------
    X
        ``cells × genes`` log-normalized matrix (dense or sparse).
    groups
        Length-``N`` group labels (array/Series/Categorical).
    var_names
        Gene names labelling the columns of ``X``.

    Returns
    -------
    pd.DataFrame
        One row per (gene, group): columns ``features``, ``group``, ``avgExpr``,
        ``logFC``, ``auc``, ``pval``, ``padj``, ``pct_in``, ``pct_out``.
    """
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    N, G = X.shape
    cat = groups if isinstance(groups, pd.Categorical) else pd.Categorical(groups)
    levels = list(cat.categories)
    codes = cat.codes
    if len(levels) < 2:
        raise ValueError(
            "one-vs-rest differential expression needs at least 2 cell groups; "
            f"got {len(levels)}. Provide a groupby with multiple categories."
        )

    ranks, tie_term = _rankdata_columns(X)          # (N,G), (G,)
    nonzero = X > 0                                  # for pct
    # per-group means / rank-sums via matrix mult against a group indicator
    K = len(levels)
    ind = np.zeros((N, K), dtype=float)
    ind[np.arange(N), codes] = 1.0
    n_in = ind.sum(axis=0)                           # (K,)
    sum_expr = ind.T @ X                             # (K,G) group expr sums
    sum_rank = ind.T @ ranks                         # (K,G) group rank sums
    sum_nz = ind.T @ nonzero.astype(float)           # (K,G) nonzero counts
    total_expr = X.sum(axis=0)                       # (G,)
    total_nz = nonzero.sum(axis=0)                   # (G,)

    rows = []
    for k, lvl in enumerate(levels):
        n1 = n_in[k]
        n2 = N - n1
        if n1 == 0 or n2 == 0:
            continue
        mean_in = sum_expr[k] / n1
        mean_out = (total_expr - sum_expr[k]) / n2
        logfc = mean_in - mean_out
        # Mann-Whitney U from the in-group rank sum (presto compute_ustat)
        R1 = sum_rank[k]
        U = R1 - n1 * (n1 + 1) / 2.0
        auc = U / (n1 * n2)
        # presto compute_pval: normal approximation WITH continuity correction
        #   z = U - 0.5*n1*n2 ; z = z - sign(z)*0.5
        #   usigma = sqrt( n1*n2 * (N^3-N - sum(t^3-t)) / (12*(N^2-N)) )
        n1n2 = n1 * n2
        z = U - 0.5 * n1n2
        z = z - np.sign(z) * 0.5                       # continuity correction
        var = n1n2 * ((N ** 3 - N) - tie_term) / (12.0 * (N ** 2 - N))
        sigma = np.sqrt(np.maximum(var, 1e-12))
        z = z / sigma
        pval = 2.0 * norm.sf(np.abs(z))
        pct_in = 100.0 * sum_nz[k] / n1
        pct_out = 100.0 * (total_nz - sum_nz[k]) / n2
        df = pd.DataFrame({
            "features": var_names,
            "group": str(lvl),
            "avgExpr": mean_in,
            "logFC": logfc,
            "auc": auc,
            "pval": pval,
            "pct_in": pct_in,
            "pct_out": pct_out,
        })
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    # Benjamini-Hochberg FDR across the whole table (presto adjusts per call)
    out["padj"] = _bh_fdr(out["pval"].to_numpy())
    return out


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest p downward
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(ranked, 0, 1)
    return out
