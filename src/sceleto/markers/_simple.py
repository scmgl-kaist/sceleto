"""Simple (non-graph) marker detection.

Cluster-level marker detection based on mean expression and dropout ratio.
No graph/PAGA structure required — direct comparison across all clusters.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from ._base import MarkersBase

# -- type alias for cluster stats --
ClusterStats = Dict[str, Dict[str, np.ndarray]]


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def _compute_cluster_stats(
    adata,
    group_key: str,
    *,
    exclude: Optional[Sequence[str]] = None,
    min_cells: int = 0,
    min_expr_cells: int = 0,
) -> ClusterStats:
    """Compute per-cluster mean expression, expressing-cell count, and expression fraction.

    Returns
    -------
    dict with keys "counts", "frac_expr", "mean", each mapping group_name -> array(n_genes).
    """
    groups = sorted(set(adata.obs[group_key].astype(str)))
    if exclude:
        ex = set(str(x) for x in exclude)
        groups = [g for g in groups if g not in ex]

    from sceleto._expr import resolve_expression
    X, _, _ = resolve_expression(adata)
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr()

    n_genes = X.shape[1]
    obs_labels = adata.obs[group_key].astype(str).to_numpy()

    means: Dict[str, np.ndarray] = {}
    counts: Dict[str, np.ndarray] = {}
    frac_expr: Dict[str, np.ndarray] = {}

    for g in groups:
        mask = obs_labels == g
        n_cells = int(mask.sum())
        if n_cells < min_cells:
            continue

        X_g = X[mask]

        cnt = np.asarray(X_g.getnnz(axis=0), dtype=np.float64)
        counts[g] = cnt
        frac_expr[g] = cnt / n_cells

        mean = np.asarray(X_g.mean(axis=0)).ravel()
        if min_expr_cells > 0:
            mean[cnt < min_expr_cells] = 0
        means[g] = mean

    return {"counts": counts, "frac_expr": frac_expr, "mean": means}


# ---------------------------------------------------------------------------
# Unified marker finding
# ---------------------------------------------------------------------------

def _collect_gene_values(
    gene_idx: int,
    stats: ClusterStats,
    groups: np.ndarray,
    min_count: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Collect frac_expr and mean for one gene across groups, filtering by count.

    Returns (frac_arr, mean_arr, selected_groups).
    """
    counts, frac, mean = stats["counts"], stats["frac_expr"], stats["mean"]
    frac_vals, mean_vals, selected = [], [], []
    for g in groups:
        if counts[g][gene_idx] < min_count:
            continue
        frac_vals.append(frac[g][gene_idx])
        mean_vals.append(mean[g][gene_idx])
        selected.append(g)
    return np.array(frac_vals), np.array(mean_vals), selected


def _passes_threshold(
    group: str,
    gene_idx: int,
    stats: ClusterStats,
    min_mean: float,
    min_frac: float,
) -> bool:
    """Check if a gene in a group passes mean and frac_expr thresholds."""
    return stats["mean"][group][gene_idx] >= min_mean and stats["frac_expr"][group][gene_idx] >= min_frac


def _find_markers_impl(
    adata,
    stats: ClusterStats,
    *,
    mode: str = "single",
    target_groups: Optional[Sequence[str]] = None,
    gap_thres: float = 0.2,
    min_mean: float = 0.2,
    min_frac: float = 0.2,
    min_count: int = 0,
) -> Dict[str, List[Tuple[str, float]]]:
    """Unified marker detection across modes.

    Parameters
    ----------
    mode
        "single"   — top-1 cluster per gene (largest gap to 2nd highest).
        "multiple" — top cluster per gene (any gap > threshold).
        "negative" — bottom-1 cluster per gene (lowest expression).
        "groups"   — genes distinguishing *target_groups* from the rest.
    target_groups
        Required when mode="groups". Clusters to treat as the "in" group.
    """
    mean_dt = stats["mean"]
    groups = np.array(sorted(mean_dt.keys()))

    markers: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    from sceleto._expr import resolve_expression
    _, var_names, _ = resolve_expression(adata)
    gene_names = var_names

    for gene_idx, gene in enumerate(gene_names):
        _, mean_arr, selected = _collect_gene_values(gene_idx, stats, groups, min_count)
        if len(mean_arr) == 0:
            continue

        mean_norm = mean_arr / np.max(mean_arr)

        if mode == "groups":
            if target_groups is None:
                raise ValueError("target_groups is required for mode='groups'")
            score = _score_groups(mean_norm, selected, target_groups)
            if score is not None and score > gap_thres:
                if all(mean_dt[g][gene_idx] >= min_mean for g in target_groups if g in mean_dt):
                    markers["_group"].append((gene, score))

        elif mode == "negative":
            if len(mean_norm) < 2:
                continue
            order = np.argsort(mean_norm)
            sorted_groups = np.array(selected)[order]
            gaps = np.diff(mean_norm[order])
            if gaps[0] > gap_thres:
                markers[sorted_groups[0]].append((gene, float(gaps[0])))

        else:
            # single / multiple
            if len(mean_norm) == 1:
                if mode == "single" and mean_norm[0] > gap_thres:
                    g = selected[0]
                    if _passes_threshold(g, gene_idx, stats, min_mean, min_frac):
                        markers[g].append((gene, float(mean_norm[0])))
                continue

            order = np.argsort(mean_norm)
            sorted_groups = np.array(selected)[order]
            gaps = np.diff(mean_norm[order])

            if mode == "single":
                if gaps[-1] > gap_thres:
                    top = sorted_groups[-1]
                    if _passes_threshold(top, gene_idx, stats, min_mean, min_frac):
                        markers[top].append((gene, float(gaps[-1])))
            else:  # multiple
                if np.any(gaps > gap_thres):
                    top = sorted_groups[-1]
                    first_gap = float(gaps[np.argmax(gaps > gap_thres)])
                    if _passes_threshold(top, gene_idx, stats, min_mean, min_frac):
                        markers[top].append((gene, first_gap))

    for g in markers:
        markers[g].sort(key=lambda x: -x[1])

    return dict(markers)


def _score_groups(
    mean_norm: np.ndarray,
    selected: List[str],
    target_groups: Sequence[str],
) -> Optional[float]:
    """Score how well *target_groups* separate from the rest in normalized mean."""
    target_set = set(target_groups)
    is_target = np.array([g in target_set for g in selected])
    if not np.any(is_target) or not np.any(~is_target):
        return None
    return float(np.min(mean_norm[is_target]) - np.max(mean_norm[~is_target]))


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def show_marker(
    marker_output,
    celltype: Optional[str] = None,
    n: int = 40,
    result: bool = False,
):
    """Print or return top marker genes sorted by score."""
    source = marker_output[celltype] if celltype is not None else marker_output
    ranked = sorted(source, key=lambda x: -x[1])[:n]
    if result:
        return ranked
    print(ranked)


# ---------------------------------------------------------------------------
# Public class: MarkersSimple
# ---------------------------------------------------------------------------

class MarkersSimple(MarkersBase):
    """Simple (non-graph) marker workflow.

    Detects markers by comparing mean expression and dropout ratio
    across clusters without requiring graph structure.

    Parameters
    ----------
    adata
        AnnData object (must have .raw populated).
    groupby
        Column in adata.obs to group cells by.
    single
        If True (default), use single-top mode; otherwise multi-top.
    gap_thres, min_mean, min_frac, min_count
        Thresholds for marker filtering.
    run_scanpy
        If True, also run scanpy's rank_genes_groups for comparison.
    """

    def __init__(
        self,
        adata,
        groupby: str,
        *,
        single: bool = True,
        gap_thres: float = 0.2,
        min_mean: float = 0.2,
        min_frac: float = 0.2,
        min_count: int = 10,
        run_scanpy: bool = False,
        **kwargs,
    ):
        super().__init__(adata, groupby, **kwargs)

        self._stats = _compute_cluster_stats(adata, groupby)
        adata.uns[f"cdm_{groupby}"] = self._stats

        mode = "single" if single else "multiple"
        self._mks = _find_markers_impl(
            adata, self._stats,
            mode=mode,
            gap_thres=gap_thres,
            min_mean=min_mean,
            min_frac=min_frac,
            min_count=min_count,
        )
        if run_scanpy:
            import scanpy as sc
            sc.tl.rank_genes_groups(adata, groupby)
            adata.uns["rnk"] = pd.DataFrame(adata.uns["rank_genes_groups"]["names"])

    @property
    def markers(self) -> dict:
        """Per-group marker gene lists."""
        return self._mks

    @property
    def stats(self) -> ClusterStats:
        """Access computed cluster statistics."""
        return self._stats

    def show_marker(self, celltype=None, n: int = 40, result: bool = False):
        return show_marker(self._mks, celltype=celltype, n=n, result=result)

    def find_markers_groups(self, groups, **kwargs):
        """Find markers distinguishing specific groups from the rest."""
        return _find_markers_impl(
            self.adata, self._stats, mode="groups", target_groups=groups, **kwargs,
        ).get("_group", [])

    def find_markers_negative(self, **kwargs):
        """Find negative markers (lowest expression in one cluster)."""
        return _find_markers_impl(
            self.adata, self._stats, mode="negative", **kwargs,
        )
