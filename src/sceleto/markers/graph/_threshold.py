"""FC threshold sweep and suggestion for graph-based marker detection.

Helps determine an appropriate thres_fc by sweeping across thresholds
and tracking how many edges lose all marker genes at each level.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def _run_sweep(
    df: pd.DataFrame,
    total_edges: int,
    thresholds: np.ndarray,
    gt_in_data: Optional[List[str]] = None,
    edge_metric: Literal["fc", "delta"] = "fc",
    coverable_edges: Optional[int] = None,
) -> pd.DataFrame:
    """Internal: sweep thresholds on a pre-computed edge-gene DataFrame.

    ``coverable_edges`` is the number of edges that retain at least one
    candidate marker in *df* at the metric floor (i.e. edges that *can* be
    covered by some threshold). When a batch t-test gate has removed every
    marker of an edge, that edge is a *genuine orphan* and is excluded from the
    coverage target via ``n_edges_uncovered_coverable`` — otherwise a single
    unmarkerable edge would drag the suggested threshold down to the floor.
    Defaults to ``total_edges`` (no genuine orphans), preserving old behaviour.
    """
    if coverable_edges is None:
        coverable_edges = total_edges
    n_genuine_orphan = total_edges - coverable_edges

    rows = []
    for t in thresholds:
        df_t = df[df[edge_metric] >= t]
        n_pairs = len(df_t)
        n_genes = df_t["gene"].nunique()
        edges_covered = set(df_t["edge"])
        n_covered = len(edges_covered)
        n_uncovered = total_edges - n_covered
        # covered edges are always a subset of coverable edges, so this is >= 0
        n_uncovered_coverable = coverable_edges - n_covered

        if n_pairs > 0:
            mpe = df_t.groupby("edge")["gene"].nunique()
            med, mn, mx = float(mpe.median()), int(mpe.min()), int(mpe.max())
        else:
            med, mn, mx = 0.0, 0, 0

        row = {
            "threshold": float(t),
            "n_pairs": n_pairs,
            "n_genes": n_genes,
            "n_edges_total": total_edges,
            "n_edges_covered": n_covered,
            "n_edges_uncovered": n_uncovered,
            "n_edges_coverable": coverable_edges,
            "n_edges_genuine_orphan": n_genuine_orphan,
            "n_edges_uncovered_coverable": n_uncovered_coverable,
            "markers_per_edge_median": med,
            "markers_per_edge_min": mn,
            "markers_per_edge_max": mx,
        }

        if gt_in_data is not None:
            df_gt = df_t[df_t["gene"].isin(gt_in_data)]
            row["gt_genes_surviving"] = df_gt["gene"].nunique()
            row["gt_genes_total"] = len(gt_in_data)
            row["gt_edges_covered"] = df_gt["edge"].nunique()

        rows.append(row)

    return pd.DataFrame(rows)


def sweep_fc_threshold(
    adata,
    groupby: str,
    *,
    edge_metric: Literal["fc", "delta"] = "fc",
    thresholds: Union[str, Sequence[float]] = "auto",
    ground_truth: Optional[Sequence[str]] = None,
    use_raw: bool = True,
    n_steps: int = 10,
    # Expression filters (forwarded to compute_fc_delta so the sweep sees the
    # same edges as the main pipeline)
    eps: float = 1e-3,
    min_mean_any: float = 0.05,
    min_mean_high: float = 0.5,
    min_frac_high: float = 0.2,
    max_mean_low: float = 0.2,
    min_nexpr_any: int = 0,
    # Batch t-test (applied as a candidate gate before sweeping, so the
    # suggested threshold is coverage-aware w.r.t. the t-test survivors)
    batch_key: Optional[str] = None,
    batch_min_cells: int = 5,
    batch_ttest_alpha: float = 0.05,
    batch_ttest_min_batches: int = 3,
    **ctx_kwargs,
) -> pd.DataFrame:
    """Sweep edge-metric thresholds and summarize edge/gene statistics.

    Despite the legacy name, this also handles ``edge_metric="delta"``.

    Parameters
    ----------
    adata
        AnnData object.
    groupby
        Column in adata.obs.
    edge_metric
        Which column to sweep: "fc" (ratio) or "delta" (subtraction).
    thresholds
        "auto" for two-phase adaptive sweep, or a list of explicit values.
    ground_truth
        Optional list of known marker genes. If provided, their survival
        across thresholds is tracked.
    use_raw
        Whether to use adata.raw.
    n_steps
        Number of steps per phase when thresholds="auto".
    batch_key
        If provided, apply the batch pseudobulk Welch t-test as a candidate
        gate *before* sweeping. Edges are then covered/uncovered relative to
        the t-test survivors, so the suggested threshold accounts for genes
        the t-test will drop downstream. ``n_edges_total`` is still counted on
        the pre-t-test (expression-passing) edge set, so edges left orphaned
        by the t-test show up as uncovered at every threshold.
    batch_min_cells, batch_ttest_alpha, batch_ttest_min_batches
        Forwarded to :func:`sceleto.markers.graph._batch.filter_edge_gene_df_by_ttest`.
    **ctx_kwargs
        Passed to build_context (e.g. k, exclude, min_cells_per_group).

    Returns
    -------
    DataFrame with columns:
        threshold, n_pairs, n_genes, n_edges_total, n_edges_covered,
        n_edges_uncovered, n_edges_coverable, n_edges_genuine_orphan,
        n_edges_uncovered_coverable, markers_per_edge_median,
        markers_per_edge_min, markers_per_edge_max.
        ``n_edges_coverable`` is the edges that still have a marker after the
        t-test gate; ``n_edges_genuine_orphan`` = total - coverable; and
        ``n_edges_uncovered_coverable`` counts only coverable edges lost at a
        threshold (the target used by suggest_fc_threshold). Without a batch
        gate coverable == total and the coverable column equals n_edges_uncovered.
        If ground_truth is provided: gt_genes_surviving, gt_edges_covered.
    """
    from ._context import build_context
    from ._metrics import compute_fc_delta
    import scanpy as sc
    import matplotlib.pyplot as _plt

    if edge_metric not in ("fc", "delta"):
        raise ValueError(f"edge_metric must be 'fc' or 'delta', got {edge_metric!r}")

    # Validate k and prepare graph inputs (mirrors run_marker_graph).
    k = ctx_kwargs.get("k", 5)
    if k == "all":
        if "X_umap" not in getattr(adata, "obsm", {}):
            raise ValueError(
                "k='all' requires adata.obsm['X_umap'] for cluster positions. "
                "Run sc.tl.umap(adata) first."
            )
    elif isinstance(k, int):
        if "neighbors" not in getattr(adata, "uns", {}):
            raise ValueError(
                f"k={k} (PAGA trim) requires precomputed neighbors. "
                "Run sc.pp.neighbors(adata) first, or set k='all' to skip PAGA."
            )
        paga = getattr(adata, "uns", {}).get("paga", None)
        need_paga = paga is None or "connectivities" not in paga
        if not need_paga:
            n_groups = adata.obs[groupby].nunique()
            if paga["connectivities"].shape[0] != n_groups:
                need_paga = True
        if need_paga:
            sc.tl.paga(adata, groups=groupby)
            try:
                sc.pl.paga_compare(adata, show=False)
            except Exception:
                sc.pl.paga(adata, show=False)
            _plt.close("all")
    else:
        raise ValueError(f"k must be int or 'all', got {k!r}")

    ctx = build_context(adata, groupby=groupby, use_raw=use_raw, **ctx_kwargs)

    # Baseline: keep all edges that pass expression filters (no metric threshold).
    # For FC mode use thres_fc=1.0; for delta mode use thres_delta=0.0.
    expr_filter_kw = dict(
        eps=eps,
        min_mean_any=min_mean_any,
        min_mean_high=min_mean_high,
        min_frac_high=min_frac_high,
        max_mean_low=max_mean_low,
        min_nexpr_any=min_nexpr_any,
    )
    if edge_metric == "fc":
        df = compute_fc_delta(ctx, edge_metric="fc", thres_fc=1.0, **expr_filter_kw)
        lo_default = 1.0
    else:
        df = compute_fc_delta(ctx, edge_metric="delta", thres_delta=0.0, **expr_filter_kw)
        lo_default = 0.0

    df["edge"] = df["start"].astype(str) + "->" + df["end"].astype(str)
    # n_edges_total is fixed on the pre-t-test (expression-passing) edge set so
    # that edges orphaned by the t-test remain visible as uncovered.
    total_edges = df["edge"].nunique()

    # Ground truth genes present in data (measured on the pre-t-test pool so
    # GT genes that fail the t-test are still tracked as dropping out).
    gt_in_data = None
    if ground_truth is not None:
        all_genes = set(df["gene"])
        gt_in_data = [g for g in ground_truth if g in all_genes]

    # Batch t-test candidate gate: drop non-reproducible (gene, edge) rows
    # before the threshold sweep so suggest_fc_threshold measures coverage on
    # the same survivor pool the main pipeline will use. The t-test p-value is a
    # property of (gene, edge) independent of the metric value, so gating here
    # and thresholding later commute with the main pipeline's order.
    if batch_key is not None:
        from ._batch import filter_edge_gene_df_by_ttest
        df = filter_edge_gene_df_by_ttest(
            adata, ctx, df, batch_key,
            use_raw=use_raw, min_cells=batch_min_cells,
            min_batches=batch_ttest_min_batches, alpha=batch_ttest_alpha, eps=eps,
        )

    # Coverable edges = edges that still have >=1 candidate marker at the floor
    # (after the optional t-test gate). Edges the gate emptied out are genuine
    # orphans; the coverage target below excludes them so they can't collapse
    # the suggested threshold. Without a gate this equals total_edges.
    coverable_edges = int(df["edge"].nunique())

    if isinstance(thresholds, str) and thresholds == "auto":
        # Phase 1: coarse sweep from baseline to 95th percentile of the metric
        metric_values = df[edge_metric].values
        if edge_metric == "fc":
            hi = max(np.percentile(metric_values, 95), 2.0)
        else:
            hi = max(np.percentile(metric_values, 95), 2.0 * lo_default + 1e-3)
        coarse = np.linspace(lo_default, hi, n_steps)
        coarse_df = _run_sweep(df, total_edges, coarse, gt_in_data,
                               edge_metric=edge_metric, coverable_edges=coverable_edges)

        # Find where a *coverable* edge first becomes uncovered (genuine orphans
        # are ignored so the fine sweep centres on the real transition).
        first_uncovered_idx = coarse_df[coarse_df["n_edges_uncovered_coverable"] > 0].index
        if len(first_uncovered_idx) == 0:
            # No coverable edge uncovered even at 95th pct — just return coarse
            return coarse_df

        # Phase 2: fine sweep around the transition point
        idx = first_uncovered_idx[0]
        fine_lo = float(coarse_df.loc[max(idx - 1, 0), "threshold"])
        fine_hi = float(coarse_df.loc[min(idx + 1, len(coarse_df) - 1), "threshold"])

        fine = np.linspace(fine_lo, fine_hi, n_steps)
        fine_df = _run_sweep(df, total_edges, fine, gt_in_data,
                             edge_metric=edge_metric, coverable_edges=coverable_edges)

        # Merge: fine + coarse (beyond fine range)
        result = pd.concat([
            fine_df,
            coarse_df[coarse_df["threshold"] > fine_hi],
        ], ignore_index=True).sort_values("threshold").reset_index(drop=True)
        return result

    # Explicit thresholds
    thresholds = np.asarray(thresholds, dtype=float)
    return _run_sweep(df, total_edges, thresholds, gt_in_data,
                      edge_metric=edge_metric, coverable_edges=coverable_edges)


def suggest_fc_threshold(summary_df: pd.DataFrame) -> float:
    """Suggest FC threshold: the highest value before any *coverable* edge is lost.

    Coverage is measured against the coverable edge set when available
    (``n_edges_uncovered_coverable``), i.e. genuine orphans — edges left with no
    marker by a batch t-test gate — are excluded from the target. This prevents
    a single unmarkerable edge from collapsing the suggestion to the floor.
    Falls back to ``n_edges_uncovered`` for sweeps produced without the column.

    Parameters
    ----------
    summary_df
        Output of sweep_fc_threshold().

    Returns
    -------
    Suggested threshold value.
    """
    col = (
        "n_edges_uncovered_coverable"
        if "n_edges_uncovered_coverable" in summary_df.columns
        else "n_edges_uncovered"
    )
    covered = summary_df[summary_df[col] == 0]
    if len(covered) == 0:
        # Every threshold loses a coverable edge; return the lowest
        return float(summary_df["threshold"].iloc[0])
    return float(covered["threshold"].iloc[-1])


def plot_fc_threshold(
    summary_df: pd.DataFrame,
    *,
    suggested: Optional[float] = None,
    edge_metric: Literal["fc", "delta"] = "fc",
    figsize: Optional[Tuple[float, float]] = None,
    save: Optional[str] = None,
    show: bool = True,
):
    """Plot FC threshold sweep results.

    Parameters
    ----------
    summary_df
        Output of sweep_fc_threshold().
    suggested
        If provided, draw a vertical line at this threshold.
        Use suggest_fc_threshold() to get this value.
    figsize
        Figure size.
    save
        Path to save as PDF.
    show
        Whether to call plt.show().

    Returns
    -------
    fig, axes
    """
    import matplotlib.pyplot as plt

    has_gt = "gt_genes_surviving" in summary_df.columns
    thresholds = summary_df["threshold"]

    def _add_suggested(ax):
        if suggested is not None:
            ax.axvline(suggested, color="green", ls="--", alpha=0.7, label=f"suggested={suggested:.1f}")
            ax.legend(fontsize=8)

    if has_gt:
        # 2x2: candidate genes, uncovered edges, gt survival, gt edge coverage
        if figsize is None:
            figsize = (10, 8)
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        ax = axes[0, 0]
        ax.plot(thresholds, summary_df["n_genes"], "s-", color="darkorange", markersize=4)
        ax.set_xlabel(f"{edge_metric.upper()} threshold")
        ax.set_ylabel("Count")
        ax.set_title("Unique candidate genes")
        _add_suggested(ax)

        ax = axes[0, 1]
        ax.plot(thresholds, summary_df["n_edges_uncovered"], "x--", color="red", markersize=5)
        ax.set_xlabel(f"{edge_metric.upper()} threshold")
        ax.set_ylabel("# edges")
        ax.set_title("Uncovered edges")
        _add_suggested(ax)

        ax = axes[1, 0]
        ax.plot(thresholds, summary_df["gt_genes_surviving"], "D-", color="purple", markersize=4)
        ax.set_xlabel(f"{edge_metric.upper()} threshold")
        ax.set_ylabel("# genes")
        gt_total = summary_df["gt_genes_total"].iloc[0]
        ax.set_title(f"Ground truth survival ({gt_total} genes)")
        _add_suggested(ax)

        ax = axes[1, 1]
        ax.plot(thresholds, summary_df["gt_edges_covered"], "^-", color="crimson", markersize=4)
        n_total = summary_df["n_edges_total"].iloc[0]
        ax.axhline(n_total, color="gray", ls=":", alpha=0.5, label=f"total={n_total}")
        ax.set_xlabel(f"{edge_metric.upper()} threshold")
        ax.set_ylabel("# edges")
        ax.set_title("Edges with ground truth markers")
        ax.legend(fontsize=8)
        _add_suggested(ax)
    else:
        # 1x2: candidate genes, uncovered edges
        if figsize is None:
            figsize = (10, 4)
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax = axes[0]
        ax.plot(thresholds, summary_df["n_genes"], "s-", color="darkorange", markersize=4)
        ax.set_xlabel(f"{edge_metric.upper()} threshold")
        ax.set_ylabel("Count")
        ax.set_title("Unique candidate genes")
        _add_suggested(ax)

        ax = axes[1]
        ax.plot(thresholds, summary_df["n_edges_uncovered"], "x--", color="red", markersize=5)
        ax.set_xlabel(f"{edge_metric.upper()} threshold")
        ax.set_ylabel("# edges")
        ax.set_title("Uncovered edges")
        _add_suggested(ax)

    plt.suptitle(f"{edge_metric.upper()} threshold sweep", fontsize=13, y=1.01)
    plt.tight_layout()

    if save:
        plt.savefig(save, bbox_inches="tight", format="pdf", dpi=300)
    if show:
        plt.show()

    return fig, axes
