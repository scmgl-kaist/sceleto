"""FC threshold sweep and suggestion for graph-based marker detection.

Helps determine an appropriate thres_fc by sweeping across thresholds
and tracking how many edges lose all marker genes at each level.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def _run_sweep(
    df: pd.DataFrame,
    total_edges: int,
    thresholds: np.ndarray,
    gt_in_data: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Internal: sweep thresholds on a pre-computed edge-gene DataFrame."""
    rows = []
    for t in thresholds:
        df_t = df[df["fc"] >= t]
        n_pairs = len(df_t)
        n_genes = df_t["gene"].nunique()
        edges_covered = set(df_t["edge"])
        n_covered = len(edges_covered)
        n_uncovered = total_edges - n_covered

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
    thresholds: Union[str, Sequence[float]] = "auto",
    ground_truth: Optional[Sequence[str]] = None,
    use_raw: bool = True,
    n_steps: int = 10,
    **ctx_kwargs,
) -> pd.DataFrame:
    """Sweep FC thresholds and summarize edge/gene statistics.

    Parameters
    ----------
    adata
        AnnData object.
    groupby
        Column in adata.obs.
    thresholds
        "auto" for two-phase adaptive sweep, or a list of explicit values.
    ground_truth
        Optional list of known marker genes. If provided, their survival
        across thresholds is tracked.
    use_raw
        Whether to use adata.raw.
    n_steps
        Number of steps per phase when thresholds="auto".
    **ctx_kwargs
        Passed to build_context (e.g. k, exclude, min_cells_per_group).

    Returns
    -------
    DataFrame with columns:
        threshold, n_pairs, n_genes, n_edges_total, n_edges_covered,
        n_edges_uncovered, markers_per_edge_median, markers_per_edge_min,
        markers_per_edge_max.
        If ground_truth is provided: gt_genes_surviving, gt_edges_covered.
    """
    from ._context import build_context
    from ._metrics import compute_fc_delta
    import scanpy as sc
    import matplotlib.pyplot as _plt

    # Ensure PAGA exists
    paga = getattr(adata, "uns", {}).get("paga", None)
    need_paga = paga is None or "connectivities" not in paga
    if not need_paga:
        n_groups = adata.obs[groupby].nunique()
        if paga["connectivities"].shape[0] != n_groups:
            need_paga = True
    if need_paga:
        if "neighbors" not in getattr(adata, "uns", {}):
            sc.pp.neighbors(adata)
        sc.tl.paga(adata, groups=groupby)
        try:
            sc.pl.paga_compare(adata, show=False)
        except Exception:
            sc.pl.paga(adata, show=False)
        _plt.close("all")

    ctx = build_context(adata, groupby=groupby, use_raw=use_raw, **ctx_kwargs)

    # Compute all edge-gene pairs at FC >= 1.0 (baseline)
    df = compute_fc_delta(ctx, thres_fc=1.0, eps=1e-3)
    df["edge"] = df["start"].astype(str) + "->" + df["end"].astype(str)
    total_edges = df["edge"].nunique()

    # Ground truth genes present in data
    gt_in_data = None
    if ground_truth is not None:
        all_genes = set(df["gene"])
        gt_in_data = [g for g in ground_truth if g in all_genes]

    if isinstance(thresholds, str) and thresholds == "auto":
        # Phase 1: coarse sweep from 1.0 to 95th percentile
        fc_values = df["fc"].values
        hi = max(np.percentile(fc_values, 95), 2.0)
        coarse = np.linspace(1.0, hi, n_steps)
        coarse_df = _run_sweep(df, total_edges, coarse, gt_in_data)

        # Find where uncovered edges first appear
        first_uncovered_idx = coarse_df[coarse_df["n_edges_uncovered"] > 0].index
        if len(first_uncovered_idx) == 0:
            # No uncovered edges even at 95th pct — just return coarse
            return coarse_df

        # Phase 2: fine sweep around the transition point
        idx = first_uncovered_idx[0]
        fine_lo = float(coarse_df.loc[max(idx - 1, 0), "threshold"])
        fine_hi = float(coarse_df.loc[min(idx + 1, len(coarse_df) - 1), "threshold"])

        fine = np.linspace(fine_lo, fine_hi, n_steps)
        fine_df = _run_sweep(df, total_edges, fine, gt_in_data)

        # Merge: fine + coarse (beyond fine range)
        result = pd.concat([
            fine_df,
            coarse_df[coarse_df["threshold"] > fine_hi],
        ], ignore_index=True).sort_values("threshold").reset_index(drop=True)
        return result

    # Explicit thresholds
    thresholds = np.asarray(thresholds, dtype=float)
    return _run_sweep(df, total_edges, thresholds, gt_in_data)


def suggest_fc_threshold(summary_df: pd.DataFrame) -> float:
    """Suggest FC threshold: the highest value before any edge becomes uncovered.

    Parameters
    ----------
    summary_df
        Output of sweep_fc_threshold().

    Returns
    -------
    Suggested threshold value.
    """
    covered = summary_df[summary_df["n_edges_uncovered"] == 0]
    if len(covered) == 0:
        # All thresholds have uncovered edges; return the lowest
        return float(summary_df["threshold"].iloc[0])
    return float(covered["threshold"].iloc[-1])


def plot_fc_threshold(
    summary_df: pd.DataFrame,
    *,
    suggested: Optional[float] = None,
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
        ax.set_xlabel("FC threshold")
        ax.set_ylabel("Count")
        ax.set_title("Unique candidate genes")
        _add_suggested(ax)

        ax = axes[0, 1]
        ax.plot(thresholds, summary_df["n_edges_uncovered"], "x--", color="red", markersize=5)
        ax.set_xlabel("FC threshold")
        ax.set_ylabel("# edges")
        ax.set_title("Uncovered edges")
        _add_suggested(ax)

        ax = axes[1, 0]
        ax.plot(thresholds, summary_df["gt_genes_surviving"], "D-", color="purple", markersize=4)
        ax.set_xlabel("FC threshold")
        ax.set_ylabel("# genes")
        gt_total = summary_df["gt_genes_total"].iloc[0]
        ax.set_title(f"Ground truth survival ({gt_total} genes)")
        _add_suggested(ax)

        ax = axes[1, 1]
        ax.plot(thresholds, summary_df["gt_edges_covered"], "^-", color="crimson", markersize=4)
        n_total = summary_df["n_edges_total"].iloc[0]
        ax.axhline(n_total, color="gray", ls=":", alpha=0.5, label=f"total={n_total}")
        ax.set_xlabel("FC threshold")
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
        ax.set_xlabel("FC threshold")
        ax.set_ylabel("Count")
        ax.set_title("Unique candidate genes")
        _add_suggested(ax)

        ax = axes[1]
        ax.plot(thresholds, summary_df["n_edges_uncovered"], "x--", color="red", markersize=5)
        ax.set_xlabel("FC threshold")
        ax.set_ylabel("# edges")
        ax.set_title("Uncovered edges")
        _add_suggested(ax)

    plt.suptitle("FC threshold sweep", fontsize=13, y=1.01)
    plt.tight_layout()

    if save:
        plt.savefig(save, bbox_inches="tight", format="pdf", dpi=300)
    if show:
        plt.show()

    return fig, axes
