from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd




@dataclass
class MarkerGraphRun:

    """Container for one-step marker-graph pipeline results.

    Notes
    -----
    - Keeps intermediate artifacts for debugging and inspection.
    """
    # Core artifacts
    ctx: Any
    edge_gene_df: pd.DataFrame
    edge_fc: pd.DataFrame
    edge_delta: pd.DataFrame
    labels: Any
    note_df: pd.DataFrame

    # Graph + viz
    G: Any
    pos: Any
    gene_edge_fc: Dict[str, Dict[Tuple[object, object], float]]
    gene_to_edges: Dict[str, List[str]]
    viz: Any

    # Specific marker outputs
    specific_ranking_df: pd.DataFrame
    _marker_log: Dict[str, List[str]]

    # Batch key (None if not provided)
    batch_key: Optional[str] = None

    # FC threshold sweep (None if thres_fc was not "auto")
    sweep_df: Optional[pd.DataFrame] = None
    suggested_thres_fc: Optional[float] = None

    def plot_fc_threshold(self, **kwargs):
        """Plot FC threshold sweep results. Only available if thres_fc="auto" was used."""
        if self.sweep_df is None:
            raise ValueError("No sweep data. Re-run with thres_fc='auto'.")
        from ._threshold import plot_fc_threshold
        return plot_fc_threshold(self.sweep_df, suggested=self.suggested_thres_fc, **kwargs)

    def plot_gene_edges_fc(self, gene: str, **kwargs):
        return self.viz.plot_gene_edges_fc(gene, **kwargs)

    def plot_gene_levels_with_edges(self, gene: str, level: Optional[int] = None, **kwargs):
        return self.viz.plot_gene_levels_with_edges(gene, level=level, **kwargs)

    def plot_highlight_edges(self, edges, **kwargs):
        return self.viz.plot_highlight_edges(edges, **kwargs)

    @property
    def markers(self) -> Dict[str, List[str]]:
        """Per-group marker gene lists, ranked by specificity score."""
        return self._marker_log

    def plot(self, n_top: int = 10, **kwargs):
        """Dotplot of ``self.markers`` against ``self.ctx.groupby``.

        Genes are shown with per-cluster bracket labels. ``n_top`` controls
        how many genes per group are included. Remaining kwargs are forwarded
        to ``sceleto.dotplot``.
        """
        from sceleto.dotplot import dotplot
        var_names = {k: v[:n_top] for k, v in self.markers.items() if v}
        return dotplot(self.ctx.adata, var_names, self.ctx.groupby, **kwargs)

    def batch_mean_detail(
        self,
        adata,
        gene: str,
        group: str,
    ):
        """Return per-batch mean expression for a specific (gene, group).

        Parameters
        ----------
        adata : AnnData
            The same AnnData used in :func:`run_marker_graph`.
        gene : str
            Marker gene name.
        group : str
            Cluster where the gene is highly expressed.

        Returns
        -------
        DataFrame with columns:
            ``edge_start``, ``edge_end``, ``batch``,
            ``mean_start``, ``mean_end``, ``n_cells_start``, ``n_cells_end``.
        """
        if self.batch_key is None:
            raise ValueError(
                "No batch data. Re-run run_marker_graph() with batch_key='...'."
            )
        from ._batch import get_batch_mean_detail

        return get_batch_mean_detail(
            adata, self.ctx, self.edge_gene_df, self.batch_key,
            gene, group,
        )



def run_marker_graph(
    adata: Any,
    *,
    groupby: str,
    thres_fc: Union[float, str] = "auto",
    # Specific ranking params
    specific_A: float = 1.0,
    specific_B: float = 0.5,
    specific_only_high_markers: bool = True,
    specific_score_col: str = "specific_weight",
    specific_score_fn: Optional[Callable[[pd.DataFrame], object]] = None,
    # Context defaults
    use_raw: bool = True,
    k: int = 5,
    exclude: Optional[List[str]] = None,
    min_cells_per_group: int = 0,
    min_expr_cells_per_gene: int = 0,
    # FC/delta defaults
    eps: float = 1e-3,
    min_mean_any: float = 0.05,
    min_mean_high: float = 0.5,
    min_frac_high: float = 0.2,
    max_mean_low: float = 0.2,
    min_nexpr_any: int = 0,
    # Labeling defaults
    fc_cutoff: Optional[float] = None,
    label_k: float = 2.0,
    sigma_method: str = "sd",
    min_gap: float = 0.2,
    min_margin: float = 0.0,
    level: int = 3,
    # Graph/Viz defaults
    bidirectional: bool = True,
    node_size_scale: float = 10.0,
    # Batch t-test (activated automatically when batch_key is provided)
    batch_key: Optional[str] = None,
    batch_min_cells: int = 5,
    batch_ttest_alpha: float = 0.05,
    batch_ttest_min_batches: int = 3,
) -> MarkerGraphRun:
    """One-step wrapper: context -> edge metrics -> labels -> viz -> specific marker discovery"""
    from ._context import build_context
    from ._metrics import compute_fc_delta, edge_gene_df_to_matrices, build_gene_edge_fc_from_edge_gene_df
    from ._labels import label_levels, labels_to_note_df
    from ._viz import GraphVizContext, build_graph_and_pos_from_ctx
    from ._local import (
        build_local_marker_inputs,
        weight_local_prioritized,
    )

    import scanpy as sc
    import matplotlib.pyplot as plt

    # --- Ensure PAGA exists and matches the current groupby ---
    paga = getattr(adata, "uns", {}).get("paga", None)
    need_paga = paga is None or "connectivities" not in paga

    if not need_paga:
        # Re-run if connectivities shape doesn't match the groupby categories
        n_groups = adata.obs[groupby].nunique()
        if paga["connectivities"].shape[0] != n_groups:
            need_paga = True

    if need_paga:
        # PAGA requires neighbors graph
        if "neighbors" not in getattr(adata, "uns", {}):
            sc.pp.neighbors(adata)
        sc.tl.paga(adata, groups=groupby)

    # --- Ensure PAGA positions exist; populate if missing or stale ---
    paga = adata.uns.get("paga", {})
    if "pos" not in paga or need_paga:
        try:
            sc.pl.paga_compare(adata, show=False)
        except Exception:
            sc.pl.paga(adata, show=False)
        plt.close("all")

    # --- Auto threshold ---
    sweep_df = None
    suggested_thres_fc = None

    if isinstance(thres_fc, str) and thres_fc == "auto":
        from ._threshold import sweep_fc_threshold, suggest_fc_threshold
        sweep_df = sweep_fc_threshold(adata, groupby, use_raw=use_raw,
                                       k=k, exclude=exclude,
                                       min_cells_per_group=min_cells_per_group,
                                       min_expr_cells_per_gene=min_expr_cells_per_gene)
        suggested_thres_fc = suggest_fc_threshold(sweep_df)
        thres_fc = suggested_thres_fc
        print(f"  Auto thres_fc: {thres_fc:.2f}")

    if fc_cutoff is None:
        fc_cutoff = thres_fc

    ctx = build_context(
        adata,
        groupby=groupby,
        use_raw=use_raw,
        exclude=exclude,
        min_cells_per_group=min_cells_per_group,
        min_expr_cells_per_gene=min_expr_cells_per_gene,
        k=k,
    )

    edge_gene_df = compute_fc_delta(
        ctx,
        thres_fc=thres_fc,
        eps=eps,
        min_mean_any=min_mean_any,
        min_mean_high=min_mean_high,
        min_frac_high=min_frac_high,
        max_mean_low=max_mean_low,
        min_nexpr_any=min_nexpr_any,
    )
    # --- Batch t-test filter (activated when batch_key is provided) ---
    if batch_key is not None:
        from ._batch import filter_edge_gene_df_by_ttest
        edge_gene_df = filter_edge_gene_df_by_ttest(
            adata, ctx, edge_gene_df, batch_key,
            use_raw=use_raw, min_cells=batch_min_cells,
            min_batches=batch_ttest_min_batches, alpha=batch_ttest_alpha, eps=eps,
        )

    edge_fc, edge_delta = edge_gene_df_to_matrices(edge_gene_df)

    labels = label_levels(
        ctx,
        edge_gene_df,
        fc_cutoff=float(fc_cutoff),
        k=label_k,
        sigma_method=sigma_method,  # type: ignore[arg-type]
        min_gap=min_gap,
        min_margin=min_margin,
    )
    note_df = labels_to_note_df(ctx, labels, level=level)  # type: ignore[arg-type]

    G, pos = build_graph_and_pos_from_ctx(ctx, bidirectional=bidirectional)
    gene_edge_fc = build_gene_edge_fc_from_edge_gene_df(edge_fc, G=G)

    sub = edge_gene_df[edge_gene_df["fc"] >= float(thres_fc)]
    gene_to_edges: Dict[str, List[str]] = {}
    if len(sub) > 0:
        for g, sdf in sub.groupby("gene"):
            gene_to_edges[str(g)] = (
                sdf["start"].astype(str) + "->" + sdf["end"].astype(str)
            ).tolist()

    viz = GraphVizContext(
        G=G,
        ctx=ctx,
        note_df=note_df,
        labels=labels,
        gene_edge_fc=gene_edge_fc,
        gene_to_edges=gene_to_edges,
        node_size_scale=node_size_scale,
    )

    specific_ranking_df: pd.DataFrame
    _marker_log: Dict[str, List[str]]

    specific_inputs_df = build_local_marker_inputs(
        ctx=ctx,
        labels=labels,
        note_df=note_df,
        edge_fc=edge_fc,
        edge_delta=edge_delta,
        only_high_markers=specific_only_high_markers,
    )

    specific_ranking_df = specific_inputs_df.copy()

    if specific_score_fn is not None:
        specific_ranking_df[specific_score_col] = specific_score_fn(specific_ranking_df)
    else:
        specific_ranking_df[specific_score_col] = weight_local_prioritized(
            specific_ranking_df, A=specific_A, B=specific_B
        )

    specific_ranking_df = specific_ranking_df.sort_values(
        ["group", specific_score_col], ascending=[True, False]
    )

    _marker_log = {str(g): [] for g in ctx.groups}
    for g, sdf in specific_ranking_df.groupby("group", sort=False):
        _marker_log[str(g)] = sdf["gene"].astype(str).tolist()

    return MarkerGraphRun(
        ctx=ctx,
        edge_gene_df=edge_gene_df,
        edge_fc=edge_fc,
        edge_delta=edge_delta,
        labels=labels,
        note_df=note_df,
        G=G,
        pos=pos,
        gene_edge_fc=gene_edge_fc,
        gene_to_edges=gene_to_edges,
        viz=viz,
        specific_ranking_df=specific_ranking_df,
        _marker_log=_marker_log,
        batch_key=batch_key,
        sweep_df=sweep_df,
        suggested_thres_fc=suggested_thres_fc,
    )
