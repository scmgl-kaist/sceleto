from __future__ import annotations

from typing import Any, Callable, List, Optional, Union

import pandas as pd

from ._simple import MarkersSimple
from ._gene_filter import GeneFilter
from ._hierarchy import hierarchy as _hierarchy, HierarchyRun


def simple(adata, groupby: str, **kwargs) -> MarkersSimple:
    """Simple (non-graph) marker workflow."""
    return MarkersSimple(adata, groupby, **kwargs)



def marker(
    adata,
    groupby: str,
    *,
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
):
    """Graph-based marker workflow (one-word entry point).

    Wraps :func:`sceleto.markers.graph.run_marker_graph` with the same parameters.
    When *batch_key* is provided, a Welch's t-test filter is automatically applied
    after FC filtering (genes that can be tested but fail p-value are dropped).
    """
    from .graph import run_marker_graph
    return run_marker_graph(
        adata,
        groupby=groupby,
        thres_fc=thres_fc,
        specific_A=specific_A,
        specific_B=specific_B,
        specific_only_high_markers=specific_only_high_markers,
        specific_score_col=specific_score_col,
        specific_score_fn=specific_score_fn,
        use_raw=use_raw,
        k=k,
        exclude=exclude,
        min_cells_per_group=min_cells_per_group,
        min_expr_cells_per_gene=min_expr_cells_per_gene,
        eps=eps,
        min_mean_any=min_mean_any,
        min_mean_high=min_mean_high,
        min_frac_high=min_frac_high,
        max_mean_low=max_mean_low,
        min_nexpr_any=min_nexpr_any,
        fc_cutoff=fc_cutoff,
        label_k=label_k,
        sigma_method=sigma_method,
        min_gap=min_gap,
        min_margin=min_margin,
        level=level,
        bidirectional=bidirectional,
        node_size_scale=node_size_scale,
        batch_key=batch_key,
        batch_min_cells=batch_min_cells,
        batch_ttest_alpha=batch_ttest_alpha,
        batch_ttest_min_batches=batch_ttest_min_batches,
    )


def hierarchy(adata, marker_runs, **kwargs) -> HierarchyRun:
    """Wrapper for cross-resolution hierarchy workflow."""
    return _hierarchy(adata, marker_runs, **kwargs)


def sweep_fc(adata, groupby: str, **kwargs):
    """Sweep FC thresholds to help determine thres_fc.

    See :func:`sceleto.markers.graph.sweep_fc_threshold` for full docs.
    """
    from .graph import sweep_fc_threshold
    return sweep_fc_threshold(adata, groupby, **kwargs)


def __dir__():
    return ["simple", "marker", "hierarchy", "sweep_fc", "GeneFilter"]


__all__ = ["simple", "marker", "hierarchy", "sweep_fc", "GeneFilter"]
