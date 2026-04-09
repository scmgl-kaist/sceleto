from __future__ import annotations

from typing import Any

from ._simple import MarkersSimple
from ._gene_filter import GeneFilter
from ._hierarchy import hierarchy as _hierarchy, HierarchyRun


def simple(adata, groupby: str, **kwargs) -> MarkersSimple:
    """Simple (non-graph) marker workflow."""
    return MarkersSimple(adata, groupby, **kwargs)


# backward compat
classic = simple


def marker(
    adata,
    groupby: str,
    *,
    k: int = 5,
    thres_fc: float = 3.0,
    **kwargs: Any,
):
    """Graph-based marker workflow (one-word entry point)."""
    from .graph import run_marker_graph
    return run_marker_graph(
        adata,
        groupby=groupby,
        k=k,
        thres_fc=thres_fc,
        **kwargs,
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
    return ["simple", "classic", "marker", "hierarchy", "sweep_fc", "GeneFilter"]


__all__ = ["simple", "classic", "marker", "hierarchy", "sweep_fc", "GeneFilter"]
