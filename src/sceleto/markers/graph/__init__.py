"""Graph-based marker detection using PAGA connectivity."""

# -- Main entry point --
from ._onestep import MarkerGraphRun, run_marker_graph

# -- Building blocks (for advanced / step-by-step usage) --
from ._context import MarkerContext, build_context
from ._metrics import compute_fc_delta, edge_gene_df_to_matrices
from ._labels import MarkerLabels, label_levels, labels_to_note_df
from ._viz import GraphVizContext, build_graph_and_pos_from_ctx
from ._threshold import sweep_fc_threshold, suggest_fc_threshold, plot_fc_threshold

__all__ = [
    # Main API
    "MarkerGraphRun",
    "run_marker_graph",
    # Building blocks
    "MarkerContext",
    "build_context",
    "compute_fc_delta",
    "edge_gene_df_to_matrices",
    "MarkerLabels",
    "label_levels",
    "labels_to_note_df",
    "GraphVizContext",
    "build_graph_and_pos_from_ctx",
    # Threshold sweep
    "sweep_fc_threshold",
    "suggest_fc_threshold",
    "plot_fc_threshold",
]
