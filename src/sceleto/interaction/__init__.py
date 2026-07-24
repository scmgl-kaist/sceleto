"""Cell–cell communication inference — a Python port of CellChat.

Infers, quantifies, and visualizes ligand–receptor–mediated communication
between cell groups from single-cell data, using the curated CellChatDB
interaction database. Operates on :class:`~anndata.AnnData`; cells are grouped
by an ``.obs`` column.

Quickstart
----------
>>> import sceleto as scl
>>> db = scl.interaction.load_cellcommdb("human")
>>> cc = scl.interaction.CellComm(adata, groupby="cell_type", db=db)
>>> cc.identify_overexpressed()          # DE + over-expressed L-R selection
>>> cc.compute_communication()           # Hill-model prob + permutation p-values
>>> cc.aggregate()                        # cluster x cluster network
>>> df = cc.subset_communication()        # tidy results

Building blocks (reusable independently of the pipeline object):

- :func:`load_cellcommdb`, :func:`load_ppi`, :func:`list_cellcommdb`
- :class:`CellCommDB`  — the interaction database + gene resolution
- :class:`GeneMapper`  — alias-aware symbol normalization (cross-species,
  robust to gene-symbol updates)
"""

from __future__ import annotations

from ._database import (
    CellCommDB,
    load_cellcommdb,
    load_ppi,
    list_cellcommdb,
    ANNOTATIONS,
    SPECIES,
)
from ._genes import GeneMapper, MappingResult
from ._communication import CellComm
from ._analysis import (
    compute_centrality,
    signaling_role,
    signaling_role_pathway,
    network_graph,
)
from ._viz import (
    plot_network_circle,
    plot_heatmap,
    plot_bubble,
    plot_chord,
    plot_rank_net,
    plot_signaling_role_heatmap,
    plot_contribution,
)
from ._viz2 import (
    plot_signaling_role_scatter,
    plot_signaling_role_network,
    plot_chord_gene,
    plot_communication_patterns_river,
    plot_communication_patterns_dot,
    plot_net_embedding_zoom,
    plot_aggregate,
)
from ._patterns import (
    identify_communication_patterns,
    select_k,
    plot_communication_patterns,
)
from ._manifold import (
    compute_net_similarity,
    net_embedding,
    net_clustering,
    plot_net_embedding,
)
from ._spatial import (
    compute_region_distance,
    spatial_probability_factor,
)
from ._multi import MultiCellComm

__all__ = [
    # database
    "CellCommDB",
    "load_cellcommdb",
    "load_ppi",
    "list_cellcommdb",
    "ANNOTATIONS",
    "SPECIES",
    # gene-symbol normalization
    "GeneMapper",
    "MappingResult",
    # core inference
    "CellComm",
    # network analysis
    "compute_centrality",
    "signaling_role",
    "signaling_role_pathway",
    "network_graph",
    # visualization
    "plot_network_circle",
    "plot_heatmap",
    "plot_bubble",
    "plot_chord",
    "plot_rank_net",
    "plot_signaling_role_heatmap",
    "plot_contribution",
    "plot_signaling_role_scatter",
    "plot_signaling_role_network",
    "plot_chord_gene",
    "plot_communication_patterns_river",
    "plot_communication_patterns_dot",
    "plot_net_embedding_zoom",
    "plot_aggregate",
    # NMF communication patterns
    "identify_communication_patterns",
    "select_k",
    "plot_communication_patterns",
    # manifold similarity / clustering
    "compute_net_similarity",
    "net_embedding",
    "net_clustering",
    "plot_net_embedding",
    # spatial constraints
    "compute_region_distance",
    "spatial_probability_factor",
    # multi-dataset comparison
    "MultiCellComm",
]
