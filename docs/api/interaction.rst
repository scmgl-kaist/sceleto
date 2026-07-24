sceleto.interaction
====================

Cell–cell communication inference — a Python port of CellChat. Infers
ligand–receptor–mediated signaling between cell groups in an
:class:`~anndata.AnnData`, using the curated CellChatDB database.

Database
--------

.. autofunction:: sceleto.interaction.load_cellcommdb
.. autofunction:: sceleto.interaction.load_ppi
.. autofunction:: sceleto.interaction.list_cellcommdb
.. autoclass:: sceleto.interaction.CellCommDB
   :members:


Gene-symbol Normalization
-------------------------

.. autoclass:: sceleto.interaction.GeneMapper
   :members:
.. autoclass:: sceleto.interaction.MappingResult
   :members:


Core Inference
--------------

The :class:`~sceleto.interaction.CellComm` class drives the whole pipeline;
its methods mirror CellChat's workflow: ``identify_overexpressed`` →
``compute_communication`` → ``filter_communication`` →
``compute_communication_pathway`` → ``aggregate`` → ``subset_communication`` /
``rank_net`` / ``contribution``.

.. autoclass:: sceleto.interaction.CellComm
   :members:


Network Analysis
----------------

.. autofunction:: sceleto.interaction.compute_centrality
.. autofunction:: sceleto.interaction.signaling_role
.. autofunction:: sceleto.interaction.signaling_role_pathway
.. autofunction:: sceleto.interaction.network_graph


Communication Patterns (NMF)
----------------------------

Outgoing/incoming communication-pattern discovery via non-negative matrix
factorization (CellChat ``selectK`` / ``identifyCommunicationPatterns``). The
factorization uses the same Lee multiplicative-update rule as R's ``NMF`` package
(``scikit-learn`` ``NMF(solver='mu')``).

.. autofunction:: sceleto.interaction.select_k
.. autofunction:: sceleto.interaction.identify_communication_patterns
.. autofunction:: sceleto.interaction.plot_communication_patterns


Pathway Similarity & Clustering (manifold)
------------------------------------------

Functional/structural similarity between signaling-pathway networks, UMAP
embedding, and clustering (CellChat ``computeNetSimilarity`` / ``netEmbedding`` /
``netClustering``).

.. autofunction:: sceleto.interaction.compute_net_similarity
.. autofunction:: sceleto.interaction.net_embedding
.. autofunction:: sceleto.interaction.net_clustering
.. autofunction:: sceleto.interaction.plot_net_embedding


Multi-dataset Comparison
------------------------

Compare communication across conditions (e.g. disease vs normal). Build per-
condition :class:`~sceleto.interaction.CellComm` objects, then wrap them in
:class:`~sceleto.interaction.MultiCellComm` to compare total interactions, rank
pathways by differential information flow (with a Wilcoxon test), and compute
differential networks.

.. autoclass:: sceleto.interaction.MultiCellComm
   :members:


Spatial Constraints
-------------------

For spatially-resolved data, build the :class:`~sceleto.interaction.CellComm`
object with ``coordinates`` and ``spatial_factors={'ratio':…, 'tol':…}``, then
pass ``interaction_range`` / ``contact_range`` to ``compute_communication``. The
communication probability is constrained by spatial proximity between cell
groups.

.. autofunction:: sceleto.interaction.compute_region_distance
.. autofunction:: sceleto.interaction.spatial_probability_factor


Visualization
-------------

.. autofunction:: sceleto.interaction.plot_network_circle
.. autofunction:: sceleto.interaction.plot_heatmap
.. autofunction:: sceleto.interaction.plot_bubble
.. autofunction:: sceleto.interaction.plot_chord
.. autofunction:: sceleto.interaction.plot_rank_net
.. autofunction:: sceleto.interaction.plot_signaling_role_heatmap
.. autofunction:: sceleto.interaction.plot_contribution
.. autofunction:: sceleto.interaction.plot_signaling_role_scatter
.. autofunction:: sceleto.interaction.plot_signaling_role_network
.. autofunction:: sceleto.interaction.plot_chord_gene
.. autofunction:: sceleto.interaction.plot_communication_patterns_river
.. autofunction:: sceleto.interaction.plot_communication_patterns_dot
.. autofunction:: sceleto.interaction.plot_net_embedding_zoom
.. autofunction:: sceleto.interaction.plot_aggregate
