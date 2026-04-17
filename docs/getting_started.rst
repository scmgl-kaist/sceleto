Getting Started
===============

Installation
------------

.. code-block:: bash

   pip install git+https://github.com/scmgl-kaist/sceleto.git

With optional dependencies:

.. code-block:: bash

   pip install "sceleto[all] @ git+https://github.com/scmgl-kaist/sceleto.git"


Marker Discovery
----------------

.. code-block:: python

   import scanpy as sc
   import sceleto as scl

   adata = sc.read_h5ad("your_data.h5ad")

   # Graph-based marker detection
   result = scl.markers.marker(adata, groupby="cell_type")

   # Per-group marker genes
   result.markers  # {'T_cell': ['CD3D', 'CD3E', ...], ...}

   # Dotplot
   scl.dotplot(adata, result.markers, "cell_type")


Cross-resolution Hierarchy
--------------------------

.. code-block:: python

   m1 = scl.markers.marker(adata, groupby="leiden_0.1")
   m2 = scl.markers.marker(adata, groupby="leiden_0.5")
   m3 = scl.markers.marker(adata, groupby="leiden_1.0")

   res = scl.markers.hierarchy(adata, [m1, m2, m3])
   res.compare_markers("0")                           # static heatmap
   res.interactive_viewer(adata, save="viewer.html")  # interactive HTML


Gene Network
------------

.. code-block:: python

   # Correlation-based gene network for a gene of interest
   import sceleto.network as scn

   corr_matrix = scn.build_corr_matrix("CD3D", [adata_cd4, adata_cd8], labels=["CD4", "CD8"])
   G = scn.build_gene_network(corr_matrix)
   scn.plot_network(G, corr_matrix)


Cell Type Annotation
--------------------

.. code-block:: python

   # Label transfer from reference
   result = scl.annotation.transfer(ref_adata, query_adata, groupby="cell_type")

   # Manual incremental annotation
   ann = scl.Annotator(adata)
   ann.update("T cell", select="CD3D", obs_key="leiden")
   ann.summary()
