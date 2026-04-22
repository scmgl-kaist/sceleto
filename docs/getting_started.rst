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

   import sceleto as scl

   # Correlation-based gene network for a gene of interest
   corr_df = scl.network.build_corr_matrix(
       {"CD4": adata_cd4, "CD8": adata_cd8},
       gene="CD3D",
   )
   top_genes = scl.network.select_top_genes(corr_df, top_n=10, exclude_gene="CD3D")
   feat = scl.network.build_feature_matrix(top_genes, corr_df)
   G = scl.network.build_gene_network(feat, k=5)
   scl.network.plot_network(G, corr_df)

   # One-shot from PANGEA pre-computed correlation DB
   corr_df, feat, G = scl.network.corr_pangea("CD3D", data_dir="path/to/pangea_corr/")


Cell Type Annotation
--------------------

.. code-block:: python

   import sceleto as scl

   # PANGEA-based cell type annotation
   pred = scl.annotation.cellannotator(adata)
   meta = scl.annotation.metaannotator(pred)
