Interactive demos
=================

Self-contained HTML viewers showing what ``sceleto`` produces on real
datasets. Click an ``icls`` cluster on the UMAP, then click a gene cell in
the marker comparison heatmap (or use the chip strip / mouse wheel /
``←``/``→``) to drive the edge-activation graph on the bottom-right.

Built from
:meth:`sceleto.markers.HierarchyRun.interactive_viewer` (or the lower-level
``build_branching_html``) combined with a
:func:`sceleto.markers.marker` graph run on ``icls``.

PDAC fibroblast
---------------

`Non-batch hierarchy viewer <demos/pdac_fibroblast.html>`__
   Three Leiden resolutions (0.1 / 0.5 / 1.0). Heatmap shows top-10
   marker presence per resolution. Edge-activation graph at bottom-right
   colours each edge of the icls graph by per-gene FC bin.

`Batch hierarchy viewer <demos/pdac_fibroblast_batch.html>`__
   Same dataset built with ``batch_key="Sample"``. Heatmap cells are
   split into per-batch expression strips; the edge-activation graph is
   identical to the non-batch viewer.

How to reproduce
----------------

.. code-block:: python

    import scanpy as sc
    import sceleto as scl

    adata = sc.read_h5ad("PCI01.v43.fibrawt_final_02.h5ad")

    levels = ["leiden_0.1", "leiden_0.5", "leiden_1.0"]
    for lv in levels:
        if lv not in adata.obs.columns:
            res = float(lv.split("_")[1])
            sc.tl.leiden(adata, resolution=res, key_added=lv)

    marker_runs = {lv: scl.markers.marker(adata, lv) for lv in levels}
    hr = scl.markers.hierarchy(adata, marker_runs.values(), n_top_markers=10)
    mgr = scl.markers.marker(adata, "icls")
    hr.interactive_viewer(adata, mgr, save="viewer.html", n_top=10)

For the batch variant, pass ``batch_key="Sample"`` to
:func:`sceleto.markers.hierarchy`.
