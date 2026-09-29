Bundled Data
============

``sceleto.data`` ships curated gene sets and categorical color palettes with the
package, so no download is needed. The gene sets live in
``sceleto/data/gene_categories.json`` and are exposed through three lookup
functions.

Gene sets
---------

.. code-block:: python

   import sceleto as scl

   # list every available gene set name
   scl.data.available_categories()
   # ['IG_Gene', 'Mito_RNA', 'Non_coding_Misc', 'Protein_Coding', ...]

   # one gene set -> list[str]
   cc = scl.data.get_category('Cell_Cycle')

   # several gene sets -> {name: list[str]}; omit the argument to get all of them
   phases = scl.data.get_categories(['Cell_Cycle_S', 'Cell_Cycle_G2M'])

An unknown name raises ``ValueError`` listing the available names. Each call
returns a fresh copy, so modifying the returned list does not affect later calls.

.. note::

   Gene symbols are stored in upper case (human-style, e.g. ``CD3E``). For
   mouse data, convert to orthologs or match the capitalization of your
   ``adata.var_names`` first.

Available gene sets
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Group
     - Gene sets (number of genes)
   * - Gene biotype
     - ``Protein_Coding`` (19,477), ``Pseudogene`` (10,507), ``lncRNA`` (6,207),
       ``miRNA`` (1,852), ``snRNA`` (1,836), ``Non_coding_Misc`` (1,043),
       ``snoRNA`` (456), ``IG_Gene`` (213), ``TR_Gene`` (195), ``rRNA`` (42),
       ``Mito_RNA`` (24), ``Uncertain`` (23), ``scaRNA`` (19)
   * - Function
     - ``Surface`` (2,799), ``Transcription_Factor`` (1,892)
   * - Cell cycle
     - ``Cell_Cycle`` (559), ``Cell_Cycle_S`` (43), ``Cell_Cycle_G2M`` (55)
   * - Aging / senescence
     - ``Cell_Senescence_Up`` (525), ``Aging_Up`` (159),
       ``Aging_Thymus_Epi_Fb`` (123), ``Ageing_Known`` (18)
   * - T / NK cell states
     - ``T_Exhaustion`` (17), ``T_Resident_Memory`` (7), ``T_Naive`` (5),
       ``T_Central_Memory`` (3), ``T_Effector_Memory`` (3), ``T_EMRA`` (3),
       ``MAIT`` (2), ``NK_CD56bright_Naive`` (3), ``NK_CD56dim_Mature`` (2)
   * - B cell states
     - ``B_Follicular`` (10), ``B_Memory`` (10), ``B_Naive`` (10),
       ``B_Plasma`` (10), ``Human_B1`` (5), ``Human_B1_VDJ`` (4),
       ``Mouse_B1_Metabolism`` (13), ``B_Activated`` (2)
   * - Myeloid / other immune
     - ``Macrophage_M1`` (7), ``Macrophage_M2`` (6), ``Microglia`` (3),
       ``Eosinophil`` (3), ``Granulocyte`` (2), ``Mast_Or_ILC2`` (2),
       ``Autoimmune`` (3)
   * - Kupffer cell programs
     - ``Kupffer_Others`` (9), ``Kupffer_mGluRs`` (8), ``Kupffer_Chemokines`` (7),
       ``Kupffer_Neurologic_Transporters`` (7), ``Kupffer_TGFb_Signaling`` (6),
       ``Kupffer_IFNg_Signaling`` (3), ``Kupffer_Ly6h_Mono`` (3),
       ``Kupffer_Liver_Macrophage`` (2)
   * - Tissue-specific
     - ``Kidney_Renin`` (2), ``Liver_Hepatic_Stellate`` (1)

Common uses
~~~~~~~~~~~

.. code-block:: python

   import scanpy as sc
   import sceleto as scl

   # drop immunoglobulin / TCR genes before clustering
   adata = scl.remove_geneset(
       adata,
       scl.data.get_category('IG_Gene') + scl.data.get_category('TR_Gene'),
   )

   # cell-cycle scoring with scanpy
   sc.tl.score_genes_cell_cycle(
       adata,
       s_genes=scl.data.get_category('Cell_Cycle_S'),
       g2m_genes=scl.data.get_category('Cell_Cycle_G2M'),
   )

   # keep only surface-protein candidates from a marker list
   surface = set(scl.data.get_category('Surface'))
   surface_markers = [g for g in markers if g in surface]

.. autofunction:: sceleto.data.available_categories
.. autofunction:: sceleto.data.get_category
.. autofunction:: sceleto.data.get_categories

Color palettes
--------------

Lists of hex colors for categorical annotations:

- ``vega_20``: the 20-color Vega/matplotlib ``tab20`` palette
- ``vega_20_scanpy``: scanpy's default 20-color ordering, greys replaced (20 colors)
- ``zeileis_26``: 28 perceptually spaced colors
- ``godsnot_64``: 103 high-contrast colors (no black), useful for many clusters

.. code-block:: python

   n = adata.obs['leiden'].nunique()
   adata.uns['leiden_colors'] = scl.data.godsnot_64[:n]
