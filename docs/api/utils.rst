Utilities
=========

Dotplot
-------

.. autofunction:: sceleto.dotplot

Annotator
---------

Build cell-type annotations incrementally by mapping cluster IDs to labels.

.. code-block:: python

   import sceleto as sl

   # start a new 'celltype' column (all cells begin as 'unknown')
   ann = sl.Annotator(adata, 'celltype')

   # one call = one cluster -> one label (exact string match)
   ann.annotate('leiden', '0', 'T cell')
   ann.annotate('leiden', '1', 'B cell')

   # label several clusters at once by looping a dict
   for cluster, label in {'2': 'Monocyte', '3': 'Monocyte', '4': 'NK'}.items():
       ann.annotate('leiden', cluster, label)

   # only fill in cells still left as 'unknown'
   ann.annotate('leiden', '5', 'other', unknown_only=True)

   ann.summary()   # value counts of the current labels

.. autoclass:: sceleto.Annotator
   :members:
   :undoc-members: False

UMAP
----

.. autofunction:: sceleto.us

Preprocessing
-------------

.. autofunction:: sceleto.sc_process
.. autofunction:: sceleto.read_process
.. autofunction:: sceleto.remove_geneset
