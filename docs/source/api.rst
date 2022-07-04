===
API
===

Marker Genes: markers
=====================

The ``scl.markers`` module provides some useful functions and classes that can be used to find and visualise marker genes.  

Import Sceleto as:

.. code-block:: python

    import sceleto as scl


scl.markers.marker [Class]
--------------------------

``scl.markers.marker`` is a class that can be used to find the marker genes for a given label of an `AnnData <https://scanpy.readthedocs.io/en/stable/usage-principles.html#anndata>`_ object.



.. code-block:: python

   marker_object = scl.markers.marker(adata, 'leiden')
   
The :code:`marker` class has two methods associated with it, :code:`plot_marker` and :code:`show_marker`.

.. code-block:: python

   marker_object.plot_marker()

   marker_object.show_marker()

:code:`plot_marker` function creates a dot plot of the marker genes, while :code:`show_marker` prints out the marker genes related to that label.

scl.markers.find_markers()
--------------------------

:py:func:`markers.find_markers` finds the marker genes for the given label and stores them as a dictionary at ``adata.uns['cdm_groupby']``.

In order to use the other functions in the module, it is advised to first run this function or to create a marker class.
The produced dictionary at ``adata.uns['cdm_groupby']`` can then be used as the input for the ``cdm_out`` argument for the other functions.

.. code-block:: python

   scl.markers.find_markers(adata, 'leiden')
   
In the above example, the dictionary containing marker genes will be created at ``adata.uns['cdm_leiden']``.

scl.markers.find_markers_single()
---------------------------------

:py:func:`markers.find_marker_single` finds all of the marker genes for the given ``adata`` and ``cdm_out`` dictrionary.

.. code-block:: python

   scl.markers.find_markers(adata,'leiden')
   scl.markers.find_markers_single(adata, cdm_out=adata.uns['cdm_leiden'])

Creating a ``scl.markers.marker`` class and accesing the `mks` paramater of the created object gives out the same result as above

.. code-block:: python

   marker_dictionary = scl.markers.marker(adata, 'leiden').mks

This is because the ``scl.markers.marker`` class uses :py:func:`markers.find_markers` and :py:func:`markers.find_markers_single` by default.

scl.markers.volcano_plot [Class]
--------------------------------

This class creates a volcano plot of the desired groups (``comp1`` and ``comp2``) of the given ``anno_key`` (i.e. leiden, age, status ...) beloning to  ``AnnData``.

.. code-block:: python

   my_plot = scl.markers.volcano_plot(adata, 'leiden', 2, 5) #compares leiden groups 2 and 5

However, the above function itself will not plot the volcano plot. The :py:func:`draw()` method has to be called on the object to acquire the plot.

.. code-block:: python

   my_plot.draw()


Predicting Annotation: model
===========================

scl.model.transfer_annotation_jp
--------------------------------

The sceleto package contains the function :py:func:`model.transfer_annotation_jp`, which uses an annotated ``AnnData`` object to predict the annotations
of another ``AnnData`` object, which is not annotated. The function makes use of logistic regression and the gene expressions of the cells.

.. code-block:: python

   scl.model.transfer_annotation_jp(muscle, 'annotation', liver, 'predicted_annotation')

In the above example one can imagine that an annotated AnnData object called muscle was used to predict the annotation of the cells in an AnnData object
that was called liver. 


Useful functions
================

scl.sc_process()
----------------

Performs desired scanpy preprocessing according to the letters passed into the pid parameter

.. code-block:: python

   scl.sc_process(adata, pid = 'fspkuc')

========  =================
letter    function
========  =================
n         normalise
l         log
f         filter hvg
r         remove cc_genes
s         scale
p         pca
k         knn_neighbors
u         umap
c         leiden clustering
        
========  =================

scl.us()
--------

.. code-block:: python 

   scl.us(adata, genes)

Creates a umap using a list of genes. Genes can either be provided as a list or as a comma seperated string

