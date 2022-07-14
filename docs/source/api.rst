===
API
===

Marker Genes: markers
=====================

The ``scl.markers`` module provides some useful functions and classes that can be used to find and visualise marker genes.  

Import Sceleto as:

.. code-block:: python

    import sceleto as scl


scl.markers.marker 
-------------------

``scl.markers.marker`` is a class that can be used to find the marker genes for a given label of an `AnnData <https://scanpy.readthedocs.io/en/stable/usage-principles.html#anndata>`_ object.



.. code-block:: python

   marker_object = scl.markers.marker(adata, 'leiden')
   
The :code:`marker` class has two methods associated with it, :code:`plot_marker` and :code:`show_marker`.

.. code-block:: python

   marker_object.plot_marker()

   marker_object.show_marker()

:code:`plot_marker` function creates a dot plot of the marker genes, while :code:`show_marker` prints out the marker genes related to that label.

scl.markers.find_markers
-------------------------

:py:func:`markers.find_markers` finds the marker genes for the given label and stores them as a dictionary at ``adata.uns['cdm_groupby']``.

In order to use the other functions in the module, it is advised to first run this function or to create a marker class.
The produced dictionary at ``adata.uns['cdm_groupby']`` can then be used as the input for the ``cdm_out`` argument for the other functions.

.. code-block:: python

   scl.markers.find_markers(adata, 'leiden')
   
In the above example, the dictionary containing marker genes will be created at ``adata.uns['cdm_leiden']``.

scl.markers.find_markers_single
-------------------------------

:py:func:`markers.find_marker_single` finds all of the marker genes for the given ``adata`` and ``cdm_out`` dictrionary.

.. code-block:: python

   scl.markers.find_markers(adata,'leiden')
   scl.markers.find_markers_single(adata, cdm_out=adata.uns['cdm_leiden'])

Creating a ``scl.markers.marker`` class and accesing the `mks` paramater of the created object gives out the same result as above

.. code-block:: python

   marker_dictionary = scl.markers.marker(adata, 'leiden').mks

This is because the ``scl.markers.marker`` class uses :py:func:`markers.find_markers` and :py:func:`markers.find_markers_single` by default.

scl.markers.volcano_plot 
-------------------------

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

Extra Functions/Classes for DEG Analysis: jhk
=============================================

The set of functions created by Junho Kang to aid his studies are provided in this module. 

scl.jhk.diffxpy_deg
--------------------

This function can be used to find the correlation between differentially expressed genes 
belonging to two subgroups of a main group. I.e. correlation between deg of two disease types.

.. code-block:: python

    scl.jhk.diffxpy_deg(adata,'Disease','ALS',"Alzheimer's", cell_type='predicted_annotation', tissue='brain', test='t_test')

Note: It returns a dictionary.

scl.jhk.plot_volcano
--------------------

Can be used to create a volcano plot of differentially expressed genes belonging to a certain cell type.

.. code-block:: python

   deg_als_alzheimer = scl.jhk.diffxpy_deg(adata,'Disease','ALS',"Alzheimer's", cell_type='predicted_annotation', tissue='brain', test='t_test')
   scl.jhk.plot_volcano(deg_als_alzheimer,'tissue_microglia')

The dictionary that is returned after running the :py:func:`scl.jhk.diffxpy_deg` function can be used as an input for this function.

scl.jhk.deg_summary
--------------------

Returns a dictionary which contains some data about the differentialy expressed genes belonging to a certain cell type.

.. code-block:: python

   deg_als_alzheimer = scl.jhk.diffxpy_deg(adata,'Disease','ALS',"Alzheimer's", cell_type='predicted_annotation', tissue='brain', test='t_test')
   scl.jhk.deg_summary(deg_als_alzheimer,'tissue_microglia')

The dictionary that is returned after running the :py:func:`scl.jhk.diffxpy_deg` function can be used as an input for this function.

scl.jhk.ad_summary
--------------------

Returns a dictionary which contains some data regarding the provided ``AnnData`` object.

.. code-block:: python

   scl.jhk.ad_summary(anndata)

scl.jhk.sample_volcano
----------------------

Creates a sample_volcane object by using samplewise data that is inputted using patient_id.
To draw the volcano plot call the :py:func:`draw` method.

.. code-block:: python

   volcano_example_1 = scl.jhk.sample_volcano(dcdata,patient_id='PatientID',anno_key='ASDC',comp1='ASDC',comp2='DC',quick=True)
   volcano_example_2 = scl.jhk.sample_volcano(natsub,patient_id='sample',anno_key='status',comp1='MS postTx',comp2='MS preTx',quick=True)
   
   volcano_example_2.draw(title='Title',sig_mode='pval',x_pos=0.7,pvalue_cut=1.7,show=True)

Useful functions
================

scl.sc_process
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

scl.us
--------

.. code-block:: python 

   scl.us(adata, genes)

Creates a umap using a list of genes. Genes can either be provided as a list or as a comma seperated string

