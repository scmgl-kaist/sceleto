API
=====

Import Sceleto as:

.. code-block:: python

    import sceleto as scl


Marker
------

``scl.markers.marker`` is a class that can be used to find the marker genes for a given label of an `AnnData <https://scanpy.readthedocs.io/en/stable/usage-principles.html#anndata>`_ object.



.. code-block:: python

   marker_object = scl.markers.marker(adata, 'leiden')
   
The ``marker`` class has two methods associated with it, ``plot_marker`` and ``show_marker``.

.. code-block:: python

   marker_object.plot_marker()

   marker_object.show_marker()

``plot_marker`` function creates a dot plot of the marker genes, while ``show_marker`` prints out the marker genes related to that label.

