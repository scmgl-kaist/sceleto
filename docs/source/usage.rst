Usage
=====

Import Sceleto as:

.. code-block:: python

    import sceleto as scl


Workflow
------------

A typical workflow consists of calling functions in the form of:

.. code-block:: python
   
   scl.function_name(adata, ***function_params)

AnnData
------------

Sceleto is a wrapper package for scanpy, thus it is based on the AnnData object used in the scanpy package.
For more information, visit the related scanpy `documentation <https://scanpy.readthedocs.io/en/stable/usage-principles.html#anndata>`_
