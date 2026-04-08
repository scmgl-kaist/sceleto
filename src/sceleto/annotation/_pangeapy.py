"""Thin wrappers around pangeapy for convenience access via sceleto.annotation."""


def _check_pangeapy():
    try:
        import pangeapy  # noqa: F401
    except ImportError:
        raise ImportError(
            "pangeapy is required for this function. "
            "Install it first (e.g. pip install pangeapy)."
        )


def cellannotator(adata, **kwargs):
    """Run pangeapy CellAnnotator on an AnnData object.

    Returns the annotated prediction object.
    """
    _check_pangeapy()
    from pangeapy import CellAnnotator

    return CellAnnotator(**kwargs).annotate(adata)


def metaannotator(pred, **kwargs):
    """Run pangeapy MetaAnnotator on a CellAnnotator prediction.

    Returns the meta-annotated result.
    """
    _check_pangeapy()
    from pangeapy import MetaAnnotator

    return MetaAnnotator(**kwargs).annotate(pred)
