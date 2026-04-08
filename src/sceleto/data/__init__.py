from sceleto.markers._gene_filter import _load_categories
from ._signatures import genes, B_cell_genes, Kuppfer_mouse_genes, cc_genes_s, cc_genes_m
from ._colors import vega_20, vega_20_scanpy, zeileis_26, godsnot_64


def available_gene_categories() -> list[str]:
    """Return the list of available gene category names."""
    return list(_load_categories().keys())


__all__ = [
    "available_gene_categories",
    "genes",
    "B_cell_genes",
    "Kuppfer_mouse_genes",
    "cc_genes_s",
    "cc_genes_m",
    "vega_20",
    "vega_20_scanpy",
    "zeileis_26",
    "godsnot_64",
]
