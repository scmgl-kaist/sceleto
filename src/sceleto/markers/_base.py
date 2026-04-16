from __future__ import annotations
from dataclasses import dataclass
from typing import Any

def _flatten_markers(markers: dict, n_top: int) -> list:
    """Flatten a markers dict to a deduplicated gene list, n_top genes per group."""
    seen, flat = set(), []
    for genes in markers.values():
        count = 0
        for g in genes:
            gene = g[0] if isinstance(g, tuple) else str(g)
            if gene not in seen:
                flat.append(gene)
                seen.add(gene)
                count += 1
                if count >= n_top:
                    break
    return flat


# ---- Minimal exceptions (names만 잡아둠) ----
class SceletoError(Exception):
    """Base error for sceleto."""
    pass

class MissingPAGAError(SceletoError):
    """Raised when PAGA info is required but missing."""
    pass

class GroupKeyError(SceletoError):
    """Raised when the given 'groupby' key is not found in adata.obs."""
    pass

class NotComputedError(SceletoError):
    """Raised when a result is requested before compute step."""
    pass

# ---- Minimal config/data holders (필요시 확장) ----
@dataclass
class MarkerConfig:
    groupby: str

# ---- Minimal base class ----
class MarkersBase:
    """
    Very small base for marker workflows. Only stores inputs.
    """
    def __init__(self, adata: Any, groupby: str, **kwargs) -> None:
        self.adata = adata
        self.groupby = groupby
        self.config = MarkerConfig(groupby=groupby)
        # NOTE: No validation here (skeleton). Add later.

    def summary(self) -> str:
        n_obs = getattr(self.adata, "n_obs", "?")
        n_vars = getattr(self.adata, "n_vars", "?")
        has_graph = hasattr(self, "_graph")
        return (
            f"{self.__class__.__name__}(groupby='{self.groupby}', "
            f"n_obs={n_obs}, n_vars={n_vars}, built_graph={has_graph})"
        )

    def __repr__(self) -> str:
        return self.summary()

    def plot(self, n_top: int = 10, **kwargs):
        """Dotplot of ``self.markers`` against ``self.groupby``.

        Genes are flattened from the markers dict (no bracket grouping).
        ``n_top`` controls how many genes per group are included.
        Remaining kwargs are forwarded to ``sceleto.dotplot``.
        """
        from sceleto.dotplot import dotplot
        return dotplot(self.adata, _flatten_markers(self.markers, n_top), self.groupby, n_top=None, **kwargs)

