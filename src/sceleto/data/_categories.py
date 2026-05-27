"""Gene category lookup backed by ``gene_categories.json``."""

from __future__ import annotations

import json
from importlib import resources
from typing import Optional, Sequence


def _load_categories() -> dict[str, list[str]]:
    """Load the full ``{category: gene_list}`` mapping from the bundled JSON."""
    ref = resources.files("sceleto.data").joinpath("gene_categories.json")
    with resources.as_file(ref) as path:
        with open(path) as f:
            return json.load(f)


def available_categories() -> list[str]:
    """Return the list of available gene category names."""
    return list(_load_categories().keys())


def get_category(name: str) -> list[str]:
    """Return the gene list for a single category.

    Parameters
    ----------
    name
        Category name (see :func:`available_categories`).
    """
    cats = _load_categories()
    if name not in cats:
        raise ValueError(
            f"Unknown category {name!r}. Available: {list(cats.keys())}"
        )
    return list(cats[name])


def get_categories(
    names: Optional[Sequence[str]] = None,
) -> dict[str, list[str]]:
    """Return a ``{category: gene_list}`` dict.

    Parameters
    ----------
    names
        If given, restrict to these categories. Otherwise return all.
    """
    cats = _load_categories()
    if names is None:
        return {k: list(v) for k, v in cats.items()}
    out: dict[str, list[str]] = {}
    for n in names:
        if n not in cats:
            raise ValueError(
                f"Unknown category {n!r}. Available: {list(cats.keys())}"
            )
        out[n] = list(cats[n])
    return out
