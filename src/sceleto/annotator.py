"""Interactive cell annotation helper.

Usage
-----
>>> import sceleto as sl
>>> ann = sl.Annotator(adata, 'celltype')
>>> ann.update('leiden', '0', 'T cell')
>>> ann.update('leiden', '1,2,3', 'B cell')
>>> ann.update('leiden', '4', 'Monocyte', unknown_only=True)
"""

from __future__ import annotations

import numpy as np


class Annotator:
    """Build cell-type annotations incrementally on an AnnData object.

    Parameters
    ----------
    adata
        AnnData object.
    label_key
        Name of the new column in adata.obs.
    copy_from
        If given, initialize from an existing adata.obs column.
    """

    def __init__(self, adata, label_key: str, copy_from: str | None = None):
        if copy_from:
            adata.obs[label_key] = adata.obs[copy_from].astype(str)
        else:
            adata.obs[label_key] = "unknown"

        self._adata = adata
        self._key = label_key
        self._labels = np.array(adata.obs[label_key], dtype=object)

    def update(
        self,
        obs_key: str,
        select: str | list,
        label: str,
        unknown_only: bool = False,
    ):
        """Assign *label* to cells matching *select* in *obs_key*.

        Parameters
        ----------
        obs_key
            Column in adata.obs to match against (e.g. 'leiden').
        select
            Value(s) to match. Comma-separated string or list.
        label
            The annotation label to assign.
        unknown_only
            If True, only update cells still labeled 'unknown'.
        """
        if isinstance(select, str) and "," in select:
            select = [s.strip() for s in select.split(",")]

        if isinstance(select, list):
            mask = self._adata.obs[obs_key].isin(select)
        else:
            mask = self._adata.obs[obs_key] == select

        if unknown_only:
            mask = mask & (self._adata.obs[self._key] == "unknown")

        self._labels[mask] = label
        self._adata.obs[self._key] = self._labels

    def update_mask(self, mask, label: str):
        """Assign *label* to cells matching a boolean mask directly."""
        self._labels[mask] = label
        self._adata.obs[self._key] = self._labels

    def summary(self):
        """Print value counts of current annotations."""
        print(self._adata.obs[self._key].value_counts())
