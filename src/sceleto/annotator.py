"""Interactive cell annotation helper.

Usage
-----
>>> import sceleto as sl
>>> ann = sl.Annotator(adata, 'celltype')
>>> ann.annotate('leiden', '0', 'T cell')
>>> ann.annotate('leiden', '1', 'B cell')
>>> ann.annotate('leiden', '4', 'Monocyte', unknown_only=True)
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

    def annotate(
        self,
        obs_key: str,
        select: str,
        label: str,
        unknown_only: bool = False,
    ):
        """Assign *label* to cells whose ``obs_key`` value equals ``select``.

        One call = one decision. ``select`` is a single string matched
        exactly against ``adata.obs[obs_key]``; no list or comma splitting.
        To label multiple groups with the same label, call repeatedly
        (e.g. in a dict loop).

        Parameters
        ----------
        obs_key
            Column in adata.obs to match against (e.g. 'leiden', 'leiden_R').
        select
            Exact value to match in ``adata.obs[obs_key]``.
        label
            The annotation label to assign.
        unknown_only
            If True, only update cells still labeled 'unknown'.
        """
        mask = self._adata.obs[obs_key] == select

        if unknown_only:
            mask = mask & (self._adata.obs[self._key] == "unknown")

        self._labels[mask] = label
        self._adata.obs[self._key] = self._labels

    def annotate_mask(self, mask, label: str):
        """Assign *label* to cells matching a boolean mask directly."""
        self._labels[mask] = label
        self._adata.obs[self._key] = self._labels

    def summary(self):
        """Print value counts of current annotations."""
        print(self._adata.obs[self._key].value_counts())
