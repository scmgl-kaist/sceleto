"""Spatial constraints for spatially-resolved cell–cell communication.

Port of CellChat's spatial machinery (``computeRegionDistance`` in
``modeling.R``): from per-cell/spot coordinates and cell-group labels it derives
a cluster×cluster distance matrix and contact-adjacency matrix, which constrain
the communication probability so that only spatially proximal groups can signal.

Pure geometry (portable): the per-cluster-pair distances use a 1-nearest-neighbor
query (``scipy.spatial.cKDTree``), matching CellChat's ``queryKNN(k=1)``. The
distance→probability transform (``P.spatial = 1/d``) and the range/contact
adjacency thresholds are CellChat-specific and reproduced exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import trim_mean


@dataclass
class SpatialResult:
    """Cluster-level spatial constraints.

    Attributes
    ----------
    d_spatial
        ``K × K`` mean center-to-center distance (µm) between groups; ``NaN``
        where groups are farther apart than ``interaction_range``.
    adj_contact
        ``K × K`` 0/1 contact adjacency (within ``contact_range``).
    group_names
        Group order for both matrices.
    """

    d_spatial: np.ndarray
    adj_contact: np.ndarray
    group_names: list[str]


def compute_region_distance(
    coordinates: np.ndarray,
    groups,
    *,
    interaction_range: float = 250.0,
    ratio: float = 1.0,
    tol: float = 5.0,
    k_min: int = 10,
    contact_dependent: bool = True,
    contact_range: Optional[float] = None,
    do_symmetric: bool = True,
) -> SpatialResult:
    """Cluster×cluster spatial distance + contact adjacency (``computeRegionDistance``).

    Parameters
    ----------
    coordinates
        ``n_cells × 2`` array of spot/cell centroid coordinates (pixels or µm).
    groups
        Per-cell group labels (array/Series/Categorical); defines the K groups.
    interaction_range
        Max diffusion distance (µm). Group pairs whose nearest-neighbor distance
        exceeds this (beyond ``tol``) are marked non-interacting (``d = NaN``).
    ratio
        Pixel→µm conversion (multiplies raw coordinate distances).
    tol
        Tolerance (µm) added when comparing distances against the ranges
        (typically half the spot/cell size).
    k_min
        Minimum number of neighbor spots within range for an edge to count.
    contact_dependent
        If ``True`` build ``adj_contact`` using ``contact_range``; else it is all
        ones (no contact restriction).
    contact_range
        Contact distance (µm) for juxtacrine signaling (required when
        ``contact_dependent``).
    do_symmetric
        Symmetrize the adjacency/distance (an edge needs both directions).

    Returns
    -------
    SpatialResult
    """
    coords = np.asarray(coordinates, dtype=float)
    cat = groups if isinstance(groups, pd.Categorical) else pd.Categorical(groups)
    levels = list(cat.categories)
    K = len(levels)
    codes = cat.codes

    if contact_dependent and contact_range is None:
        raise ValueError(
            "contact_dependent=True requires contact_range (µm), e.g. 10 for "
            "single-cell resolution or ~100 for 10x Visium."
        )
    c_range = contact_range if contact_dependent else 1e4

    d_spatial = np.full((K, K), np.nan)
    adj_spatial = np.zeros((K, K))
    adj_contact = np.zeros((K, K))

    # precompute per-group coordinates + KD-trees
    idx_by_group = [np.where(codes == g)[0] for g in range(K)]
    trees = [cKDTree(coords[idx]) if len(idx) else None for idx in idx_by_group]

    for i in range(K):
        ci = coords[idx_by_group[i]]
        if len(ci) == 0:
            continue
        for j in range(K):
            if trees[j] is None:
                continue
            # 1-NN distance from each point of group i to group j (µm)
            dist, nn = trees[j].query(ci, k=1)
            dist = np.asarray(dist, dtype=float) * ratio
            nn = np.asarray(nn).ravel()

            in_range = (dist - interaction_range) < tol
            adj_spatial[i, j] = 1.0 if len(np.unique(nn[in_range])) >= k_min else 0.0
            in_contact = (dist - c_range) < tol
            adj_contact[i, j] = 1.0 if len(np.unique(nn[in_contact])) >= k_min else 0.0
            # trimmed mean distance between the two groups
            d_spatial[i, j] = trim_mean(dist, 0.1) if len(dist) else np.nan

    if do_symmetric:
        adj_spatial = adj_spatial * adj_spatial.T
        adj_contact = adj_contact * adj_contact.T
    d_spatial = (d_spatial + d_spatial.T) / 2.0

    # zero adjacency → NaN distance (non-interacting)
    adj_spatial = np.where(adj_spatial == 0, np.nan, adj_spatial)
    d_spatial = d_spatial * adj_spatial

    return SpatialResult(d_spatial=d_spatial, adj_contact=adj_contact, group_names=levels)


def spatial_probability_factor(
    d_spatial: np.ndarray,
    *,
    scale_distance: float = 0.01,
) -> np.ndarray:
    """Convert the cluster distance matrix into the ``P.spatial`` factor.

    Reproduces CellChat's ``distance.use=TRUE`` branch:
    ``P.spatial = 1/(d_spatial*scale_distance)``, non-interacting pairs → 0, and
    the diagonal set to the max (self-signaling gets the strongest weight).

    Parameters
    ----------
    d_spatial
        ``K × K`` distance matrix from :func:`compute_region_distance`.
    scale_distance
        Scale so the minimum scaled distance is in ``[1, 2]`` (CellChat guidance).

    Returns
    -------
    np.ndarray
        ``K × K`` non-negative spatial probability factor.
    """
    d = d_spatial * scale_distance
    np.fill_diagonal(d, np.nan)
    if np.isfinite(d).any():
        d_min = np.nanmin(d)
        if d_min < 1:
            raise ValueError(
                f"Minimum scaled distance is {d_min:.3f} (< 1). Increase "
                f"scale_distance (try a value slightly below {1.0 / d_min:.2g})."
            )
    P = 1.0 / d
    P[np.isnan(d)] = 0.0
    # R: diag(P.spatial) <- max(P.spatial). When no pair is proximal, max is 0,
    # so the diagonal is 0 too (whole matrix zero) — match that exactly.
    np.fill_diagonal(P, P.max())
    return P
