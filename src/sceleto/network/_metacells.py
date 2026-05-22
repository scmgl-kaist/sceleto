"""Build metacells from a single-cell-type AnnData.

Per-sample pipeline (run in a loop over ``obs[sample_key]``):

    1. subset to one sample
    2. HVG → scale → PCA → neighbors
    3. random + refined nhood sampling on the kNN graph (_make_nhoods)
    4. nhood × gene count matrix = nhoods.T @ counts
    5. normalize_total(1e4) + log1p
    6. tag with sample

All per-sample metacells are concatenated into a single AnnData and returned.
``var`` is the intersection of per-sample vars (sc.concat default).

The input adata is expected to already be subset to a single cell type — the
caller controls the cell type axis (see ``build_metacells_dir`` for a
directory-driven convenience wrapper).
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from sklearn.metrics.pairwise import euclidean_distances


# ─────────────────────────────────────────────────────────────────────────────
# Private: nhood sampling on a kNN graph (milopy-derived)
# ─────────────────────────────────────────────────────────────────────────────

def _make_nhoods(
    adata: AnnData,
    prop: float = 0.1,
    seed: int = 42,
) -> None:
    """Sample + refine vertices on adata.obsp['connectivities'] to define nhoods.

    Requires sc.pp.neighbors(adata) already run.  Modifies ``adata`` in place:

        adata.obsm['nhoods']                 (n_cells × n_nhoods, binary sparse)
        adata.obs['nhood_ixs_random']        (0/1)
        adata.obs['nhood_ixs_refined']       (0/1)
        adata.obs['nhood_kth_distance']      (distance to k-th NN at refined vertices)
    """
    try:
        use_rep = adata.uns["neighbors"]["params"]["use_rep"]
    except KeyError:
        use_rep = "X_pca"

    if "connectivities" not in adata.obsp:
        raise KeyError(
            "adata.obsp['connectivities'] missing — run sc.pp.neighbors(adata) first"
        )
    knn_graph = adata.obsp["connectivities"].copy()

    if use_rep == "X":
        X_dimred = adata.X
        if sp.issparse(X_dimred):
            X_dimred = X_dimred.toarray()
    else:
        X_dimred = adata.obsm[use_rep]

    n_ixs = int(np.round(adata.n_obs * prop))
    knn_graph[knn_graph != 0] = 1

    random.seed(seed)
    random_vertices = sorted(random.sample(range(adata.n_obs), k=n_ixs))

    ixs_nn = knn_graph[random_vertices, :]
    non_zero_rows = ixs_nn.nonzero()[0]
    non_zero_cols = ixs_nn.nonzero()[1]

    refined_vertices = np.empty(shape=[len(random_vertices)])
    for i in range(len(random_vertices)):
        members = non_zero_cols[non_zero_rows == i]
        if members.size == 0:
            refined_vertices[i] = random_vertices[i]
            continue
        nh_pos = np.median(X_dimred[members, :], axis=0).reshape(-1, 1)
        dists = euclidean_distances(X_dimred[members, :], nh_pos.T)
        refined_vertices[i] = members[dists.argmin()]

    refined_vertices = np.unique(refined_vertices.astype("int"))
    refined_vertices.sort()

    adata.obsm["nhoods"] = knn_graph[:, refined_vertices]
    adata.obs["nhood_ixs_random"] = adata.obs_names.isin(
        adata.obs_names[random_vertices]
    ).astype("int")
    adata.obs["nhood_ixs_refined"] = adata.obs_names.isin(
        adata.obs_names[refined_vertices]
    ).astype("int")

    knn_dists = adata.obsp["distances"]
    nhood_mask = (adata.obs["nhood_ixs_refined"] == 1).values
    dist_mat = knn_dists[nhood_mask, :]
    k_distances = dist_mat.max(1).toarray().ravel()
    adata.obs["nhood_kth_distance"] = 0
    adata.obs.loc[nhood_mask, "nhood_kth_distance"] = k_distances


# ─────────────────────────────────────────────────────────────────────────────
# Private: per-sample helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_counts(adata: AnnData, counts: str) -> np.ndarray | sp.spmatrix:
    """Return the matrix used as raw counts for metacell aggregation."""
    if counts == "raw":
        if adata.raw is None:
            raise ValueError(
                'counts="raw" but adata.raw is None. '
                'Pass counts="X" or a layer name explicitly.'
            )
        return adata.raw.to_adata().X
    if counts == "X":
        return adata.X
    if counts in adata.layers:
        return adata.layers[counts]
    raise ValueError(
        f"counts={counts!r}: not 'raw', not 'X', and not a layer key. "
        f"Available layers: {list(adata.layers.keys())}"
    )


def _build_one_sample(
    sub: AnnData,
    sample: str,
    counts: str,
    n_neighbors: int,
    prop: float,
    normalize_target_sum: Optional[float],
    log1p: bool,
    seed: int,
) -> AnnData:
    """Pipeline for one sample subset → metacell AnnData."""
    counts_mat = _resolve_counts(sub, counts)
    sub = sub.copy()
    sub.layers["__metacell_counts__"] = counts_mat

    sc.pp.highly_variable_genes(sub)
    sc.pp.scale(sub)
    sc.tl.pca(sub, use_highly_variable=True)
    sc.pp.neighbors(sub, n_neighbors=n_neighbors)

    _make_nhoods(sub, prop=prop, seed=seed)

    nhoods = sub.obsm["nhoods"]
    if not sp.issparse(nhoods):
        nhoods = sp.csr_matrix(nhoods)
    cnt = sub.layers["__metacell_counts__"]

    X_nhood = nhoods.T @ cnt
    refined_mask = (sub.obs["nhood_ixs_refined"] == 1).values
    index_cells = sub.obs_names[refined_mask].values
    n_cells_per_nhood = np.asarray(nhoods.sum(axis=0)).ravel()

    obs = pd.DataFrame(
        {
            "index_cell": index_cells,
            "n_cells": n_cells_per_nhood,
            "sample": sample,
        },
        index=[f"{sample}__nhood_{i}" for i in range(X_nhood.shape[0])],
    )

    mc = ad.AnnData(X=X_nhood, obs=obs, var=sub.var.copy())

    if normalize_target_sum is not None:
        sc.pp.normalize_total(mc, target_sum=normalize_target_sum)
    if log1p:
        sc.pp.log1p(mc)

    return mc


# ─────────────────────────────────────────────────────────────────────────────
# Public
# ─────────────────────────────────────────────────────────────────────────────

def build_metacells(
    adata: AnnData,
    *,
    sample_key: str,
    counts: str = "raw",
    min_cells_per_sample: int = 100,
    n_neighbors: int = 15,
    prop: float = 0.1,
    normalize_target_sum: Optional[float] = 1e4,
    log1p: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> AnnData:
    """Build metacells for a single cell type, per sample.

    The input adata is expected to be subset to a single cell type. For each
    sample (``adata.obs[sample_key]``), runs HVG → PCA → neighbors → nhood
    sampling, sums counts within each nhood, and normalizes.  Results across
    samples are concatenated.

    Parameters
    ----------
    adata
        Single-cell-type AnnData with at least one sample column.
    sample_key
        Column in ``adata.obs`` identifying samples.
    counts
        Source for raw counts to sum across nhoods.  One of:
        - ``"raw"`` (default): ``adata.raw.X``
        - ``"X"``: use ``adata.X`` directly (must be counts)
        - any layer name: ``adata.layers[counts]``
    min_cells_per_sample
        Skip samples with fewer than this many cells.
    n_neighbors
        Passed to ``sc.pp.neighbors``.
    prop
        Fraction of cells to seed as nhood anchors (refined to ~unique medoids).
    normalize_target_sum
        ``sc.pp.normalize_total(target_sum=...)`` on each metacell.
        Set ``None`` to skip.
    log1p
        Apply ``sc.pp.log1p`` after normalization.
    seed
        Reproducibility seed for nhood sampling.
    verbose
        Show per-sample tqdm progress.

    Returns
    -------
    AnnData
        Concatenated metacell AnnData (rows: ``{sample}__nhood_{i}``).
        ``obs`` columns: ``index_cell``, ``n_cells``, ``sample``.
    """
    if sample_key not in adata.obs.columns:
        raise KeyError(f"sample_key {sample_key!r} not in adata.obs")

    # Silence the noisy "Using X_pca as default embedding" warnings from
    # _make_nhoods when neighbors uns lacks the use_rep param key.
    logger = logging.getLogger()
    prev_level = logger.level
    if verbose:
        logger.setLevel(logging.ERROR)

    samples = list(pd.Categorical(adata.obs[sample_key]).categories)
    if not samples:
        samples = sorted(adata.obs[sample_key].astype(str).unique())

    iterator = samples
    if verbose:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(samples, desc="samples")
        except ImportError:
            pass

    parts: list[AnnData] = []
    skipped: list[tuple[str, int]] = []

    try:
        for sample in iterator:
            sub = adata[adata.obs[sample_key] == sample]
            n = sub.n_obs
            if n < min_cells_per_sample:
                skipped.append((str(sample), n))
                continue
            sub = sub.copy()
            mc = _build_one_sample(
                sub,
                sample=str(sample),
                counts=counts,
                n_neighbors=n_neighbors,
                prop=prop,
                normalize_target_sum=normalize_target_sum,
                log1p=log1p,
                seed=seed,
            )
            parts.append(mc)
    finally:
        logger.setLevel(prev_level)

    if not parts:
        raise RuntimeError(
            f"No samples passed min_cells_per_sample={min_cells_per_sample}. "
            f"Sample sizes: {[(s, n) for s, n in skipped][:10]} ..."
        )

    metacells = ad.concat(parts, join="outer", merge="first")

    if verbose:
        print(
            f"build_metacells: {len(parts)}/{len(samples)} samples used "
            f"(skipped {len(skipped)} below {min_cells_per_sample} cells), "
            f"total metacells: {metacells.n_obs}"
        )

    return metacells
