"""Build a correlation DB from per-cell-type metacell AnnDatas.

Writes the on-disk layout consumed by :func:`load_corr_db`::

    {out_dir}/{name}_corr_{CT}_{version}.npy     # float16, p × p
    {out_dir}/{name}_gene_names_{version}.npy    # str array, length p
    {out_dir}/{name}_n_obs_{version}.json        # {CT: n_obs}

All metacell AnnDatas must share the same gene set (``var_names`` in identical
order).  Per-CT pipeline mirrors the PANGEA build:

    X(float32) → per-gene standardize(mean=0, std=1, ddof=1)
              → C = X_norm.T @ X_norm / (n-1)
              → fill_diagonal(1.0) → cast float16 → save .npy
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

_SAFE_CT_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def build_corr_db(
    metacells: Mapping[str, AnnData],
    out_dir: str | Path,
    name: str,
    version: str = "v01",
    layer: Optional[str] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Build a correlation database from per-cell-type metacell AnnDatas.

    Parameters
    ----------
    metacells
        Mapping ``{cell_type: metacell_adata}``.  Each AnnData has metacells
        as rows and the *shared* gene set as columns.  Typically the output
        of :func:`build_metacells` collected across cell types.
    out_dir
        Output directory.  Created if missing.
    name
        DB prefix (e.g. ``"pangea"``, ``"hlca"``).
    version
        DB version tag (e.g. ``"v01"``).
    layer
        Layer to use as the input expression matrix.  ``None`` = ``adata.X``.
    overwrite
        If False, skip a cell type when its target npy already exists.
    verbose
        Print per-CT progress.

    Returns
    -------
    Path
        ``out_dir`` (as Path).  Files written:

        - ``{name}_corr_{CT}_{version}.npy`` (float16, p × p, mmap-friendly)
        - ``{name}_gene_names_{version}.npy``
        - ``{name}_n_obs_{version}.json``
    """
    if not metacells:
        raise ValueError("metacells is empty")

    # Validate cell type names — they end up in file paths
    for ct in metacells:
        if not _SAFE_CT_NAME.match(ct):
            raise ValueError(
                f"Cell type name {ct!r} contains characters outside "
                f"[A-Za-z0-9_-]. Rename keys before building."
            )

    # Validate shared gene set (order matters — corr matrices share an axis)
    cts = list(metacells.keys())
    ref_genes = np.array(metacells[cts[0]].var_names)
    for ct in cts[1:]:
        cur = np.array(metacells[ct].var_names)
        if cur.shape != ref_genes.shape or not np.array_equal(cur, ref_genes):
            raise ValueError(
                f"Gene set mismatch: {cts[0]!r} vs {ct!r}. "
                f"All metacells must share the same var_names in identical order."
            )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write shared gene names once
    gn_path = out_dir / f"{name}_gene_names_{version}.npy"
    if overwrite or not gn_path.exists():
        np.save(gn_path, ref_genes)
        if verbose:
            print(f"[shared] gene_names → {gn_path.name} ({len(ref_genes)} genes)")

    # Per-CT corr build
    n_obs_map: dict[str, int] = {}

    # If sidecar n_obs.json exists, merge with it (so partial rebuilds preserve state)
    meta_path = out_dir / f"{name}_n_obs_{version}.json"
    if meta_path.exists():
        with open(meta_path) as f:
            n_obs_map = json.load(f)

    for ct, adata in metacells.items():
        npy_path = out_dir / f"{name}_corr_{ct}_{version}.npy"
        if npy_path.exists() and not overwrite:
            if verbose:
                print(f"[{ct}] {npy_path.name} exists, skipping (overwrite=False)")
            n_obs_map.setdefault(ct, int(adata.n_obs))
            continue

        t0 = time.time()
        X = adata.layers[layer] if layer is not None else adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        n, p = X.shape
        if verbose:
            print(f"[{ct}] X: {n} × {p}, standardizing ...")

        if n < 2:
            raise ValueError(
                f"[{ct}] needs at least 2 metacells (n_obs={n}) "
                f"to compute correlation."
            )

        means = X.mean(axis=0)
        stds = X.std(axis=0, ddof=1)
        stds[stds == 0] = 1.0  # zero-variance genes → corr stays 0
        X_norm = (X - means) / stds
        del X

        if verbose:
            print(f"[{ct}] matmul ({p}×{p}, float32) ...")
        C = (X_norm.T @ X_norm) / (n - 1)
        del X_norm
        np.fill_diagonal(C, 1.0)

        np.save(npy_path, C.astype(np.float16))
        del C

        n_obs_map[ct] = int(n)

        if verbose:
            size_gb = npy_path.stat().st_size / 1e9
            print(f"[{ct}] DONE → {npy_path.name} "
                  f"({size_gb:.2f} GB, {time.time()-t0:.0f}s)")

    # Write n_obs sidecar
    with open(meta_path, "w") as f:
        json.dump(n_obs_map, f, indent=2)
    if verbose:
        print(f"[shared] n_obs → {meta_path.name}: {n_obs_map}")

    return out_dir
