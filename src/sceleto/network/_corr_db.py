"""Pre-computed correlation database loader.

Each corr DB on disk follows the layout::

    {data_dir}/{name}_corr_{CT}_{version}.npy     # float32, p × p, mmap-friendly
    {data_dir}/{name}_gene_names_{version}.npy    # str array, length p
    {data_dir}/{name}_n_obs_{version}.json        # {CT: n_obs}

PANGEA ships as ``name="pangea"`` / ``version="v03"`` with cell types
``{B, ENDO, EPI, FIBRO, MACRO, T}``.  Any other DB built via ``build_corr_db``
follows the same layout and is loaded by overriding ``name``/``version``.

``load_corr_db(gene)`` uses memory-mapped npy files to read only the gene's
row from each cell type — O(p × 4 B) per cell type regardless of matrix size.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

# ─────────────────────────────────────────────────────────────────────────────
# Default DB identifiers (PANGEA)
# ─────────────────────────────────────────────────────────────────────────────

PANGEA_NAME = "pangea"
PANGEA_VERSION = "v03"
PANGEA_CELL_TYPES = ["B", "ENDO", "EPI", "FIBRO", "MACRO", "T"]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def list_cell_types(
    data_dir: str | Path | None = None,
    name: str = PANGEA_NAME,
    version: str = PANGEA_VERSION,
) -> list[str]:
    """Return available cell types in a corr database.

    Parameters
    ----------
    data_dir
        Directory containing the DB.  If ``None``, returns the PANGEA defaults
        without touching disk.
    name, version
        DB identifiers (only used when ``data_dir`` is given).

    Returns
    -------
    list[str]
        Cell type keys (read from ``{name}_n_obs_{version}.json``).
    """
    if data_dir is None:
        return list(PANGEA_CELL_TYPES)
    meta_path = Path(data_dir) / f"{name}_n_obs_{version}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found")
    with open(meta_path) as f:
        n_obs_map = json.load(f)
    return list(n_obs_map.keys())


def load_corr_db(
    gene: str,
    data_dir: str | Path,
    cell_types: Optional[list[str]] = None,
    name: str = PANGEA_NAME,
    version: str = PANGEA_VERSION,
) -> pd.DataFrame:
    """Load pre-computed correlations for a gene of interest.

    Uses memory-mapped npy files for fast random row access.

    Parameters
    ----------
    gene
        Gene name (must exist in the corr database).
    data_dir
        Directory containing ``{name}_corr_{CT}_{version}.npy``,
        ``{name}_gene_names_{version}.npy``, and ``{name}_n_obs_{version}.json``.
    cell_types
        Subset of cell types to load.  ``None`` = all (auto-discovered from
        ``{name}_n_obs_{version}.json``).
    name, version
        DB identifiers.  Defaults are PANGEA (``"pangea"`` / ``"v03"``).

    Returns
    -------
    pd.DataFrame
        Wide table: ``gene`` + ``{CT}_corr`` + ``{CT}_pval`` per cell type.
        Compatible with :func:`select_top_genes`.
    """
    data_dir = Path(data_dir)

    # Load n_obs metadata — also gives us the full cell type list
    meta_path = data_dir / f"{name}_n_obs_{version}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found")
    with open(meta_path) as f:
        n_obs_map = json.load(f)
    available_cts = list(n_obs_map.keys())

    cts = cell_types or available_cts
    unknown = [ct for ct in cts if ct not in available_cts]
    if unknown:
        raise ValueError(
            f"Unknown cell type(s) {unknown}. Available: {available_cts}"
        )

    # Load shared gene names
    gn_path = data_dir / f"{name}_gene_names_{version}.npy"
    if not gn_path.exists():
        raise FileNotFoundError(f"{gn_path} not found")
    gene_names = np.load(gn_path, allow_pickle=True)

    idx_arr = np.where(gene_names == gene)[0]
    if idx_arr.size == 0:
        raise ValueError(f"Gene {gene!r} not found in corr database")
    idx = int(idx_arr[0])

    merged: Optional[pd.DataFrame] = None
    for ct in cts:
        npy_path = data_dir / f"{name}_corr_{ct}_{version}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"{npy_path} not found")

        # mmap: only reads the requested row from disk
        corr = np.load(npy_path, mmap_mode="r")
        r_vals = corr[idx].astype(np.float32)

        n_obs = n_obs_map[ct]
        dfree = n_obs - 2

        # Compute p-values on the fly
        r_clipped = np.clip(r_vals, -0.9999999, 0.9999999)
        valid = np.abs(r_clipped) < 1.0
        p_vals = np.zeros(len(r_vals), dtype=np.float32)
        t_vals = r_clipped[valid] * np.sqrt(dfree / (1 - r_clipped[valid] ** 2))
        p_vals[valid] = (2 * student_t.sf(np.abs(t_vals), dfree)).astype(np.float32)

        sub = pd.DataFrame({
            "gene": gene_names,
            f"{ct}_corr": r_vals,
            f"{ct}_pval": p_vals,
        })

        merged = sub if merged is None else merged.merge(sub, on="gene", how="outer")

    return merged.reset_index(drop=True)
