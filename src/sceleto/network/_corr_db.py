"""Pre-computed correlation database loader.

Pre-computed corr tables are generated from PANGEA Milo metacell h5ad files.
Each npy stores the full (p × p) Pearson correlation matrix (float16) for one
cell type.  Gene names and metacell counts are stored in shared sidecar files.

``load_corr_db(gene)`` uses memory-mapped npy files to read only the gene's
row from each cell type — O(72 KB) per cell type regardless of matrix size.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PANGEA_CELL_TYPES = ["B", "ENDO", "EPI", "FIBRO", "MACRO", "T"]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def list_cell_types() -> list[str]:
    """Return available cell types in the corr database."""
    return list(PANGEA_CELL_TYPES)


def load_corr_db(
    gene: str,
    data_dir: str | Path,
    cell_types: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load pre-computed correlations for a gene of interest.

    Uses memory-mapped npy files for fast random row access.

    Parameters
    ----------
    gene
        Gene name (must exist in the corr database).
    data_dir
        Directory containing ``pangea_corr_{CT}_v03.npy``,
        ``pangea_gene_names_v03.npy``, and ``pangea_n_obs_v03.json``.
    cell_types
        Subset of cell types to load.  ``None`` = all 6.

    Returns
    -------
    pd.DataFrame
        Wide table: ``gene`` + ``{CT}_corr`` + ``{CT}_pval`` per cell type.
        Compatible with :func:`select_top_genes`.
    """
    cts = cell_types or PANGEA_CELL_TYPES
    for ct in cts:
        if ct not in PANGEA_CELL_TYPES:
            raise ValueError(
                f"Unknown cell type {ct!r}. Available: {PANGEA_CELL_TYPES}"
            )

    data_dir = Path(data_dir)

    # Load shared gene names
    gn_path = data_dir / "pangea_gene_names_v03.npy"
    if not gn_path.exists():
        raise FileNotFoundError(f"{gn_path} not found")
    gene_names = np.load(gn_path, allow_pickle=True)

    idx_arr = np.where(gene_names == gene)[0]
    if idx_arr.size == 0:
        raise ValueError(f"Gene {gene!r} not found in corr database")
    idx = int(idx_arr[0])

    # Load n_obs metadata
    meta_path = data_dir / "pangea_n_obs_v03.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found")
    with open(meta_path) as f:
        n_obs_map = json.load(f)

    merged: Optional[pd.DataFrame] = None
    for ct in cts:
        npy_path = data_dir / f"pangea_corr_{ct}_v03.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"{npy_path} not found")

        # mmap: only reads the requested row from disk (~72 KB)
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
