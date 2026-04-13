"""Preprocessing functions (ported from scjp, modernized)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import scanpy as sc


def _load_cc_genes():
    """Load cell cycle gene list from gene_categories.json."""
    import json
    from pathlib import Path
    path = Path(__file__).parent / "data" / "gene_categories.json"
    with open(path) as f:
        d = json.load(f)
    return d["Cell_Cycle"]


def remove_geneset(adata, geneset):
    """Remove genes in *geneset* from *adata* and return a copy."""
    return adata[:, ~adata.var_names.isin(list(geneset))].copy()


def sc_process(adata, steps: str = "fspkuc", n_pcs: int = 50):
    """Scanpy preprocessing pipeline controlled by a step string.

    Each letter in *steps* triggers one preprocessing step, executed in order:

    ===  ============================
    n    normalize_total (1e4)
    l    log1p + store .raw
    f    highly_variable_genes + filter
    r    remove cell-cycle genes
    s    scale (max_value=10)
    p    PCA
    k    kNN neighbors
    u    UMAP
    c    leiden clustering
    ===  ============================

    Parameters
    ----------
    adata : AnnData
    steps : str
        Letters selecting which steps to run. Default ``"fspkuc"``.
    n_pcs : int
        Number of PCs for neighbor search. Default 50.
    """
    if "n" in steps:
        sc.pp.normalize_total(adata, target_sum=1e4)
    if "l" in steps:
        sc.pp.log1p(adata)
        adata.raw = adata
        print("adding raw...")
    if "f" in steps:
        if adata.raw is None:
            adata.raw = adata
            print("adding raw...")
        sc.pp.highly_variable_genes(adata)
        adata = adata[:, adata.var.highly_variable].copy()
    if "r" in steps:
        cc_genes = _load_cc_genes()
        adata = remove_geneset(adata, cc_genes)
        print("removing cc_genes...")
    if "s" in steps:
        sc.pp.scale(adata, max_value=10)
    if "p" in steps:
        sc.pp.pca(adata)
    if "k" in steps:
        sc.pp.neighbors(adata, n_pcs=n_pcs)
    if "u" in steps:
        sc.tl.umap(adata)
    if "c" in steps:
        sc.tl.leiden(adata)
    return adata


def read_process(
    adata,
    version: str,
    *,
    species: str = "human",
    sample: Optional[str] = None,
    define_var: bool = True,
    call_doublet: bool = True,
    write: bool = True,
    min_n_counts: int = 1000,
    min_n_genes: int = 500,
    max_n_genes: int = 7000,
    max_pct_mito: float = 0.5,
):
    """QC filtering + optional doublet detection + write.

    Parameters
    ----------
    adata : AnnData
        Raw count matrix.
    version : str
        Version tag for the output filename.
    species : str
        ``"human"`` or ``"mouse"`` (determines mito gene prefix).
    sample : str, optional
        Sample name stored in ``adata.obs["Sample"]``.
    define_var : bool
        If True, copy gene names / Ensembl IDs into ``adata.var``.
    call_doublet : bool
        If True, run scrublet for doublet detection (lazy import).
    write : bool
        If True, save filtered adata as h5ad.
    min_n_counts, min_n_genes, max_n_genes : int
        Cell-level count / gene number thresholds.
    max_pct_mito : float
        Maximum mitochondrial fraction (0–1).
    """
    if sample:
        adata.obs["Sample"] = sample
    if define_var:
        adata.var["GeneName"] = list(adata.var.gene_ids.index)
        adata.var["EnsemblID"] = list(adata.var.gene_ids)

    # QC metrics via scanpy
    mito_prefix = {"human": "MT-", "mouse": "mt-"}
    if species not in mito_prefix:
        raise ValueError(f"Unknown species: {species!r}. Use 'human' or 'mouse'.")

    adata.var["mt"] = adata.var_names.str.startswith(mito_prefix[species])
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    print(f"calculating mito... as species = {species}")
    print(
        f"filtering cells... >{min_n_counts} counts, "
        f"{min_n_genes}–{max_n_genes} genes, <{max_pct_mito} pct_mito..."
    )

    keep = (
        (adata.obs["total_counts"] > min_n_counts)
        & (adata.obs["n_genes_by_counts"] > min_n_genes)
        & (adata.obs["n_genes_by_counts"] < max_n_genes)
        & (adata.obs["pct_counts_mt"] / 100 < max_pct_mito)
    )
    adata = adata[keep].copy()

    if call_doublet:
        import scrublet as scr
        print("calling doublets using scrublet...")
        scrub = scr.Scrublet(adata.X)
        doublet_scores, predicted_doublets = scrub.scrub_doublets(verbose=False)
        adata.obs["doublet_scores"] = doublet_scores
        adata.obs["predicted_doublets"] = predicted_doublets

    if write:
        out_path = f"{version}{sample}_filtered.h5ad"
        print(f"writing output to {out_path} ...")
        adata.write_h5ad(out_path)
    return adata
