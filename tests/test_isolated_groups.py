"""Isolated clusters in the PAGA graph must still get real markers.

A cluster whose kNN neighbors are all internal has PAGA connectivity 0 to every
other cluster. After top-k trimming it has no edge, so no FC/delta is computed
for it and every gene ties at score 0: the ranking silently became the
alphabetical order of gene names. ``build_context(isolated=...)`` now adds
fallback edges for such clusters.

Real-data case: ``~/temp.h5ad`` (pseudobulk, 16 leiden clusters). Cluster 13 is
Mast/Basophil (KIT/HDC/MS4A2), 12 is pDC (LILRA4/CLEC4C/TLR9); both are PAGA
islands. Skipped when the file is absent.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")

import sceleto as scl
from sceleto.markers.graph._context import build_context, _connect_isolated_groups


REAL = "/home/smcheong/temp.h5ad"
needs_real = pytest.mark.skipif(not os.path.exists(REAL), reason="temp.h5ad not present")


@pytest.fixture(scope="module")
def real():
    import anndata as ad
    return ad.read_h5ad(REAL)


# ----------------------------------------------------------------------------
# Unit: fallback edge construction on a hand-built graph
# ----------------------------------------------------------------------------

def _toy_adata(n_groups=5, d=3, seed=0):
    import anndata as ad
    rng = np.random.default_rng(seed)
    groups = [str(i) for i in range(n_groups)]
    obs = pd.DataFrame({"g": np.repeat(groups, 10)}, index=[f"c{i}" for i in range(10 * n_groups)])
    a = ad.AnnData(X=np.zeros((len(obs), 2), dtype=np.float32), obs=obs)
    # centroids on a line so nearest neighbors are unambiguous: group i at x=i
    emb = np.zeros((len(obs), d))
    emb[:, 0] = np.repeat(np.arange(n_groups, dtype=float), 10) + rng.normal(0, 1e-3, len(obs))
    a.obsm["X_pca"] = emb
    a.uns["neighbors"] = {"params": {"use_rep": "X_pca"}}
    return a, groups


def test_no_isolated_is_noop():
    a, groups = _toy_adata()
    edges = [("0", "1"), ("1", "2"), ("2", "3"), ("3", "4")]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out, iso, fb = _connect_isolated_groups(
            a, groupby="g", groups=groups, undirected_edges=edges, k=2, isolated="nearest"
        )
    assert out == edges and iso == [] and fb == []


def test_nearest_connects_to_k_nearest_centroids():
    a, groups = _toy_adata()
    edges = [("0", "1"), ("1", "2"), ("2", "3")]  # "4" isolated; nearest are 3 then 2
    with pytest.warns(UserWarning, match=r"Groups \['4'\]"):
        out, iso, fb = _connect_isolated_groups(
            a, groupby="g", groups=groups, undirected_edges=edges, k=2, isolated="nearest"
        )
    assert iso == ["4"]
    assert fb == [("2", "4"), ("3", "4")]
    assert set(out) == set(edges) | set(fb)


def test_all_connects_to_everyone():
    a, groups = _toy_adata()
    edges = [("0", "1"), ("1", "2"), ("2", "3")]
    with pytest.warns(UserWarning):
        out, iso, fb = _connect_isolated_groups(
            a, groupby="g", groups=groups, undirected_edges=edges, k=2, isolated="all"
        )
    assert fb == [("0", "4"), ("1", "4"), ("2", "4"), ("3", "4")]


def test_error_raises():
    a, groups = _toy_adata()
    with pytest.raises(ValueError, match="no PAGA edge"):
        _connect_isolated_groups(
            a, groupby="g", groups=groups, undirected_edges=[("0", "1")], k=2, isolated="error"
        )


def test_nearest_uses_use_rep_then_falls_back():
    a, groups = _toy_adata()
    del a.obsm["X_pca"]
    a.uns["neighbors"]["params"]["use_rep"] = "X_pca_harmony"  # absent
    with pytest.raises(ValueError, match="Cannot compute cluster centroids"):
        _connect_isolated_groups(
            a, groupby="g", groups=groups, undirected_edges=[("0", "1")], k=2, isolated="nearest"
        )
    a.obsm["X_umap"] = np.repeat(np.arange(len(groups), dtype=float), 10)[:, None].repeat(2, 1)
    with pytest.warns(UserWarning, match="X_umap"):
        _connect_isolated_groups(
            a, groupby="g", groups=groups, undirected_edges=[("0", "1")], k=1, isolated="nearest"
        )


# ----------------------------------------------------------------------------
# Real data: PAGA islands 12 (pDC) and 13 (Mast) in temp.h5ad
# ----------------------------------------------------------------------------

@needs_real
def test_real_islands_are_detected(real):
    with pytest.warns(UserWarning, match=r"Groups \['12', '13'\]"):
        ctx = build_context(real, "leiden", k=5)
    assert ctx.isolated_groups == ("12", "13")
    deg = {g: 0 for g in ctx.groups}
    for u, v in ctx.undirected_edges:
        deg[u] += 1
        deg[v] += 1
    assert deg["12"] == 5 and deg["13"] == 5
    assert all(e in ctx.undirected_edges for e in ctx.fallback_edges)

    with pytest.raises(ValueError, match="no PAGA edge"):
        build_context(real, "leiden", k=5, isolated="error")


@needs_real
@pytest.mark.parametrize("isolated", ["nearest", "all"])
def test_real_islands_get_biological_markers(real, isolated):
    with pytest.warns(UserWarning):
        r = scl.markers.marker(real, "leiden", isolated=isolated)

    mast = r.markers["13"][:15]
    pdc = r.markers["12"][:15]
    assert len({"KIT", "HDC", "MS4A2", "GATA2", "CPA3", "TPSAB1"} & set(mast)) >= 3, mast
    assert len({"LILRA4", "CLEC4C", "TLR9", "IL3RA", "GZMB", "TCL1A"} & set(pdc)) >= 2, pdc

    # not alphabetical any more
    assert mast != sorted(mast)
    assert pdc != sorted(pdc)

    sub = r.specific_ranking_df[r.specific_ranking_df["group"].astype(str) == "13"]
    assert (sub["max_fc"] > 0).any()
    assert sub["specific_weight"].max() > 0
