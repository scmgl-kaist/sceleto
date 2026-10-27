"""Isolated clusters in the PAGA graph must still get real markers.

A cluster whose kNN neighbors are all internal has PAGA connectivity 0 to every
other cluster. After top-k trimming it has no edge, so no FC/delta is computed
for it and every gene ties at score 0: the ranking silently became the
alphabetical order of gene names. ``build_context(isolated=...)`` now adds
fallback edges for such clusters.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")

import sceleto as scl
from sceleto.markers.graph._context import build_context, _connect_isolated_groups


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
# End-to-end: a PAGA island with planted markers
# ----------------------------------------------------------------------------

N_GROUPS = 6           # "5" is the island
ISLAND = "5"
CELLS_PER_GROUP = 40
N_FILLER = 120
# Island markers are named Z* so that the old failure (alphabetical ranking)
# would put them last, not first.
ISLAND_MARKERS = [f"ZISL{i}" for i in range(5)]


def _island_adata(seed=0):
    """log1p-like data; groups 0-4 form a PAGA chain, group 5 has no PAGA edge."""
    import anndata as ad
    from scipy import sparse

    rng = np.random.default_rng(seed)
    groups = [str(i) for i in range(N_GROUPS)]
    lab = np.repeat(groups, CELLS_PER_GROUP)
    n = lab.size

    own = {g: [f"M{g}_{j}" for j in range(5)] for g in groups if g != ISLAND}
    own[ISLAND] = ISLAND_MARKERS
    genes = [f"A{i:03d}" for i in range(N_FILLER)] + [x for g in groups for x in own[g]]
    col = {x: i for i, x in enumerate(genes)}

    # filler: sparse low background everywhere
    X = rng.random((n, len(genes))) * 0.3
    X[rng.random(X.shape) < 0.7] = 0.0
    for g in groups:
        rows = np.where(lab == g)[0]
        for x in own[g]:
            on = rows[rng.random(rows.size) < 0.9]
            X[on, col[x]] = 2.0 + rng.random(on.size)

    obs = pd.DataFrame({"leiden": pd.Categorical(lab, categories=groups)},
                       index=[f"c{i}" for i in range(n)])
    a = ad.AnnData(X=sparse.csr_matrix(X.astype(np.float32)), obs=obs,
                   var=pd.DataFrame(index=genes))

    # embedding: chain 0..4 along x, island far away
    cent = {g: np.array([float(i), 0.0, 0.0]) for i, g in enumerate(groups)}
    cent[ISLAND] = np.array([2.0, 10.0, 0.0])
    emb = np.vstack([cent[g] for g in lab]) + rng.normal(0, 0.1, (n, 3))
    a.obsm["X_pca"] = emb
    a.obsm["X_umap"] = emb[:, :2]
    a.uns["neighbors"] = {"params": {"use_rep": "X_pca"}}

    con = np.zeros((N_GROUPS, N_GROUPS))
    for i in range(N_GROUPS - 2):  # 0-1-2-3-4 chain
        con[i, i + 1] = con[i + 1, i] = 0.5
    a.uns["paga"] = {
        "connectivities": sparse.csr_matrix(con),
        "pos": np.vstack([cent[g][:2] for g in groups]),
        "groups": "leiden",
    }
    return a


def test_island_detected_and_connected():
    a = _island_adata()
    with pytest.warns(UserWarning, match=r"Groups \['5'\]"):
        ctx = build_context(a, "leiden", k=2)
    assert ctx.isolated_groups == (ISLAND,)
    # nearest by centroid: island sits above group 2, then 1/3 tie -> any 2 of {1,2,3}
    nb = {u if v == ISLAND else v for u, v in ctx.fallback_edges}
    assert len(nb) == 2 and "2" in nb and nb <= {"1", "2", "3"}

    with pytest.raises(ValueError, match="no PAGA edge"):
        build_context(a, "leiden", k=2, isolated="error")


@pytest.mark.parametrize("isolated", ["nearest", "all"])
def test_island_gets_planted_markers(isolated):
    a = _island_adata()
    with pytest.warns(UserWarning, match="no PAGA edge"):
        r = scl.markers.marker(a, "leiden", k=2, isolated=isolated)

    # auto thres_fc may drop one or two planted genes; whatever is returned on
    # top must be planted markers, not alphabetical filler (A***).
    top = r.markers[ISLAND][:3]
    assert len(top) == 3 and set(top) <= set(ISLAND_MARKERS), r.markers[ISLAND]

    sub = r.specific_ranking_df[r.specific_ranking_df["group"].astype(str) == ISLAND]
    assert (sub["max_fc"] > 0).any()
    assert sub["specific_weight"].max() > 0
