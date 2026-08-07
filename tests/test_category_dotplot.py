"""Tests for ``sceleto.category_dotplot`` (PyComplexHeatmap dot-heatmap)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import anndata as ad

import sceleto as scl

pytest.importorskip("PyComplexHeatmap")


def _toy_adata(n=300, seed=0):
    rng = np.random.default_rng(seed)
    genes = ["G1", "G2", "G3", "G4", "G5"]
    X = np.log1p(rng.poisson(0.6, size=(n, len(genes))).astype(float))
    obs = pd.DataFrame(
        {
            "row": pd.Categorical(rng.choice(["A", "B", "C"], size=n)),
            "cat": pd.Categorical(rng.choice(["x", "y"], size=n)),
        }
    )
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    a.raw = a
    return a


def test_returns_fig_and_plotter():
    a = _toy_adata()
    fig, cm = scl.category_dotplot(
        a, ["G1", "G2", "G3"], groupby="row", category="cat", show=False
    )
    assert isinstance(fig, plt.Figure)
    assert cm is not None
    plt.close("all")


def test_bracket_col_split_runs():
    a = _toy_adata()
    fig, cm = scl.category_dotplot(
        a,
        {"grpA": ["G1", "G2"], "grpB": ["G3", "G4", "G5"]},
        groupby="row", category="cat", show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_scale_options():
    a = _toy_adata()
    for scale in ("var", "group", None, False):
        fig, cm = scl.category_dotplot(
            a, ["G1", "G2"], groupby="row", category="cat",
            standard_scale=scale, show=False,
        )
        plt.close("all")


def test_max_scale_runs():
    a = _toy_adata()
    fig, cm = scl.category_dotplot(
        a, ["G1", "G2", "G3"], groupby="row", category="cat",
        max_scale=True, show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_max_scale_and_standard_scale_conflict():
    a = _toy_adata()
    with pytest.raises(ValueError):
        scl.category_dotplot(a, ["G1"], groupby="row", category="cat",
                             max_scale=True, standard_scale="var", show=False)


def test_subset_and_no_sidebar_no_split():
    a = _toy_adata()
    fig, cm = scl.category_dotplot(
        a, ["G1", "G2"], groupby="row", category="cat",
        groups=["A", "B"], categories=["x"],
        sidebar=False, row_split=False, show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_bad_scale_raises():
    a = _toy_adata()
    with pytest.raises(ValueError):
        scl.category_dotplot(a, ["G1"], groupby="row", category="cat",
                             standard_scale="nope", show=False)


def test_no_valid_genes_raises():
    a = _toy_adata()
    with pytest.raises(ValueError):
        scl.category_dotplot(a, ["NOPE"], groupby="row", category="cat", show=False)


def test_missing_obs_column_raises():
    a = _toy_adata()
    with pytest.raises(ValueError):
        scl.category_dotplot(a, ["G1"], groupby="row", category="missing", show=False)


def test_use_raw_none_raises():
    a = _toy_adata()
    a.raw = None
    with pytest.raises(ValueError):
        scl.category_dotplot(a, ["G1"], groupby="row", category="cat",
                             use_raw=True, show=False)
