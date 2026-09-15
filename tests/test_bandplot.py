"""Tests for ``sceleto.bandplot`` (compact band-style dotplot alternative)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import anndata as ad

import sceleto as scl


def _toy_adata(n=300, seed=0):
    rng = np.random.default_rng(seed)
    genes = ["G1", "G2", "G3", "G4", "G5"]
    X = np.log1p(rng.poisson(0.6, size=(n, len(genes))).astype(float))
    obs = pd.DataFrame(
        {"grp": pd.Categorical(rng.choice(["A", "B", "C"], size=n))}
    )
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    a.raw = a
    return a


def test_returns_fig_and_ax():
    a = _toy_adata()
    fig, ax = scl.bandplot(a, ["G1", "G2", "G3"], "grp", show=False)
    assert isinstance(fig, plt.Figure)
    assert ax is not None
    # one rectangle per non-empty (gene, group) cell (+ legend patches)
    assert len(ax.patches) > 0
    plt.close("all")


def test_swap_axes():
    a = _toy_adata()
    fig, ax = scl.bandplot(a, ["G1", "G2", "G3"], "grp", swap_axes=True, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_max_scale_option():
    a = _toy_adata()
    for ms in (True, False):
        fig, ax = scl.bandplot(a, ["G1", "G2"], "grp", max_scale=ms, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


def test_bracket_mapping():
    a = _toy_adata()
    fig, ax = scl.bandplot(
        a, {"blockA": ["G1", "G2"], "blockB": ["G3", "G4", "G5"]}, "grp", show=False
    )
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_groups_subset_and_use_raw_false():
    a = _toy_adata()
    fig, ax = scl.bandplot(
        a, ["G1", "G2"], "grp", groups=["A", "C"], use_raw=False, show=False
    )
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_missing_genes_raises():
    a = _toy_adata()
    with pytest.raises(ValueError):
        scl.bandplot(a, ["NOPE1", "NOPE2"], "grp", show=False)


def test_color_floor_and_vmax():
    a = _toy_adata()
    fig, ax = scl.bandplot(
        a, ["G1", "G2", "G3"], "grp", color_floor=0.25, vmax=1.0, show=False
    )
    assert isinstance(fig, plt.Figure)
    plt.close("all")
