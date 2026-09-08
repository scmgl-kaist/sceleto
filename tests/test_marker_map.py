"""Tests for ``HierarchyRun.marker_map`` (tree + marker grid, band/dot).

Builds a small ``HierarchyRun`` by hand (bypassing the marker/PAGA pipeline) so
the plotting + tree/row logic is exercised deterministically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sceleto.markers import HierarchyRun
from sceleto.markers.graph._context import MarkerContext


LEVELS = ["L0", "L1", "L2"]
GENES = [f"G{i}" for i in range(10)]

# 4 leaves; nested-ish paths so the tree has real bracket structure.
PATHS = {
    "0": ("L0@0", "L1@0", "L2@0"),
    "1": ("L0@0", "L1@0", "L2@1"),
    "2": ("L0@0", "L1@1", "L2@2"),
    "3": ("L0@1", "L1@2", "L2@3"),
}


def _ctx(groupby, group_ids, seed):
    rng = np.random.default_rng(seed)
    groups = list(group_ids)
    n, g = len(groups), len(GENES)
    mean = rng.random((n, g))
    mean_norm = mean / mean.max(axis=0, keepdims=True)  # per-gene 0..1
    frac = rng.random((n, g))
    return MarkerContext(
        groupby=groupby, groups=groups, genes=np.array(GENES),
        group_to_idx={c: i for i, c in enumerate(groups)},
        mean=mean, mean_norm=mean_norm,
        n_expr=(frac * 100).astype(int), frac_expr=frac,
        n_cells=np.full(n, 100),
    )


def _toy_hr():
    icls_full = {k: "|".join(v) for k, v in PATHS.items()}
    rows = []
    for icls, (a, b, c) in PATHS.items():
        rows.append([icls, icls_full[icls], a, b, c, a])
    icls_path_df = pd.DataFrame(
        rows, columns=["icls", "icls_full", "L0", "L1", "L2", "root"]
    )

    full = {}
    node_ids = {p[i] for p in PATHS.values() for i in range(3)}
    for j, nid in enumerate(sorted(node_ids)):
        full[nid] = GENES[j % len(GENES):] + GENES[: j % len(GENES)]

    contexts = {
        "L0": _ctx("L0", ["0", "1"], 1),
        "L1": _ctx("L1", ["0", "1", "2"], 2),
        "L2": _ctx("L2", ["0", "1", "2", "3"], 3),
    }
    return HierarchyRun(
        levels=LEVELS,
        params={"min_cells_for_path": 1, "n_top_markers": 4},
        icls_full_dict=icls_full,
        icls_path_df=icls_path_df,
        marker_rank_df=pd.DataFrame(),
        full_gene_lists=full,
        contexts=contexts,
        batch_expression=None,
        batch_key=None,
    )


# ---- single path (icls given) ----

def test_path_returns_fig_and_ax():
    hr = _toy_hr()
    fig, ax = hr.marker_map("0", n_markers=3)
    assert isinstance(fig, plt.Figure)
    assert ax is not None
    plt.close("all")


def test_path_dedup_reduces_rows():
    hr = _toy_hr()
    _, ax_keep = hr.marker_map("0", n_markers=4, dedup=False)
    _, ax_dedup = hr.marker_map("0", n_markers=4, dedup=True)
    n_keep = sum(1 for t in ax_keep.texts if t.get_text() in GENES)
    n_dedup = sum(1 for t in ax_dedup.texts if t.get_text() in GENES)
    assert n_dedup <= n_keep
    plt.close("all")


def test_path_column_subset_and_norm():
    hr = _toy_hr()
    for norm in ("global", "row"):
        fig, ax = hr.marker_map("0", n_markers=2, columns=["0", "1", "2"], color_norm=norm)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


def test_bad_icls_raises():
    hr = _toy_hr()
    with pytest.raises(ValueError):
        hr.marker_map("999")


# ---- full map (icls=None) ----

def test_full_map_default_and_orders():
    hr = _toy_hr()
    for order in ("dfs", "bfs"):
        fig, ax = hr.marker_map(n_markers=2, order=order)
        assert isinstance(fig, plt.Figure)
        # full map has more gene rows than a single path's 3 blocks
        n_gene_rows = sum(1 for t in ax.texts if t.get_text() in GENES)
        assert n_gene_rows > 3
        plt.close("all")


def test_full_map_bad_order_raises():
    hr = _toy_hr()
    with pytest.raises(ValueError):
        hr.marker_map(order="sideways")


# ---- modes / misc ----

def test_modes_band_and_dot():
    hr = _toy_hr()
    for scope in (None, "0"):
        for mode in ("band", "dot"):
            fig, ax = hr.marker_map(scope, n_markers=2, mode=mode)
            assert isinstance(fig, plt.Figure)
            plt.close("all")


def test_bad_mode_raises():
    hr = _toy_hr()
    with pytest.raises(ValueError):
        hr.marker_map(mode="triangle")


def test_no_tree_runs():
    hr = _toy_hr()
    fig, ax = hr.marker_map("3", n_markers=2, show_tree=False)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


# ---- dedup default (full dfs on, single path / bfs off) ----

def _n_gene_rows(ax):
    return sum(1 for t in ax.texts if t.get_text() in GENES)


def test_full_dfs_dedup_default_on():
    hr = _toy_hr()
    _, ax_default = hr.marker_map(n_markers=4, order="dfs")            # -> dedup on
    _, ax_off = hr.marker_map(n_markers=4, order="dfs", dedup=False)
    assert _n_gene_rows(ax_default) < _n_gene_rows(ax_off)
    plt.close("all")


def test_bfs_dedup_default_off():
    hr = _toy_hr()
    _, ax_default = hr.marker_map(n_markers=4, order="bfs")            # -> dedup off
    _, ax_off = hr.marker_map(n_markers=4, order="bfs", dedup=False)
    assert _n_gene_rows(ax_default) == _n_gene_rows(ax_off)
    plt.close("all")


def test_single_path_dedup_default_off():
    hr = _toy_hr()
    _, ax_default = hr.marker_map("0", n_markers=4)                    # -> dedup off
    _, ax_off = hr.marker_map("0", n_markers=4, dedup=False)
    assert _n_gene_rows(ax_default) == _n_gene_rows(ax_off)
    plt.close("all")
