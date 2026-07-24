"""Visualization of communication networks (matplotlib/seaborn).

Ports the most-used CellChat plots (``visualization.R``) in the sceleto house
style: each function accepts an optional ``ax``/``figsize``/``cmap`` and returns
the matplotlib ``Figure`` (or seaborn grid).

- :func:`plot_network_circle` — circular cell-group interaction diagram
  (``netVisual_circle``).
- :func:`plot_heatmap` — sender × receiver interaction heatmap
  (``netVisual_heatmap``).
- :func:`plot_bubble` — L–R pair × cell-pair bubble plot (``netVisual_bubble``).
- :func:`plot_chord` — chord diagram of group–group communication
  (``netVisual_chord_cell``); uses matplotlib arcs (no external chord lib).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle


def _check_square(mat: np.ndarray, names: Sequence[str]) -> None:
    """Validate that ``mat`` is a K×K matrix matching ``len(names)``."""
    mat = np.asarray(mat)
    K = len(names)
    if mat.shape != (K, K):
        raise ValueError(
            f"matrix shape {mat.shape} does not match len(names)={K}; "
            "expected a square (K, K) matrix aligned to the group names"
        )
    if K == 0:
        raise ValueError("cannot plot an empty network (no cell groups)")


def _palette(n: int) -> list:
    """n distinct colors from the bundled sceleto palettes."""
    from sceleto.data._colors import zeileis_26, godsnot_64

    base = zeileis_26 if n <= 26 else godsnot_64
    return [base[i % len(base)] for i in range(n)]


def plot_network_circle(
    mat: np.ndarray,
    names: Sequence[str],
    *,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (7, 7),
    edge_scale: float = 8.0,
    node_scale: float = 800.0,
    cmap: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Circular interaction diagram (``netVisual_circle``).

    Nodes are cell groups placed on a circle; a directed edge i→j is drawn with
    width proportional to ``mat[i, j]`` and colored by the sender.

    Parameters
    ----------
    mat
        ``K × K`` weight/count matrix (source × target).
    names
        Cell-group names.
    edge_scale, node_scale
        Visual scaling for edge widths and node sizes.
    """
    names = list(names)
    _check_square(mat, names)
    K = len(names)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    theta = np.linspace(0, 2 * np.pi, K, endpoint=False)
    pos = np.c_[np.cos(theta), np.sin(theta)]
    colors = _palette(K)

    mx = mat.max() if mat.max() > 0 else 1.0
    node_tot = mat.sum(axis=1) + mat.sum(axis=0)
    ntot_mx = node_tot.max() if node_tot.max() > 0 else 1.0

    # edges
    for i in range(K):
        for j in range(K):
            w = mat[i, j]
            if w <= 0:
                continue
            if i == j:
                # self-loop as a small circle offset outward
                c = Circle(pos[i] * 1.12, radius=0.06 * (0.5 + w / mx),
                           fill=False, edgecolor=colors[i],
                           lw=0.5 + edge_scale * w / mx, alpha=0.7)
                ax.add_patch(c)
                continue
            arrow = FancyArrowPatch(
                pos[i], pos[j],
                arrowstyle="-|>", mutation_scale=12,
                lw=0.4 + edge_scale * w / mx,
                color=colors[i], alpha=0.6,
                connectionstyle="arc3,rad=0.15",
                shrinkA=14, shrinkB=14,
            )
            ax.add_patch(arrow)

    # nodes
    ax.scatter(pos[:, 0], pos[:, 1],
               s=node_scale * (0.25 + node_tot / ntot_mx),
               c=colors, zorder=3, edgecolors="white", linewidths=1.0)
    for i, name in enumerate(names):
        ang = theta[i]
        ax.text(pos[i, 0] * 1.28, pos[i, 1] * 1.28, name,
                ha="center", va="center", fontsize=10,
                rotation=0)

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_heatmap(
    mat: np.ndarray,
    names: Sequence[str],
    *,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (6, 5),
    cmap: str = "Reds",
    title: Optional[str] = None,
) -> plt.Figure:
    """Sender × receiver heatmap of interaction strength (``netVisual_heatmap``)."""
    import seaborn as sns

    names = list(names)
    _check_square(mat, names)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    df = pd.DataFrame(mat, index=names, columns=names)
    sns.heatmap(df, cmap=cmap, ax=ax, cbar_kws={"label": "strength"},
                square=True, linewidths=0.3, linecolor="white")
    ax.set_xlabel("Target (receiver)")
    ax.set_ylabel("Source (sender)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_bubble(
    comm: pd.DataFrame,
    *,
    figsize: Optional[tuple[float, float]] = None,
    cmap: str = "Reds",
    title: Optional[str] = None,
) -> plt.Figure:
    """Bubble plot of L–R communication (``netVisual_bubble``).

    Parameters
    ----------
    comm
        Long-form table from :meth:`CellComm.subset_communication` (columns
        ``source``, ``target``, ``interaction_name``, ``prob``, ``pval``).
    """
    df = comm.copy()
    if len(df) == 0:
        # nothing significant to plot — return an empty annotated axes rather
        # than crashing inside scatter on object-dtype empty arrays
        fig, ax = plt.subplots(figsize=figsize or (5, 4))
        ax.text(0.5, 0.5, "no significant communication",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.axis("off")
        if title:
            ax.set_title(title)
        return fig
    df["pair"] = df["source"].astype(str) + " → " + df["target"].astype(str)
    x_order = sorted(df["pair"].unique())
    y_order = sorted(df["interaction_name"].unique())
    xi = {p: i for i, p in enumerate(x_order)}
    yi = {p: i for i, p in enumerate(y_order)}

    if figsize is None:
        figsize = (max(5, 0.5 * len(x_order) + 3), max(4, 0.3 * len(y_order) + 2))
    fig, ax = plt.subplots(figsize=figsize)

    x = df["pair"].map(xi).to_numpy()
    y = df["interaction_name"].map(yi).to_numpy()
    prob = df["prob"].to_numpy(dtype=float)
    # dot size ~ significance (−log10 p), color ~ probability
    size = 20 + 200 * (-np.log10(np.clip(df["pval"].to_numpy(dtype=float), 1e-4, 1)) / 4)
    sc = ax.scatter(x, y, s=size, c=prob, cmap=cmap, edgecolors="gray", linewidths=0.3)

    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(x_order, rotation=90, fontsize=8)
    ax.set_yticks(range(len(y_order)))
    ax.set_yticklabels(y_order, fontsize=8)
    ax.set_xlabel("Source → Target")
    fig.colorbar(sc, ax=ax, label="comm. prob")
    if title:
        ax.set_title(title)
    ax.margins(0.05)
    fig.tight_layout()
    return fig


def plot_chord(
    mat: np.ndarray,
    names: Sequence[str],
    *,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (7, 7),
    gap: float = 0.02,
    title: Optional[str] = None,
) -> plt.Figure:
    """Chord diagram of group→group communication (``netVisual_chord_cell``).

    A lightweight matplotlib implementation: each cell group gets an arc on the
    unit circle sized by its total interaction; ribbons connect senders to
    receivers with width proportional to the communication weight.
    """
    names = list(names)
    _check_square(mat, names)
    K = len(names)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"aspect": "equal"})
    else:
        fig = ax.get_figure()
    colors = _palette(K)

    totals = mat.sum(axis=1) + mat.sum(axis=0)
    total = totals.sum()
    if total == 0:
        ax.axis("off")
        return fig
    frac = totals / total
    gaps = gap * 2 * np.pi
    span = 2 * np.pi - K * gaps
    starts = np.zeros(K)
    acc = 0.0
    arc = {}
    for i in range(K):
        a0 = acc
        a1 = acc + frac[i] * span
        arc[i] = (a0, a1)
        acc = a1 + gaps
        # draw the group arc
        ts = np.linspace(a0, a1, 40)
        ax.plot(np.cos(ts), np.sin(ts), lw=8, color=colors[i], solid_capstyle="butt")
        mid = (a0 + a1) / 2
        ax.text(1.18 * np.cos(mid), 1.18 * np.sin(mid), names[i],
                ha="center", va="center", fontsize=10)

    # ribbons: place each i→j chord at a sub-arc position within i and j
    mx = mat.max() if mat.max() > 0 else 1.0
    for i in range(K):
        a0, a1 = arc[i]
        cur = a0
        w_i = mat[i].sum() + mat[:, i].sum()
        for j in range(K):
            w = mat[i, j]
            if w <= 0:
                continue
            ai = (a0 + a1) / 2
            aj = (arc[j][0] + arc[j][1]) / 2
            p0 = np.array([np.cos(ai), np.sin(ai)]) * 0.98
            p1 = np.array([np.cos(aj), np.sin(aj)]) * 0.98
            arrow = FancyArrowPatch(
                p0, p1, connectionstyle="arc3,rad=0.4",
                arrowstyle="-", lw=0.5 + 6 * w / mx,
                color=colors[i], alpha=0.45,
            )
            ax.add_patch(arrow)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_rank_net(
    ranking: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple[float, float]] = None,
    color: str = "#4C72B0",
    title: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar plot of pathway information flow (``rankNet``).

    Parameters
    ----------
    ranking
        Output of :meth:`CellComm.rank_net` — columns ``pathway_name`` and
        ``flow``, already sorted descending.
    """
    df = ranking.iloc[::-1]  # so the largest ends up on top
    if figsize is None:
        figsize = (5, max(3, 0.3 * len(df) + 1))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    ax.barh(df["pathway_name"].astype(str), df["flow"].to_numpy(dtype=float), color=color)
    ax.set_xlabel("Information flow")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_signaling_role_heatmap(
    role_matrix: pd.DataFrame,
    *,
    figsize: Optional[tuple[float, float]] = None,
    cmap: str = "OrRd",
    title: Optional[str] = None,
) -> plt.Figure:
    """Pathway × cell-group signaling-role heatmap (``netAnalysis_signalingRole_heatmap``).

    Parameters
    ----------
    role_matrix
        Output of :func:`sceleto.interaction.signaling_role_pathway` — rows =
        pathways, columns = cell groups.
    """
    import seaborn as sns

    if figsize is None:
        figsize = (max(4, 0.4 * role_matrix.shape[1] + 2),
                   max(4, 0.3 * role_matrix.shape[0] + 1))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(role_matrix, cmap=cmap, ax=ax, linewidths=0.3, linecolor="white",
                cbar_kws={"label": "importance"})
    ax.set_xlabel("Cell group")
    ax.set_ylabel("Signaling pathway")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_contribution(
    contrib: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[tuple[float, float]] = None,
    color: str = "#55A868",
    title: Optional[str] = None,
) -> plt.Figure:
    """Bar plot of each L–R pair's contribution to a pathway (``netAnalysis_contribution``).

    Parameters
    ----------
    contrib
        Output of :meth:`CellComm.contribution` — columns ``interaction_name``
        and ``contribution``.
    """
    df = contrib.iloc[::-1]
    if figsize is None:
        figsize = (5, max(2.5, 0.3 * len(df) + 1))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    ax.barh(df["interaction_name"].astype(str),
            df["contribution"].to_numpy(dtype=float), color=color)
    ax.set_xlabel("Relative contribution")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
