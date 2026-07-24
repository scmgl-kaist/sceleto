"""Additional visualizations of communication networks (matplotlib/seaborn).

Second batch of CellChat plots (``visualization.R`` / ``analysis.R``), ported in
the same sceleto house style as :mod:`sceleto.interaction._viz`: each function
accepts an optional ``ax``/``figsize``/``cmap`` where meaningful and returns the
matplotlib ``Figure``. Degenerate inputs (no significant communication, a single
group or pathway) yield an annotated empty axes rather than an exception, exactly
like :func:`sceleto.interaction._viz.plot_bubble`.

- :func:`plot_signaling_role_scatter` — dominant sender/receiver 2-D scatter
  (``netAnalysis_signalingRole_scatter``).
- :func:`plot_signaling_role_network` — 4-role centrality heatmap strip for one
  pathway or the aggregated network (``netAnalysis_signalingRole_network``).
- :func:`plot_chord_gene` — ligand→receptor (gene-level) chord diagram
  (``netVisual_chord_gene``); matplotlib arcs, no external chord library.
- :func:`plot_communication_patterns_river` — cell-groups ↔ patterns ↔ pathways
  alluvial/bipartite diagram (``netAnalysis_river``).
- :func:`plot_communication_patterns_dot` — cell-group × pattern (or pathway)
  loading dot plot (``netAnalysis_dot``).
- :func:`plot_net_embedding_zoom` — pathway-similarity embedding scatter, zoomed
  to one cluster (``netVisual_embeddingZoomIn``).
- :func:`plot_aggregate` — dispatcher that draws one named pathway's K×K network
  as a circle or chord (``netVisual_aggregate``).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from ._viz import _palette, plot_network_circle, plot_chord


def _empty_axes(
    message: str,
    *,
    figsize: Optional[tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Annotated blank axes for degenerate input (mirrors ``plot_bubble``)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (5, 4))
    else:
        fig = ax.get_figure()
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, color="gray")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return fig


def _pathway_index(cc, signaling: str) -> int:
    """Index of ``signaling`` in ``cc.netP['pathways']`` (raises if absent)."""
    pathways = [str(p) for p in cc.netP.get("pathways", [])]
    if str(signaling) not in pathways:
        raise ValueError(
            f"pathway {signaling!r} not found; available pathways: {pathways}"
        )
    return pathways.index(str(signaling))


def _signaling_link_count(cc, pathways, shape, *, thresh: float = 0.05):
    """K×K count of significant L-R pairs within the given pathways.

    Mirrors R ``aggregateNet(signaling=...)$count``: for each source→target
    group pair, the number of significant (``pval < thresh``) L-R interactions
    whose pathway is in ``pathways``. Falls back to zeros if the per-LR net /
    LRsig are unavailable.
    """
    net = getattr(cc, "net", {}) or {}
    lrsig = getattr(cc, "LRsig", None)
    if "pval" not in net or "prob" not in net or lrsig is None:
        return np.zeros(shape)
    pval = np.asarray(net["pval"], dtype=float)
    prob = np.asarray(net["prob"], dtype=float)
    pw = lrsig["pathway_name"].astype(str).to_numpy()
    want = set(map(str, pathways))
    sel = np.array([p in want for p in pw])
    if not sel.any():
        return np.zeros(shape)
    sig = (pval[:, :, sel] < thresh) & (prob[:, :, sel] > 0)
    return sig.sum(axis=2)


# ---------------------------------------------------------------------------
# 1. netAnalysis_signalingRole_scatter
# ---------------------------------------------------------------------------
def plot_signaling_role_scatter(
    cc,
    signaling: Optional[Sequence[str]] = None,
    *,
    thresh: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (6, 6),
    dot_scale: float = 400.0,
    label: bool = True,
    font_size: int = 9,
    title: Optional[str] = None,
) -> plt.Figure:
    """Dominant sender/receiver 2-D scatter (``netAnalysis_signalingRole_scatter``).

    Each cell group is a point at ``(outgoing strength, incoming strength)``:
    ``x`` is the total outgoing communication (row sums of the weight matrix,
    i.e. the group as sender) and ``y`` the total incoming communication (column
    sums, the group as receiver). Dot size is proportional to the total number
    of inferred links (outgoing + incoming), and dot color encodes the cell
    group. This is CellChat's dominant sender/receiver map.

    Parameters
    ----------
    cc
        A :class:`~sceleto.interaction.CellComm` after
        :meth:`~sceleto.interaction.CellComm.compute_communication` and
        :meth:`~sceleto.interaction.CellComm.aggregate` (needs
        ``cc.net['weight']`` and ``cc.net['count']``).
    signaling
        Restrict to one or more pathway names (sum over just those pathways). If
        ``None`` (default) the aggregated network over all pathways is used.
    thresh
        p-value threshold for counting significant links in the ``signaling=...``
        path (dot size). If ``None`` (default) it matches the threshold used by
        :meth:`~sceleto.interaction.CellComm.compute_communication_pathway`, so
        dot size and dot position always share the same significance cutoff.
        Ignored when ``signaling is None``.
    ax
        Existing axes to draw on; a new figure is created if ``None``.
    figsize
        Figure size when creating a new figure.
    dot_scale
        Overall scaling of the dot area (links → point size).
    label
        Annotate each point with its cell-group name.
    font_size
        Font size for the point labels.
    title
        Axes title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    names = list(getattr(cc, "group_names", []))
    net = getattr(cc, "net", {}) or {}

    if signaling is not None:
        # sum the per-pathway K×K prob over the requested pathways
        netP = getattr(cc, "netP", {}) or {}
        if "prob" not in netP:
            raise RuntimeError(
                "signaling=... requires compute_communication_pathway() first."
            )
        wanted = [str(s) for s in signaling]
        idx = [i for i, p in enumerate(netP.get("pathways", []))
               if str(p) in set(wanted)]
        if not idx:
            return _empty_axes(
                "no significant communication for the requested signaling",
                figsize=figsize, ax=ax, title=title,
            )
        prob = np.asarray(netP["prob"], dtype=float)[:, :, idx]
        weight = prob.sum(axis=2)
        # count = number of significant L-R pairs per group-pair in these
        # pathways (R aggregateNet(signaling=...)$count) — NOT (netP prob>0),
        # which would only count pathways-with-signal. Recover it from the
        # per-LR net arrays filtered to the requested pathways. Match the
        # threshold that compute_communication_pathway used for `weight` above
        # (stored in netP['thresh']) so dot size and position stay in sync;
        # an explicit `thresh` overrides it.
        link_thresh = thresh if thresh is not None else netP.get("thresh", 0.05)
        count = _signaling_link_count(cc, wanted, weight.shape, thresh=link_thresh)
    else:
        if "weight" not in net or "count" not in net:
            raise RuntimeError(
                "cc.net['weight']/['count'] missing; run "
                "compute_communication() then aggregate() first."
            )
        weight = np.asarray(net["weight"], dtype=float)
        count = np.asarray(net["count"], dtype=float)

    K = len(names)
    if K == 0 or weight.size == 0 or weight.sum() == 0:
        return _empty_axes("no significant communication",
                           figsize=figsize, ax=ax, title=title)

    outgoing = weight.sum(axis=1)     # sender strength (row sums)
    incoming = weight.sum(axis=0)     # receiver strength (col sums)
    # number of links per group (out + in), diagonal counted once (as in R:
    # rowSums + colSums − diag).
    num_link = count.sum(axis=1) + count.sum(axis=0) - np.diag(count)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    colors = _palette(K)

    nl_mx = num_link.max() if num_link.max() > 0 else 1.0
    sizes = dot_scale * (0.15 + num_link / nl_mx)
    ax.scatter(outgoing, incoming, s=sizes, c=colors,
               edgecolors="white", linewidths=0.8, alpha=0.85, zorder=3)

    if label:
        for i, name in enumerate(names):
            ax.annotate(
                str(name), (outgoing[i], incoming[i]),
                fontsize=font_size, ha="center", va="bottom",
                xytext=(0, 5), textcoords="offset points",
                color=colors[i], zorder=4,
            )

    ax.set_xlabel("Outgoing interaction strength")
    ax.set_ylabel("Incoming interaction strength")
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    ax.axvline(0, color="0.85", lw=0.6, zorder=0)
    ax.margins(0.15)
    if title is None:
        title = ("Signaling role — "
                 + (", ".join(str(s) for s in signaling)
                    if signaling is not None else "all pathways"))
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. netAnalysis_signalingRole_network
# ---------------------------------------------------------------------------
def plot_signaling_role_network(
    cc,
    signaling: Optional[str] = None,
    *,
    figsize: Optional[tuple[float, float]] = None,
    cmap: str = "BuGn",
    title: Optional[str] = None,
) -> plt.Figure:
    """4-role centrality heatmap strip (``netAnalysis_signalingRole_network``).

    Computes the four core CellChat centrality roles — Sender (out-degree),
    Receiver (in-degree), Mediator (flow betweenness), Influencer (information)
    — per cell group for one signaling pathway (or the aggregated network) via
    :func:`~sceleto.interaction.compute_centrality`, row-normalizes each role by
    its maximum (as R does with ``sweep(mat, 1, max, '/')``), and shows the
    ``roles × cell-groups`` importance strip.

    Parameters
    ----------
    cc
        A :class:`~sceleto.interaction.CellComm` after
        :meth:`~sceleto.interaction.CellComm.compute_communication_pathway`
        (for a named ``signaling``) or after
        :meth:`~sceleto.interaction.CellComm.aggregate` (for the aggregated
        network when ``signaling=None``).
    signaling
        Pathway name (from ``cc.netP['pathways']``). If ``None`` (default) the
        aggregated ``cc.net['weight']`` network is used.
    figsize
        Figure size; a sensible size is derived from the number of groups.
    cmap
        Matplotlib colormap (default ``"BuGn"``, matching R).
    title
        Heatmap title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import seaborn as sns

    from ._analysis import compute_centrality

    names = list(getattr(cc, "group_names", []))
    if signaling is not None:
        netP = getattr(cc, "netP", {}) or {}
        if "prob" not in netP:
            raise RuntimeError(
                "signaling=... requires compute_communication_pathway() first."
            )
        p = _pathway_index(cc, signaling)
        mat = np.asarray(netP["prob"], dtype=float)[:, :, p]
        ttl = f"{signaling} signaling pathway network"
    else:
        net = getattr(cc, "net", {}) or {}
        if "weight" not in net:
            raise RuntimeError(
                "cc.net['weight'] missing; run compute_communication() then "
                "aggregate(), or pass a `signaling` pathway name."
            )
        mat = np.asarray(net["weight"], dtype=float)
        ttl = "Aggregated signaling network"

    K = len(names)
    if K == 0 or mat.size == 0 or mat.sum() == 0:
        return _empty_axes("no significant communication",
                           figsize=figsize or (6, 2), title=title or ttl)

    cen = compute_centrality(mat, names)
    role_cols = ["outdeg", "indeg", "flowbet", "info"]
    role_names = ["Sender", "Receiver", "Mediator", "Influencer"]
    role_mat = cen[role_cols].T
    role_mat.index = role_names
    # row-normalize each role by its max (R: sweep(mat, 1, max, '/'))
    row_max = role_mat.max(axis=1).replace(0.0, np.nan)
    role_mat = role_mat.div(row_max, axis=0).fillna(0.0)

    if figsize is None:
        figsize = (max(5, 0.5 * K + 2), 2.6)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(role_mat, cmap=cmap, ax=ax, vmin=0.0, vmax=1.0,
                linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Importance"})
    ax.set_xlabel("Cell group")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelrotation=0)
    ax.set_title(title if title is not None else ttl)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. netVisual_chord_gene (gene / L-R level chord)
# ---------------------------------------------------------------------------
def plot_chord_gene(
    cc,
    *,
    sources: Optional[Sequence[str]] = None,
    targets: Optional[Sequence[str]] = None,
    signaling: Optional[Sequence[str]] = None,
    thresh: float = 0.05,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (8, 8),
    gap: float = 0.015,
    max_pairs: int = 40,
    font_size: int = 8,
    title: Optional[str] = None,
) -> plt.Figure:
    """Gene-level ligand→receptor chord diagram (``netVisual_chord_gene``).

    Distinct from the cell-level :func:`sceleto.interaction.plot_chord`: sectors
    around the circle are the ligand and receptor *genes* involved, and each
    ribbon is a significant ligand→receptor link, colored by the sending cell
    group. This reveals which L–R pairs drive the communication between the
    chosen sources and targets. Built from
    :meth:`~sceleto.interaction.CellComm.subset_communication`.

    Parameters
    ----------
    cc
        A :class:`~sceleto.interaction.CellComm` after
        :meth:`~sceleto.interaction.CellComm.compute_communication`.
    sources, targets
        Restrict to these sender / receiver cell groups (names). ``None`` keeps
        all.
    signaling
        Restrict to these signaling pathway names. ``None`` keeps all.
    thresh
        p-value threshold for a link to be shown.
    ax
        Existing axes; a new figure is created if ``None``.
    figsize
        Figure size when creating a new figure.
    gap
        Angular gap between gene sectors, as a fraction of the full circle.
    max_pairs
        Cap on the number of ligand→receptor links drawn (the strongest by
        probability are kept) to keep the diagram legible.
    font_size
        Gene-label font size.
    title
        Axes title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    comm = cc.subset_communication(
        thresh=thresh, sources=sources, targets=targets, signaling=signaling
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"aspect": "equal"})
    else:
        fig = ax.get_figure()

    if len(comm) == 0:
        return _empty_axes("no significant communication",
                           ax=ax, title=title)

    # aggregate identical ligand→receptor links across cell pairs, keeping the
    # dominant (highest-prob) sending group for the ribbon color.
    df = comm.copy()
    if len(df) > max_pairs:
        df = df.sort_values("prob", ascending=False).head(max_pairs)

    # sectors: ligand genes on one side, receptor genes on the other, so that a
    # ribbon always crosses the circle (ligand → receptor). A gene that is both
    # a ligand and a receptor gets one sector on each side (suffixed) to keep
    # the bipartite reading.
    ligands = list(pd.unique(df["ligand"].astype(str)))
    receptors = list(pd.unique(df["receptor"].astype(str)))
    lig_sectors = [("L", g) for g in ligands]
    rec_sectors = [("R", g) for g in receptors]
    sectors = lig_sectors + rec_sectors
    S = len(sectors)
    if S == 0:
        return _empty_axes("no significant communication", ax=ax, title=title)

    sector_idx = {s: i for i, s in enumerate(sectors)}

    gaps = gap * 2 * np.pi
    span = 2 * np.pi - S * gaps
    per = span / S
    centers = {}
    acc = 0.0
    for s in sectors:
        a0 = acc
        a1 = acc + per
        mid = (a0 + a1) / 2.0
        centers[s] = mid
        # draw sector arc
        ts = np.linspace(a0, a1, 20)
        kind = s[0]
        col = "#4C72B0" if kind == "L" else "#C44E52"
        ax.plot(np.cos(ts), np.sin(ts), lw=7, color=col, solid_capstyle="butt",
                zorder=2)
        ax.text(1.18 * np.cos(mid), 1.18 * np.sin(mid), s[1],
                ha="center", va="center", fontsize=font_size,
                rotation=np.degrees(mid) - 90 if -np.pi / 2 < mid < np.pi / 2
                else np.degrees(mid) + 90)
        acc = a1 + gaps

    # color ribbons by sending cell group
    groups = list(getattr(cc, "group_names", []))
    gcolors = {g: c for g, c in zip(groups, _palette(max(1, len(groups))))}

    mx = df["prob"].max() if df["prob"].max() > 0 else 1.0
    for _, row in df.iterrows():
        a_l = centers[("L", str(row["ligand"]))]
        a_r = centers[("R", str(row["receptor"]))]
        p0 = np.array([np.cos(a_l), np.sin(a_l)]) * 0.97
        p1 = np.array([np.cos(a_r), np.sin(a_r)]) * 0.97
        w = float(row["prob"])
        arrow = FancyArrowPatch(
            p0, p1, connectionstyle="arc3,rad=0.35",
            arrowstyle="-", lw=0.4 + 6 * w / mx,
            color=gcolors.get(str(row["source"]), "gray"),
            alpha=0.45, zorder=1,
        )
        ax.add_patch(arrow)

    # legend: sending groups actually present
    present = list(pd.unique(df["source"].astype(str)))
    handles = [plt.Line2D([0], [0], color=gcolors.get(g, "gray"), lw=3)
               for g in present]
    if handles:
        ax.legend(handles, present, title="Sender", fontsize=font_size,
                  frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")
    if title is None:
        title = "Ligand → receptor communication"
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. netAnalysis_river
# ---------------------------------------------------------------------------
def plot_communication_patterns_river(
    cc,
    pattern: str = "outgoing",
    *,
    result: Optional[dict] = None,
    cutoff: float = 0.5,
    figsize: tuple[float, float] = (10, 7),
    font_size: int = 8,
    title: Optional[str] = None,
) -> plt.Figure:
    """Cell-groups ↔ patterns ↔ pathways alluvial diagram (``netAnalysis_river``).

    A three-column linked-bipartite ("river") diagram: the left column is the
    cell groups, the middle column the learned communication patterns, and the
    right column the signaling pathways. Bands connect a cell group to a pattern
    with width proportional to its ``W`` loading, and a pattern to a pathway with
    width proportional to its ``H`` loading; loadings below ``cutoff`` (after
    per-column scaling in :func:`identify_communication_patterns`) are dropped.

    Parameters
    ----------
    cc
        A :class:`~sceleto.interaction.CellComm`. Uses ``cc.patterns[pattern]``
        unless a ``result`` dict is passed directly.
    pattern
        ``"outgoing"`` or ``"incoming"``.
    result
        A result dict from
        :func:`~sceleto.interaction.identify_communication_patterns`. If
        ``None``, read ``cc.patterns[pattern]``.
    cutoff
        Loadings below this are treated as zero (no band drawn), matching R's
        ``cutoff = 0.5``.
    figsize
        Figure size.
    font_size
        Label font size.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if result is None:
        patterns = getattr(cc, "patterns", None)
        if not patterns or pattern not in patterns:
            return _empty_axes(
                f"no {pattern!r} pattern result; run "
                "identify_communication_patterns(...) first",
                figsize=figsize, title=title,
            )
        result = patterns[pattern]

    W = result["W"]           # cell groups × patterns (rows sum to 1)
    H = result["H"]           # patterns × pathways (cols sum to 1)
    cell_groups = list(W.index)
    pat_names = list(W.columns)
    pathways = list(H.columns)

    if len(cell_groups) == 0 or len(pat_names) == 0 or len(pathways) == 0:
        return _empty_axes("empty communication patterns",
                           figsize=figsize, title=title)

    fig, ax = plt.subplots(figsize=figsize)

    # three x-columns
    x_cell, x_pat, x_path = 0.0, 1.0, 2.0
    bar_w = 0.12

    def _layout(labels, x):
        """Stacked unit-height blocks for one column; returns {label:(y0,y1)}."""
        n = len(labels)
        if n == 0:
            return {}
        gap = 0.02
        h = (1.0 - gap * (n - 1)) / n
        pos = {}
        y = 1.0
        for lab in labels:
            pos[lab] = (y - h, y)
            y -= h + gap
        return pos

    cell_pos = _layout(cell_groups, x_cell)
    pat_pos = _layout(pat_names, x_pat)
    path_pos = _layout(pathways, x_path)

    cell_colors = {g: c for g, c in zip(cell_groups, _palette(len(cell_groups)))}
    # patterns colored from a separate palette
    pat_palette = _palette(max(len(pat_names) * 2, 1))
    pat_colors = {p: pat_palette[i * 2 % len(pat_palette)]
                  for i, p in enumerate(pat_names)}

    def _bar(pos, x, colors):
        for lab, (y0, y1) in pos.items():
            ax.add_patch(plt.Rectangle((x - bar_w / 2, y0), bar_w, y1 - y0,
                                       facecolor=colors.get(lab, "gray"),
                                       edgecolor="black", lw=0.4, zorder=3))
            ax.text(x, (y0 + y1) / 2, str(lab), ha="center", va="center",
                    fontsize=font_size, zorder=4, rotation=90)

    _bar(cell_pos, x_cell, cell_colors)
    _bar(pat_pos, x_pat, pat_colors)
    _bar(path_pos, x_path, {p: "0.6" for p in pathways})

    def _band(x0, y0lo, y0hi, x1, y1lo, y1hi, color):
        """Filled cubic band between two vertical segments."""
        t = np.linspace(0, 1, 30)
        xs = x0 + (x1 - x0) * (3 * t ** 2 - 2 * t ** 3)  # smoothstep
        top = y0hi + (y1hi - y0hi) * (3 * t ** 2 - 2 * t ** 3)
        bot = y0lo + (y1lo - y0lo) * (3 * t ** 2 - 2 * t ** 3)
        ax.fill_between(xs, bot, top, color=color, alpha=0.4, lw=0, zorder=1)

    # left bands: cell group → pattern, thickness ~ W loading
    # allocate each block's height proportionally to its outgoing/incoming loads
    for g in cell_groups:
        gy0, gy1 = cell_pos[g]
        gh = gy1 - gy0
        loads = np.array([W.loc[g, p] if W.loc[g, p] >= cutoff else 0.0
                          for p in pat_names], dtype=float)
        tot = loads.sum()
        if tot <= 0:
            continue
        cur = gy1
        for p, load in zip(pat_names, loads):
            if load <= 0:
                continue
            frac = load / tot
            seg = gh * frac
            # target sub-slot inside the pattern block (proportional split)
            py0, py1 = pat_pos[p]
            _band(x_cell + bar_w / 2, cur - seg, cur,
                  x_pat - bar_w / 2, py0, py1, cell_colors[g])
            cur -= seg

    # right bands: pattern → pathway, thickness ~ H loading
    for p in pat_names:
        py0, py1 = pat_pos[p]
        ph = py1 - py0
        loads = np.array([H.loc[p, pw] if H.loc[p, pw] >= cutoff else 0.0
                          for pw in pathways], dtype=float)
        tot = loads.sum()
        if tot <= 0:
            continue
        cur = py1
        for pw, load in zip(pathways, loads):
            if load <= 0:
                continue
            frac = load / tot
            seg = ph * frac
            wy0, wy1 = path_pos[pw]
            _band(x_pat + bar_w / 2, cur - seg, cur,
                  x_path - bar_w / 2, wy0, wy1, pat_colors[p])
            cur -= seg

    ax.set_xlim(-0.4, x_path + 0.4)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([x_cell, x_pat, x_path])
    ax.set_xticklabels(["Cell groups", "Patterns", "Pathways"], fontsize=10)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is None:
        if pattern == "outgoing":
            title = "Outgoing communication patterns of secreting cells"
        else:
            title = "Incoming communication patterns of target cells"
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. netAnalysis_dot
# ---------------------------------------------------------------------------
def plot_communication_patterns_dot(
    cc,
    pattern: str = "outgoing",
    *,
    result: Optional[dict] = None,
    axis: str = "cell",
    cutoff: Optional[float] = None,
    figsize: Optional[tuple[float, float]] = None,
    cmap: str = "viridis",
    title: Optional[str] = None,
) -> plt.Figure:
    """Communication-pattern loading dot plot (``netAnalysis_dot``).

    A dot plot of the NMF loadings: for ``axis="cell"`` the y-axis is cell
    groups and each column is a pattern with dot size/color ∝ ``W`` loading; for
    ``axis="signaling"`` the y-axis is pathways with dot size/color ∝ ``H``
    loading. Loadings below ``cutoff`` are hidden.

    Parameters
    ----------
    cc
        A :class:`~sceleto.interaction.CellComm`. Uses ``cc.patterns[pattern]``
        unless a ``result`` dict is passed.
    pattern
        ``"outgoing"`` or ``"incoming"``.
    result
        A result dict from
        :func:`~sceleto.interaction.identify_communication_patterns`. If
        ``None``, read ``cc.patterns[pattern]``.
    axis
        ``"cell"`` (plot ``W``: cell groups × patterns) or ``"signaling"``
        (plot ``H``: pathways × patterns).
    cutoff
        Hide loadings below this. Defaults to ``1 / n_patterns`` (as in R).
    figsize
        Figure size; derived from the matrix shape if ``None``.
    cmap
        Colormap for the loading value.
    title
        Axes title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if result is None:
        patterns = getattr(cc, "patterns", None)
        if not patterns or pattern not in patterns:
            return _empty_axes(
                f"no {pattern!r} pattern result; run "
                "identify_communication_patterns(...) first",
                figsize=figsize, title=title,
            )
        result = patterns[pattern]

    if axis not in ("cell", "signaling"):
        raise ValueError("axis must be 'cell' or 'signaling'")

    if axis == "cell":
        mat = result["W"]                 # cell groups × patterns
        ylabel = "Cell group"
    else:
        mat = result["H"].T               # pathways × patterns
        ylabel = "Signaling pathway"

    rows = list(mat.index)
    cols = list(mat.columns)              # patterns
    if len(rows) == 0 or len(cols) == 0:
        return _empty_axes("empty communication patterns",
                           figsize=figsize, title=title)

    n_pat = len(cols)
    if cutoff is None:
        cutoff = 1.0 / n_pat

    # long form of (row, pattern, loading), dropping sub-cutoff entries
    xs, ys, vals = [], [], []
    for xi, c in enumerate(cols):
        for yi, r in enumerate(rows):
            v = float(mat.loc[r, c])
            if v >= cutoff and v > 0:
                xs.append(xi)
                ys.append(yi)
                vals.append(v)

    if figsize is None:
        figsize = (max(4, 0.7 * n_pat + 2), max(3, 0.35 * len(rows) + 1.5))
    fig, ax = plt.subplots(figsize=figsize)

    if not vals:
        return _empty_axes("no loading above cutoff",
                           ax=ax, title=title)

    vals_arr = np.asarray(vals, dtype=float)
    vmax = vals_arr.max()
    sizes = 40 + 260 * (vals_arr / vmax)
    sc = ax.scatter(xs, ys, s=sizes, c=vals_arr, cmap=cmap,
                    edgecolors="gray", linewidths=0.3, vmin=0.0)

    ax.set_xticks(range(n_pat))
    ax.set_xticklabels([str(c) for c in cols], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([str(r) for r in rows], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, n_pat - 0.5)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.grid(True, color="0.9", lw=0.4)
    ax.set_axisbelow(True)
    fig.colorbar(sc, ax=ax, label="Contribution")
    if title is None:
        if pattern == "outgoing":
            title = "Outgoing communication patterns of secreting cells"
        else:
            title = "Incoming communication patterns of target cells"
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. netVisual_embeddingZoomIn
# ---------------------------------------------------------------------------
def plot_net_embedding_zoom(
    cc,
    *,
    type: str = "functional",
    cluster: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (6, 6),
    label: bool = True,
    font_size: int = 9,
    point_size: float = 120.0,
    title: Optional[str] = None,
) -> plt.Figure:
    """Zoomed pathway-similarity embedding scatter (``netVisual_embeddingZoomIn``).

    Same manifold as :func:`sceleto.interaction.plot_net_embedding`, but zoomed
    to one cluster: only the pathways assigned to ``cluster`` are shown (with the
    axes scaled to that subset), so densely packed cluster members become
    readable. Dot size is proportional to each pathway's total communication
    probability.

    Parameters
    ----------
    cc
        A :class:`~sceleto.interaction.CellComm` with
        ``cc.netP['embedding'][type]`` (from
        :func:`~sceleto.interaction.net_embedding`) and
        ``cc.netP['clustering'][type]`` (from
        :func:`~sceleto.interaction.net_clustering`).
    type
        ``"functional"`` or ``"structural"``.
    cluster
        Cluster label (1-based, as returned by ``net_clustering``) to zoom into.
        If ``None`` all clusters are shown (equivalent to the full embedding but
        with the zoom styling).
    ax
        Existing axes; a new figure is created if ``None``.
    figsize
        Figure size when creating a new figure.
    label
        Annotate each point with its pathway name.
    font_size
        Label font size.
    point_size
        Base scatter marker size (scaled by communication probability).
    title
        Axes title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if type not in ("functional", "structural"):
        raise ValueError("type must be 'functional' or 'structural'")

    netP = getattr(cc, "netP", {}) or {}
    emb = netP.get("embedding", {}).get(type)
    if emb is None:
        raise RuntimeError(
            f"cc.netP['embedding']['{type}'] missing; run "
            f"net_embedding(cc, type='{type}') first."
        )
    clusters = netP.get("clustering", {}).get(type)

    names = list(emb.index)
    Y = emb.to_numpy()

    # per-pathway communication probability for dot size
    prob_sum = {}
    if "prob" in netP and "pathways" in netP:
        prob = np.asarray(netP["prob"], dtype=float)
        for i, pw in enumerate(netP["pathways"]):
            prob_sum[str(pw)] = float(prob[:, :, i].sum())
    psum = np.array([prob_sum.get(str(n), 1.0) for n in names], dtype=float)
    pmax = psum.max() if psum.max() > 0 else 1.0

    if cluster is not None:
        if clusters is None:
            raise RuntimeError(
                f"cc.netP['clustering']['{type}'] missing; run "
                f"net_clustering(cc, type='{type}') first to zoom into a cluster."
            )
        cl = clusters.reindex(names)
        mask = (cl == cluster).to_numpy()
        if not mask.any():
            return _empty_axes(
                f"cluster {cluster} has no pathways",
                figsize=figsize, ax=ax, title=title,
            )
    else:
        mask = np.ones(len(names), dtype=bool)

    idx = np.where(mask)[0]
    sub_names = [names[i] for i in idx]
    sub_Y = Y[idx]
    sub_p = psum[idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # color: single cluster color if zoomed, else per-cluster palette
    if cluster is not None:
        colors = _palette(max(1, int(np.nanmax(clusters.to_numpy()))
                              if clusters is not None else 1))
        col = colors[(int(cluster) - 1) % len(colors)]
        sizes = point_size * (0.3 + 0.7 * sub_p / pmax)
        ax.scatter(sub_Y[:, 0], sub_Y[:, 1], s=sizes, color=col,
                   edgecolors="white", linewidths=0.6, alpha=0.75, zorder=3)
        for (x, y), name in zip(sub_Y, sub_names):
            if label:
                ax.annotate(str(name), (x, y), fontsize=font_size,
                            ha="center", va="bottom", color=col,
                            xytext=(0, 5), textcoords="offset points", zorder=4)
    else:
        if clusters is not None:
            cl = clusters.reindex(sub_names).to_numpy()
            uniq = sorted(pd.unique(cl[~pd.isna(cl)]))
        else:
            cl = np.zeros(len(sub_names))
            uniq = [0]
        colors = _palette(max(1, len(uniq)))
        cmap_for = {c: colors[i % len(colors)] for i, c in enumerate(uniq)}
        for c in uniq:
            m = cl == c
            sizes = point_size * (0.3 + 0.7 * sub_p[m] / pmax)
            ax.scatter(sub_Y[m, 0], sub_Y[m, 1], s=sizes,
                       color=cmap_for[c], edgecolors="white", linewidths=0.6,
                       alpha=0.75, zorder=3,
                       label=f"cluster {int(c)}" if not pd.isna(c) else "n/a")
        if label:
            for (x, y), name in zip(sub_Y, sub_names):
                ax.annotate(str(name), (x, y), fontsize=font_size,
                            ha="center", va="bottom", xytext=(0, 5),
                            textcoords="offset points", zorder=4)
        if len(uniq) > 1:
            ax.legend(fontsize=font_size, frameon=False, loc="best")

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.margins(0.15)
    if title is None:
        title = (f"Group {cluster}" if cluster is not None
                 else f"{type} pathway similarity")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. netVisual_aggregate
# ---------------------------------------------------------------------------
def plot_aggregate(
    cc,
    signaling: str,
    *,
    layout: str = "circle",
    thresh: float = 0.05,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (7, 7),
    title: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """Draw one signaling pathway's aggregated network (``netVisual_aggregate``).

    Convenience dispatcher: given a signaling pathway *name*, pull its ``K × K``
    inter-group matrix from ``cc.netP['prob'][:, :, idx]`` and render it with the
    existing :func:`sceleto.interaction.plot_network_circle` (``layout="circle"``)
    or :func:`sceleto.interaction.plot_chord` (``layout="chord"``).

    Parameters
    ----------
    cc
        A :class:`~sceleto.interaction.CellComm` after
        :meth:`~sceleto.interaction.CellComm.compute_communication_pathway`.
    signaling
        Pathway name (from ``cc.netP['pathways']``).
    layout
        ``"circle"`` or ``"chord"``.
    thresh
        Unused for the pathway-level matrix (already thresholded during
        ``compute_communication_pathway``); accepted for API parity.
    ax
        Existing axes; a new figure is created if ``None``.
    figsize
        Figure size when creating a new figure.
    title
        Plot title (defaults to ``"<signaling> signaling pathway network"``).
    **kwargs
        Forwarded to the underlying plotting function.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if layout not in ("circle", "chord"):
        raise ValueError("layout must be 'circle' or 'chord'")

    netP = getattr(cc, "netP", {}) or {}
    if "prob" not in netP:
        raise RuntimeError(
            "cc.netP['prob'] missing; run compute_communication_pathway() first."
        )
    p = _pathway_index(cc, signaling)
    mat = np.asarray(netP["prob"], dtype=float)[:, :, p]
    names = list(getattr(cc, "group_names", []))

    if title is None:
        title = f"{signaling} signaling pathway network"

    if len(names) == 0 or mat.sum() == 0:
        return _empty_axes("no significant communication",
                           figsize=figsize, ax=ax, title=title)

    if layout == "circle":
        return plot_network_circle(mat, names, ax=ax, figsize=figsize,
                                   title=title, **kwargs)
    return plot_chord(mat, names, ax=ax, figsize=figsize, title=title, **kwargs)
