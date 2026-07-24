"""Network analysis on the aggregated communication network.

Ports CellChat's ``netAnalysis_computeCentrality`` (analysis.R) onto
``networkx``. For a signaling network (a K×K weighted directed matrix) it
computes the centrality scores CellChat uses to assign signaling *roles*:

===============  ============================  ==========================
CellChat metric  networkx equivalent           biological role
===============  ============================  ==========================
out-degree       weighted out-degree           **Sender**
in-degree        weighted in-degree            **Receiver**
flow betweenness betweenness centrality         **Mediator**
information      information centrality         **Influencer**
hub/authority    HITS                          (hub = sender-like)
eigenvector      eigenvector centrality
pagerank         pagerank
===============  ============================  ==========================

``flowbet``/``infocent`` come from R's ``sna`` package, which has no direct
Python equivalent; we substitute the closest standard networkx centralities
(betweenness and information/current-flow closeness), which capture the same
mediator/influencer notion.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import networkx as nx
import pandas as pd


def _digraph(mat: np.ndarray, names: list[str]) -> nx.DiGraph:
    """Build a weighted DiGraph from a K×K matrix (row=source, col=target)."""
    G = nx.DiGraph()
    G.add_nodes_from(names)
    K = len(names)
    for i in range(K):
        for j in range(K):
            w = mat[i, j]
            if w > 0:
                G.add_edge(names[i], names[j], weight=float(w))
    return G


def compute_centrality(
    mat: np.ndarray,
    names: list[str],
) -> pd.DataFrame:
    """Per-node centrality scores for one signaling network.

    Parameters
    ----------
    mat
        ``K × K`` weighted communication matrix (source × target).
    names
        Node (cell-group) names.

    Returns
    -------
    pd.DataFrame
        Index = ``names``; columns = ``outdeg`` (Sender), ``indeg`` (Receiver),
        ``flowbet`` (Mediator), ``info`` (Influencer), plus ``hub``,
        ``authority``, ``eigen``, ``pagerank``.
    """
    G = _digraph(mat, names)
    K = len(names)

    outdeg = mat.sum(axis=1)
    indeg = mat.sum(axis=0)

    # betweenness (Mediator) — R inverts edge weights (E(G)$weight <- 1/weight)
    # then igraph::betweenness with normalized = FALSE (raw counts).
    if G.number_of_edges():
        inv = {(u, v): 1.0 / d["weight"] for u, v, d in G.edges(data=True)}
        nx.set_edge_attributes(G, inv, "distance")
        betw = nx.betweenness_centrality(G, weight="distance", normalized=False)
    else:
        betw = {n: 0.0 for n in names}

    # information / current-flow closeness (Influencer) on the undirected view.
    # to_undirected() would drop one of a reciprocal i<->j pair (last-writer
    # wins); sum the two directions so no weight is silently lost. Compute
    # per connected component so a disconnected graph (an isolated cell group)
    # does not collapse every node's score to zero.
    UG = nx.Graph()
    UG.add_nodes_from(names)
    for u, v, d in G.edges(data=True):
        w = d["weight"]
        if UG.has_edge(u, v):
            UG[u][v]["weight"] += w
        else:
            UG.add_edge(u, v, weight=w)
    # Compute current-flow closeness per component, then normalize WITHIN each
    # component (divide by the component max) before combining. Absolute
    # current-flow values scale with a component's edge-weight magnitude, so
    # without per-component rescaling a peripheral high-weight pair would
    # dominate the global normalization and be mislabeled the top Influencer.
    info = {n: 0.0 for n in names}
    for comp in nx.connected_components(UG):
        if len(comp) < 2:
            continue
        sub = UG.subgraph(comp)
        try:
            part = nx.current_flow_closeness_centrality(sub, weight="weight")
        except Exception:
            continue
        pmax = max(part.values()) if part else 0.0
        if pmax > 0:
            for node, val in part.items():
                info[node] = val / pmax

    # HITS (hub = sender-like, authority = receiver-like)
    try:
        hub, auth = nx.hits(G, max_iter=1000, normalized=True)
    except Exception:
        hub = {n: 0.0 for n in names}
        auth = {n: 0.0 for n in names}

    # eigenvector centrality
    try:
        eig = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eig = {n: 0.0 for n in names}

    # pagerank
    try:
        pr = nx.pagerank(G, weight="weight")
    except Exception:
        pr = {n: 1.0 / K for n in names}

    def _vec(d: dict) -> np.ndarray:
        return np.array([d[n] for n in names], float)

    def _scale1(d: dict) -> np.ndarray:
        # igraph hub/authority/eigen scale their result so the max is 1
        v = _vec(d)
        m = v.max()
        return v / m if m > 0 else v

    # Match R's netAnalysis_computeCentrality (igraph) normalization exactly:
    #   outdeg/indeg = igraph::strength  → RAW weighted degree (no scaling)
    #   betweenness  = igraph::betweenness on inverse weights → RAW (not norm'd)
    #   hub/authority/eigen = igraph scores → scaled so max = 1
    #   page_rank    = igraph::page_rank  → sums to 1
    return pd.DataFrame(
        {
            "outdeg": outdeg,                 # Sender (raw strength)
            "indeg": indeg,                   # Receiver (raw strength)
            "flowbet": _vec(betw),            # Mediator (raw betweenness)
            "info": _vec(info),               # Influencer
            "hub": _scale1(hub),
            "authority": _scale1(auth),
            "eigen": _scale1(eig),
            "pagerank": _vec(pr),
        },
        index=names,
    )


def signaling_role(
    centrality: pd.DataFrame,
) -> pd.DataFrame:
    """Assign the dominant signaling role per cell group.

    For each node, reports the four core scores (Sender/Receiver/Mediator/
    Influencer) and the ``role`` with the highest score.
    """
    core = centrality[["outdeg", "indeg", "flowbet", "info"]].rename(
        columns={
            "outdeg": "Sender",
            "indeg": "Receiver",
            "flowbet": "Mediator",
            "info": "Influencer",
        }
    )
    out = core.copy()
    out["role"] = core.idxmax(axis=1)
    return out


def network_graph(
    mat: np.ndarray,
    names: list[str],
) -> nx.DiGraph:
    """Public helper: aggregated communication matrix → weighted DiGraph.

    Node/edge order matches ``names``; edge attribute ``weight`` carries the
    communication strength. Convenient for custom downstream graph analysis.
    """
    return _digraph(mat, names)


_ROLE_MEASURE = {
    "outgoing": "outdeg",   # sender strength
    "incoming": "indeg",    # receiver strength
    "sender": "outdeg",
    "receiver": "indeg",
    "mediator": "flowbet",
    "influencer": "info",
}


def signaling_role_pathway(cc, *, pattern: str = "outgoing") -> pd.DataFrame:
    """Per-pathway × cell-group signaling-role matrix (``netAnalysis_signalingRole_heatmap``).

    For each pathway network, computes the chosen centrality role per cell group
    and stacks them into a ``pathway × cell-group`` matrix — the data behind the
    standard CellChat signaling-role heatmap.

    Parameters
    ----------
    cc
        A :class:`~sceleto.interaction.CellComm` after
        :meth:`~sceleto.interaction.CellComm.compute_communication_pathway`.
    pattern
        Which role to score: ``"outgoing"``/``"sender"`` (out-degree),
        ``"incoming"``/``"receiver"`` (in-degree), ``"mediator"`` (betweenness),
        or ``"influencer"`` (information centrality).

    Returns
    -------
    pd.DataFrame
        Rows = pathways, columns = cell groups, values = the role score.
    """
    if pattern not in _ROLE_MEASURE:
        raise ValueError(
            f"pattern {pattern!r} invalid; choose from {sorted(_ROLE_MEASURE)}"
        )
    if not getattr(cc, "netP", None) or "prob" not in cc.netP:
        raise RuntimeError("Call compute_communication_pathway() on the CellComm first.")
    col = _ROLE_MEASURE[pattern]
    names = cc.group_names
    prob = cc.netP["prob"]
    pathways = cc.netP["pathways"]
    rows = []
    for p, pw in enumerate(pathways):
        cen = compute_centrality(prob[:, :, p], names)
        rows.append(cen[col].to_numpy())
    return pd.DataFrame(rows, index=list(pathways), columns=names)
