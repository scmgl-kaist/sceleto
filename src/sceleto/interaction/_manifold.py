"""Manifold / similarity-based clustering of signaling pathways.

Python port of CellChat's pathway-manifold analysis (``analysis.R``):
``computeNetSimilarity`` → ``netEmbedding`` → ``netClustering``. It groups
signaling pathways that behave similarly across the cell-group network by

1. :func:`compute_net_similarity` — a pathway × pathway **similarity matrix**,
   either *functional* (Jaccard overlap of the binarized K×K networks) or
   *structural* (a graph D-measure combining node-dispersion and
   alpha-centrality Jensen–Shannon divergences), then smoothed with a
   shared-nearest-neighbor (SNN) graph;
2. :func:`net_embedding` — a 2-D UMAP embedding of that similarity matrix;
3. :func:`net_clustering` — k-means (with automatic *k* via the graph-Laplacian
   eigengap) or spectral clustering of the embedded pathways;
4. :func:`plot_net_embedding` — a scatter of the embedding colored by cluster.

All functions take the :class:`~sceleto.interaction._communication.CellComm`
object first and read ``cc.netP['prob']`` (K×K×nPathway) and
``cc.netP['pathways']`` produced by ``compute_communication_pathway``. Results
are written back onto ``cc.netP`` under ``'similarity'``, ``'embedding'`` and
``'clustering'`` and also returned.

Package policy
--------------
Standard building blocks are used wherever a standard definition exists:

- functional similarity via :func:`scipy.spatial.distance.pdist` (``jaccard``);
- the kNN graph via :func:`sklearn.neighbors.kneighbors_graph`, with the SNN
  Jaccard step a few lines of :mod:`scipy.sparse` on top of it;
- UMAP via :mod:`umap` (``umap-learn``);
- Laplacian eigengap via :func:`scipy.sparse.linalg.eigsh`;
- k-means / spectral via :mod:`sklearn.cluster`;
- structural shortest paths via :mod:`networkx`.

The only genuinely CellChat-specific custom code is the structural D-measure
(:func:`_net_d_structure` and its ``nnd`` / ``alpha_centrality`` helpers), ported
directly from the R/igraph source. Every R-vs-Python approximation is noted in
the relevant docstring / comment.

Notes
-----
Approximations relative to CellChat's R implementation:

* **SNN.** CellChat calls ``FNN::get.knn`` and a C++ ``ComputeSNN``. We replace
  ``get.knn`` with :func:`sklearn.neighbors.kneighbors_graph` on the similarity
  *rows* (Euclidean) and reproduce ``ComputeSNN`` exactly as sparse ops: with a
  connectivity kNN of ``k`` columns *including self* (matching R's
  ``nn.ranked = cbind(self, k-1 neighbors)``), ``S = knn @ knn.T`` counts shared
  neighbors and the Jaccard transform is ``S / (2k − S)`` pruned below
  ``prune_snn`` — identical to the C++ ``value/(k + (k − value))``. Ties in the
  kNN can differ between FNN and sklearn, so the smoothed matrix may differ
  slightly in borderline cases; the functional-similarity and clustering
  structure are unaffected in practice.

* **Structural similarity.** ``computeNetD_structure`` is ported faithfully,
  including the node-distance histogram (``node_distance``/``nnd``) and the
  igraph ``alpha.centrality`` term (Katz-style ``(I − αAᵀ)⁻¹ exo``). Two minor
  differences from igraph: (i) the graph complement used in the alpha-centrality
  term is built without self-loops (igraph's ``graph.complementer`` likewise
  excludes them); (ii) singular ``(I − αAᵀ)`` systems are solved in the
  least-squares sense. These affect only the third (weight ``w3 = 0.1``) term.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform


# --------------------------------------------------------------------------- #
# small utilities                                                             #
# --------------------------------------------------------------------------- #
def _get_prob_and_pathways(cc) -> tuple[np.ndarray, list[str]]:
    """Pull the ``(K, K, nPathway)`` prob array and pathway names off *cc*.

    Raises a clear error if :meth:`CellComm.compute_communication_pathway` has
    not been run yet.
    """
    netP = getattr(cc, "netP", None)
    if not netP or "prob" not in netP:
        raise RuntimeError(
            "cc.netP['prob'] is missing; call "
            "compute_communication_pathway() before pathway-manifold analysis."
        )
    prob = np.asarray(netP["prob"], dtype=float)
    if prob.ndim != 3:
        raise ValueError(
            f"cc.netP['prob'] must be a 3-D (K, K, nPathway) array, got shape {prob.shape}"
        )
    pathways = [str(p) for p in netP.get("pathways", [])]
    if len(pathways) != prob.shape[2]:
        # fall back to positional names if the pathways list is missing/wrong
        pathways = [f"pathway{i + 1}" for i in range(prob.shape[2])]
    return prob, pathways


def _default_k(n: int) -> int:
    """CellChat's default kNN size: ``ceil(sqrt(n))`` (+1 when ``n > 25``)."""
    k = int(np.ceil(np.sqrt(n)))
    if n > 25:
        k += 1
    return k


# --------------------------------------------------------------------------- #
# structural D-measure (port of computeNetD_structure and helpers)            #
# --------------------------------------------------------------------------- #
def _entropia(a: np.ndarray) -> float:
    """Shannon entropy ``-sum(a*log(a))`` over strictly positive entries.

    Port of R ``entropia``. Zero/negative entries are dropped (``0*log 0 = 0``).
    """
    a = np.asarray(a, dtype=float).ravel()
    a = a[a > 0]
    if a.size == 0:
        return 0.0
    return float(-np.sum(a * np.log(a)))


def _node_distance(adj: np.ndarray) -> np.ndarray:
    """Column-normalized node-distance distribution matrix (R ``node_distance``).

    For an ``n``-node directed graph, entry ``[d-1, v]`` is the number of nodes
    at shortest-path distance ``d`` from node ``v`` (unreachable → distance
    ``n``), divided by ``n − 1`` so each column is a distance distribution.

    Uses :func:`networkx.all_pairs_shortest_path_length` (unweighted BFS), the
    Python analogue of igraph ``shortest.paths(algorithm="unweighted")``.
    """
    import networkx as nx

    n = adj.shape[0]
    if n == 1:
        return np.array([[1.0]])

    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    # full n×n shortest-path length matrix; unreachable pairs → n (matches R).
    m = np.full((n, n), float(n))
    for src, lengths in nx.all_pairs_shortest_path_length(G):
        for dst, d in lengths.items():
            m[src, dst] = d

    # For each target column v, histogram the distances-from-v into bins
    # 1..n (bin d counts nodes exactly d hops away). R excludes distance 0
    # (the node itself) implicitly because the histogram breaks are (0:n] and
    # the self-distance 0 lands in no positive bin.
    a = np.zeros((n, n))
    for v in range(n):
        col = m[:, v]  # distances from every node to... (see note below)
        # R indexes m as distance from node j (column) via which(m==quem)/n:
        # column-major flatten means the second index is the "from" node, i.e.
        # distances FROM node v. m is symmetric only for undirected graphs; for
        # CellChat's binarized directed nets we follow R's column convention.
        counts, _ = np.histogram(col, bins=np.arange(0, n + 1))
        a[:, v] = counts
    return a / (n - 1)


def _nnd(adj: np.ndarray) -> np.ndarray:
    """Network Node Dispersion vector (R ``nnd``): pdf plus a dispersion scalar.

    Returns ``[pdf_1 .. pdf_n, NND]`` where ``pdf`` is the mean node-distance
    distribution and ``NND`` is the (normalized) heterogeneity of the per-node
    distributions.
    """
    n = adj.shape[0]
    nd = _node_distance(adj)
    pdfm = nd.mean(axis=1)  # colMeans in R (columns = nodes)
    # normalization: log(max(2, #positive pdf entries over 1..n-1 + 1))
    n_pos = int(np.sum(pdfm[: n - 1] > 0)) if n > 1 else 0
    norm = np.log(max(2, n_pos + 1))
    nnd_val = max(0.0, _entropia(pdfm) - _entropia(nd) / n) / norm
    return np.concatenate([pdfm, [nnd_val]])


def _alpha_centrality(adj: np.ndarray) -> np.ndarray:
    """Sorted, scaled alpha-centrality with a residual (R ``alpha_centrality``).

    Ports ``sort(alpha.centrality(g, exo=degree(g)/(N-1), alpha=1/N)) / N^2``
    followed by an appended ``max(0, 1 - sum)`` residual.

    igraph's ``alpha.centrality`` solves ``x = α Aᵀ x + exo`` →
    ``x = (I − α Aᵀ)⁻¹ exo``. Here ``A`` is the (binary, directed) adjacency,
    ``exo`` is total degree / (N−1), and ``α = 1/N``. Singular systems are
    solved in the least-squares sense (see module Notes).
    """
    n = adj.shape[0]
    if n == 0:
        return np.array([1.0])
    A = np.asarray(adj, dtype=float)
    # igraph degree() = total degree (in + out) for a directed graph.
    deg = A.sum(axis=0) + A.sum(axis=1)
    exo = deg / (n - 1) if n > 1 else deg
    alpha = 1.0 / n
    M = np.eye(n) - alpha * A.T
    try:
        r = np.linalg.solve(M, exo)
    except np.linalg.LinAlgError:
        r = np.linalg.lstsq(M, exo, rcond=None)[0]
    r = np.sort(r) / (n ** 2)
    resid = max(0.0, 1.0 - float(np.sum(r)))
    return np.concatenate([r, [resid]])


def _graph_complement(adj: np.ndarray) -> np.ndarray:
    """Binary directed-graph complement, no self-loops (igraph ``graph.complementer``)."""
    n = adj.shape[0]
    comp = 1.0 - (np.asarray(adj) > 0).astype(float)
    np.fill_diagonal(comp, 0.0)
    return comp


def _net_d_structure(
    gi: np.ndarray,
    gj: np.ndarray,
    w1: float = 0.45,
    w2: float = 0.45,
    w3: float = 0.1,
) -> float:
    """Structural distance between two binarized networks (R ``computeNetD_structure``).

    The D-measure combines three Jensen–Shannon-style terms:

    * ``first`` — JSD between the two mean node-distance distributions;
    * ``second`` — ``|sqrt(NND_i) − sqrt(NND_j)|`` (node-dispersion gap);
    * ``third`` — JSD of the alpha-centrality distributions of the graphs and of
      their complements (averaged).

    Returns ``w1*first + w2*second + w3*third``. Two equal-node graphs are
    assumed (CellChat compares pathways over the same K cell groups), so the
    padding logic in R (``max(M, N)``) collapses to ``N == M``.
    """
    first = second = third = 0.0
    n = gi.shape[0]
    m = gj.shape[0]

    if w1 + w2 > 0:
        pg = _nnd(gi)  # length n+1 (pdf[0..n-1], NND at n)
        ph = _nnd(gj)  # length m+1
        size = max(m, n)
        PM = np.zeros(size)
        # R: PM[1:(N-1)] = pg[1:(N-1)]; PM[last] = pg[N]  (1-based)
        PM[: n - 1] += pg[: n - 1]
        PM[-1] += pg[n - 1]
        PM[: m - 1] += ph[: m - 1]
        PM[-1] += ph[m - 1]
        PM = PM / 2.0
        first = np.sqrt(
            max(
                (_entropia(PM) - (_entropia(pg[:n]) + _entropia(ph[:m])) / 2.0)
                / np.log(2),
                0.0,
            )
        )
        second = abs(np.sqrt(pg[n]) - np.sqrt(ph[m]))

    if w3 > 0:
        def _js_alpha(a: np.ndarray, gj_: np.ndarray) -> float:
            pg_ = _alpha_centrality(a)
            ph_ = _alpha_centrality(gj_)
            mm = max(len(pg_), len(ph_))
            Pg = np.zeros(mm)
            Ph = np.zeros(mm)
            Pg[mm - len(pg_):] = pg_
            Ph[mm - len(ph_):] = ph_
            val = (_entropia((Pg + Ph) / 2.0) - (_entropia(pg_) + _entropia(ph_)) / 2.0) / np.log(2)
            return np.sqrt(max(val, 0.0)) / 2.0

        third = _js_alpha(gi, gj)
        third += _js_alpha(_graph_complement(gi), _graph_complement(gj))

    return float(w1 * first + w2 * second + w3 * third)


# --------------------------------------------------------------------------- #
# SNN smoothing (port of buildSNN + C++ ComputeSNN)                           #
# --------------------------------------------------------------------------- #
def _build_snn(sim: np.ndarray, k: int, prune_snn: float = 1.0 / 15.0) -> np.ndarray:
    """Shared-nearest-neighbor smoothing of a similarity matrix (R ``buildSNN``).

    Reproduces CellChat's ``buildSNN`` + C++ ``ComputeSNN``. The kNN graph is
    built with :func:`sklearn.neighbors.kneighbors_graph` (Euclidean on the
    similarity *rows*) rather than ``FNN::get.knn`` — see module Notes on the tie
    approximation. The SNN transform is then the standard sparse computation:

    ``S = knn @ knn.T`` counts shared neighbors; ``S / (2k − S)`` is the Jaccard
    overlap (identical to the C++ ``value/(k + (k − value))``); values below
    ``prune_snn`` are dropped; the diagonal is set to 1.

    Parameters
    ----------
    sim
        ``n × n`` similarity matrix (used as ``n`` feature vectors).
    k
        Neighborhood size. Must satisfy ``k < n``.
    prune_snn
        Jaccard cutoff below which SNN edges are removed (R default ``1/15``).

    Returns
    -------
    numpy.ndarray
        Dense ``n × n`` SNN weight matrix (diagonal 1).
    """
    from sklearn.neighbors import kneighbors_graph

    n = sim.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be smaller than the number of pathways n={n}")
    if k < 2:
        # too few pathways for a meaningful shared-neighbor graph — return the
        # identity (no smoothing). The lower bound is enforced at embedding time.
        return np.eye(n)

    # Connectivity kNN with k ones per row INCLUDING self, matching R's
    # nn.ranked = cbind(1:n, nn.index[, 1:(k-1)]) (= k columns). sklearn counts
    # self WITHIN n_neighbors when include_self=True, so n_neighbors=k yields
    # exactly k ones per row (self + k-1 neighbors), matching R.
    knn = kneighbors_graph(
        sim, n_neighbors=k, mode="connectivity", include_self=True
    ).astype(float)

    snn = (knn @ knn.T).tocoo()
    # Jaccard transform: x / (k + (k - x)) = x / (2k - x)
    data = snn.data / (2.0 * k - snn.data)
    data[data < prune_snn] = 0.0
    snn = sp.coo_matrix((data, (snn.row, snn.col)), shape=snn.shape).tocsr()
    snn.eliminate_zeros()

    w = snn.toarray()
    np.fill_diagonal(w, 1.0)
    return w


# --------------------------------------------------------------------------- #
# 1. similarity                                                               #
# --------------------------------------------------------------------------- #
def compute_net_similarity(
    cc,
    *,
    slot: str = "netP",
    type: str = "functional",
    k: Optional[int] = None,
    thresh: Optional[float] = None,
) -> pd.DataFrame:
    """Pairwise similarity between per-pathway signaling networks.

    Port of CellChat's ``computeNetSimilarity`` (``analysis.R``). Builds an
    ``nPathway × nPathway`` similarity matrix over the K×K per-pathway networks
    in ``cc.netP['prob']``, then smooths it with a shared-nearest-neighbor graph.

    Parameters
    ----------
    cc
        :class:`CellComm` object with ``cc.netP['prob']`` (``K×K×nPathway``) and
        ``cc.netP['pathways']`` set by ``compute_communication_pathway``.
    slot
        Slot name to read the prob array from (default ``"netP"``); kept for API
        parity with R. Only ``"netP"`` is supported.
    type
        ``"functional"`` (Jaccard overlap of binarized networks) or
        ``"structural"`` (``1 − D``-measure; see :func:`_net_d_structure`).
    k
        kNN neighborhood for SNN smoothing. Defaults to
        ``ceil(sqrt(nPathway))`` (+1 if ``nPathway > 25``), as in R.
    thresh
        Optional fraction in ``[0, 0.25]``: interactions below this quantile of
        the non-zero prob values are zeroed before similarity is computed
        (R's ``thresh``).

    Returns
    -------
    pandas.DataFrame
        The smoothed ``nPathway × nPathway`` similarity matrix, indexed and
        columned by pathway name. Also stored at
        ``cc.netP['similarity'][type]``.

    Notes
    -----
    **Functional similarity** is exactly R's
    ``sum(Gi*Gj) / sum(Gi + Gj − Gi*Gj)`` on the binarized (``prob > 0``)
    networks — i.e. the Jaccard index of the flattened K×K binary adjacency
    vectors. It is computed here with
    :func:`scipy.spatial.distance.pdist` (``metric="jaccard"``) and
    ``similarity = 1 − distance``; the full K×K matrix including the diagonal is
    flattened, matching R (R sums over all K² entries, diagonal included). The
    diagonal of the result is forced to 1 as in R.

    **Structural similarity** is ``1 − D`` where ``D`` is the graph D-measure;
    it is best-effort faithful to R/igraph (see module Notes).
    """
    if type not in ("functional", "structural"):
        raise ValueError("type must be 'functional' or 'structural'")
    if slot != "netP":
        raise ValueError("only slot='netP' is supported in this port")

    prob, pathways = _get_prob_and_pathways(cc)
    n_path = prob.shape[2]
    if n_path < 2:
        raise ValueError(
            f"need at least 2 signaling pathways to compute similarity, got {n_path}"
        )

    prob = prob.copy()
    if thresh is not None:
        nz = prob[prob != 0]
        if nz.size:
            cutoff = np.quantile(nz, thresh)
            prob[prob < cutoff] = 0.0

    if k is None:
        k = _default_k(n_path)
    # SNN needs k < n_path; clamp so tiny pathway sets still produce a
    # similarity matrix (the meaningful lower bound is caught at embedding).
    k = min(k, n_path - 1)

    if type == "functional":
        # binarize each K×K network and flatten to a row per pathway;
        # Jaccard similarity = 1 - jaccard distance (matches R's set formula).
        K = prob.shape[0]
        binmat = (prob > 0).reshape(K * K, n_path).T.astype(bool)  # nPathway × K²
        # pdist(jaccard) is undefined for all-zero rows (0/0); guard by keeping
        # track and setting those similarities to 0 (R: NA -> 0).
        with np.errstate(invalid="ignore", divide="ignore"):
            dist = squareform(pdist(binmat, metric="jaccard"))
        S = 1.0 - dist
        S[np.isnan(S)] = 0.0
        np.fill_diagonal(S, 1.0)
    else:  # structural
        S = np.zeros((n_path, n_path))
        bins = [(prob[:, :, i] > 0).astype(float) for i in range(n_path)]
        D = np.zeros((n_path, n_path))
        for i in range(n_path - 1):
            for j in range(i + 1, n_path):
                D[i, j] = _net_d_structure(bins[i], bins[j])
        D[~np.isfinite(D)] = 0.0
        D = D + D.T
        S = 1.0 - D
        # R keeps the raw (1 - D) diagonal (D_ii = 0 -> 1). Force it for safety.
        np.fill_diagonal(S, 1.0)

    if cc.verbose:
        print(
            f"[interaction] computing {type} pathway similarity "
            f"({n_path} pathways, SNN k={k})"
        )

    snn = _build_snn(S, k=k, prune_snn=1.0 / 15.0)
    similarity = S * snn

    df = pd.DataFrame(similarity, index=pathways, columns=pathways)

    sim_slot = cc.netP.setdefault("similarity", {})
    sim_slot[type] = df
    return df


# --------------------------------------------------------------------------- #
# 2. embedding                                                                #
# --------------------------------------------------------------------------- #
def net_embedding(
    cc,
    *,
    type: str = "functional",
    n_neighbors: Optional[int] = None,
    min_dist: float = 0.3,
    metric: str = "correlation",
    seed: int = 42,
    pathway_remove: Optional[list[str]] = None,
    **umap_kwargs,
) -> pd.DataFrame:
    """UMAP embedding of the pathway similarity matrix (R ``netEmbedding``).

    Runs 2-D UMAP on the similarity matrix (each pathway's similarity row is its
    feature vector), reproducing CellChat's ``runUMAP`` defaults
    (``metric="correlation"``, ``min_dist=0.3``,
    ``n_neighbors=ceil(sqrt(n))+1``, ``seed=42``).

    Parameters
    ----------
    cc
        :class:`CellComm` object; requires ``cc.netP['similarity'][type]`` from
        :func:`compute_net_similarity`.
    type
        ``"functional"`` or ``"structural"``.
    n_neighbors
        UMAP neighborhood size. Defaults to ``ceil(sqrt(nPathway)) + 1`` (R).
        Must be smaller than the number of pathways.
    min_dist
        UMAP ``min_dist`` (default 0.3, as in R).
    metric
        UMAP metric (default ``"correlation"``, matching R's ``runUMAP``).
    seed
        Random seed (R uses 42).
    pathway_remove
        Pathway names to drop before embedding. If ``None`` (default), R's rule
        is applied: pathways whose similarity column sums to exactly 1 (i.e. only
        self-similar, disconnected from every other pathway) are removed.
    **umap_kwargs
        Extra keyword arguments forwarded to :class:`umap.UMAP`.

    Returns
    -------
    pandas.DataFrame
        ``nPathway × 2`` embedding (columns ``UMAP1``, ``UMAP2``), indexed by
        pathway name. Also stored at ``cc.netP['embedding'][type]``.

    Raises
    ------
    ValueError
        If fewer than 3 pathways remain (UMAP requires ``n_neighbors < n`` with
        ``n_neighbors >= 2``, so a meaningful embedding needs ``n >= 3``).

    Notes
    -----
    UMAP requires ``n_neighbors < n_samples``. When the requested/derived
    ``n_neighbors`` is too large for the (possibly reduced) pathway set it is
    clamped to ``n − 1`` with a verbose note, rather than raising.
    """
    if type not in ("functional", "structural"):
        raise ValueError("type must be 'functional' or 'structural'")
    sim = cc.netP.get("similarity", {}).get(type)
    if sim is None:
        raise RuntimeError(
            f"cc.netP['similarity']['{type}'] missing; run "
            f"compute_net_similarity(cc, type='{type}') first."
        )
    S = sim.copy()

    # R: remove pathways whose column-sum == 1 (self-similar only).
    if pathway_remove is None:
        colsum = S.to_numpy().sum(axis=0)
        pathway_remove = list(S.index[np.isclose(colsum, 1.0)])
    if pathway_remove:
        keep = [p for p in S.index if p not in set(pathway_remove)]
        S = S.loc[keep, keep]
        if cc.verbose and pathway_remove:
            print(
                f"[interaction] dropping {len(pathway_remove)} disconnected "
                f"pathway(s) before embedding: {list(pathway_remove)}"
            )

    n = S.shape[0]
    if n < 3:
        raise ValueError(
            f"need at least 3 connected pathways to embed, got {n}; "
            "similarity is too sparse for a meaningful manifold "
            "(try lowering `thresh` or providing more pathways)"
        )

    if n_neighbors is None:
        n_neighbors = int(np.ceil(np.sqrt(n))) + 1
    if n_neighbors >= n:
        n_neighbors = n - 1
        if cc.verbose:
            print(
                f"[interaction] clamping n_neighbors to {n_neighbors} "
                f"(< n_pathways={n})"
            )
    n_neighbors = max(2, n_neighbors)

    import umap

    # UMAP's default spectral init needs n_components+1 eigenvectors of a sparse
    # graph, which ARPACK cannot extract when n <= n_components+1 (e.g. exactly 3
    # connected pathways) — it raises "k >= N". Fall back to random init in that
    # regime unless the caller supplied their own init.
    if "init" not in umap_kwargs and n <= 3:
        umap_kwargs = {**umap_kwargs, "init": "random"}

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        **umap_kwargs,
    )
    Y = reducer.fit_transform(S.to_numpy())

    df = pd.DataFrame(Y, index=S.index, columns=["UMAP1", "UMAP2"])

    emb_slot = cc.netP.setdefault("embedding", {})
    emb_slot[type] = df
    if cc.verbose:
        print(f"[interaction] embedded {n} pathways ({type}) via UMAP")
    return df


# --------------------------------------------------------------------------- #
# Laplacian / eigengap (port of computeLaplacian + computeEigengap)           #
# --------------------------------------------------------------------------- #
def _laplacian_eigs(cm: np.ndarray, tol: float = 0.01) -> tuple[np.ndarray, int]:
    """Sorted normalized-Laplacian eigenvalues and #near-zero (R ``computeLaplacian``).

    Normalized Laplacian ``L = I − D^{-1/2} CM D^{-1/2}`` with
    ``D = colSums(CM)``. Eigenvalues are the smallest via
    :func:`scipy.sparse.linalg.eigsh`.

    Returns ``(sorted_eigs, n_zeros)`` where ``n_zeros`` counts eigenvalues
    ``<= tol``.
    """
    from scipy.sparse.linalg import eigsh

    cm = np.asarray(cm, dtype=float)
    n = cm.shape[0]
    d = cm.sum(axis=0)
    dsq = np.sqrt(np.where(d > 0, d, 1.0))
    L = -(cm / dsq).T / dsq
    # R computeLaplacian: L <- -t(CM/Dsq)/Dsq; diag(L) <- 1 + diag(L). The
    # existing diag holds -CM_ii/D_i, so the final diagonal is 1 - CM_ii/D_i
    # (per-node, NOT a constant 1.0). ADD 1, do not overwrite.
    np.fill_diagonal(L, 1.0 + np.diag(L))
    L = (L + L.T) / 2.0  # enforce symmetry for eigsh

    num = min(100, n)
    if num >= n:
        vals = np.linalg.eigvalsh(L)
    else:
        try:
            vals = eigsh(L, k=num, which="SM", return_eigenvectors=False, tol=1e-4)
        except Exception:
            vals = np.linalg.eigvalsh(L)
    vals = np.sort(np.abs(np.real(vals)))
    n_zeros = int(np.sum(vals <= tol))
    return vals, n_zeros


def _consensus_eigengap_k(
    Y: np.ndarray, *, seed: int = 0, n_init: int = 10, k_range=range(2, 11)
) -> int:
    """R ``netClustering`` automatic-k: consensus co-clustering + eigengap.

    Runs k-means on the embedding ``Y`` for each ``k`` in ``k_range`` (capped at
    ``N-1``), averages the resulting co-clustering (connectivity) matrices into a
    consensus matrix ``CM``, and returns the Laplacian eigengap of ``CM`` (R's
    ``computeEigengap(CM)$upper_bound``). This is what CellChat uses — the
    eigengap of the *consensus* matrix, not of the similarity matrix.
    """
    from sklearn.cluster import KMeans

    N = Y.shape[0]
    ks = [k for k in k_range if 2 <= k <= N - 1]
    if not ks:
        return 1
    cm = np.zeros((N, N), dtype=float)
    for k in ks:
        labels = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(Y)
        # co-clustering indicator: 1 where two points share a cluster
        cm += (labels[:, None] == labels[None, :]).astype(float)
    cm /= len(ks)
    return _eigengap_k(cm)


def _eigengap_k(cm: np.ndarray, tol: float = 0.01) -> int:
    """Number of clusters from the Laplacian eigengap (R ``computeEigengap``).

    Truncates the consensus matrix at a ``tau`` chosen from the initial
    zero-eigenvalue count, symmetrizes, recomputes the Laplacian spectrum, and
    returns ``argmax(diff(eigenvalues)) + 1`` (1-based cluster count = R's
    ``upper_bound``).
    """
    cm = np.array(cm, dtype=float)
    k_init = _laplacian_eigs(cm, tol=tol)[1]
    if k_init <= 5:
        tau = 0.3
    elif k_init <= 10:
        tau = 0.4
    else:
        tau = 0.5
    cm = cm.copy()
    cm[cm <= tau] = 0.0
    cm = (cm + cm.T) / 2.0
    vals, _ = _laplacian_eigs(cm, tol=tol)
    if vals.size < 2:
        return 1
    gaps = np.diff(vals)
    # R: which(gaps == max(gaps)) is 1-based -> index+1 clusters.
    upper_bound = int(np.argmax(gaps)) + 1
    return max(1, upper_bound)


# --------------------------------------------------------------------------- #
# 3. clustering                                                               #
# --------------------------------------------------------------------------- #
def net_clustering(
    cc,
    *,
    type: str = "functional",
    k: Optional[int] = None,
    method: str = "kmeans",
    k_eigen: Optional[int] = None,
    n_init: int = 10,
    seed: int = 0,
) -> pd.Series:
    """Cluster pathways in the similarity embedding (R ``netClustering``).

    Parameters
    ----------
    cc
        :class:`CellComm` object; requires ``cc.netP['embedding'][type]`` from
        :func:`net_embedding`.
    type
        ``"functional"`` or ``"structural"``.
    k
        Number of clusters. If ``None`` and ``method="kmeans"``, ``k`` is chosen
        automatically via the Laplacian eigengap of the similarity matrix
        (R uses a consensus-k-means eigengap; here the eigengap is taken directly
        on the SNN-smoothed similarity — see Notes).
    method
        ``"kmeans"`` (:class:`sklearn.cluster.KMeans`) or ``"spectral"``
        (:class:`sklearn.cluster.SpectralClustering`, normalized-Laplacian
        eigenvectors).
    k_eigen
        Number of eigenvectors for spectral clustering (default = ``k``).
    n_init
        k-means restarts (R uses ``nstart=10``).
    seed
        Random seed.

    Returns
    -------
    pandas.Series
        Pathway → integer cluster label (1-based, matching R). Also stored at
        ``cc.netP['clustering'][type]``.

    Notes
    -----
    R's automatic ``k`` runs k-means for ``k = 2..min(N-1, 10)``, builds a
    consensus co-clustering matrix and takes its Laplacian eigengap. We simplify
    to a single eigengap computation on the SNN-smoothed similarity matrix
    (:func:`_eigengap_k`), which uses the same normalized-Laplacian /
    largest-eigengap rule and gives the same cluster count in the typical case.
    Pass an explicit ``k`` to bypass this.
    """
    if type not in ("functional", "structural"):
        raise ValueError("type must be 'functional' or 'structural'")
    if method not in ("kmeans", "spectral"):
        raise ValueError("method must be 'kmeans' or 'spectral'")

    emb = cc.netP.get("embedding", {}).get(type)
    if emb is None:
        raise RuntimeError(
            f"cc.netP['embedding']['{type}'] missing; run "
            f"net_embedding(cc, type='{type}') first."
        )
    Y = emb.to_numpy()
    n = Y.shape[0]
    if n < 3:
        raise ValueError(f"need at least 3 embedded pathways to cluster, got {n}")

    # choose k
    if k is None:
        # R netClustering: run k-means for k=2..min(N-1,10) on the EMBEDDING,
        # build a consensus co-clustering matrix, and take ITS Laplacian eigengap
        # (NOT the eigengap of the similarity matrix — that gives too few
        # clusters). This reproduces CellChat's automatic cluster-number choice.
        k = _consensus_eigengap_k(Y, seed=seed, n_init=n_init)
        k = int(np.clip(k, 2, min(n - 1, 10)))
        if cc.verbose:
            print(f"[interaction] inferred number of pathway clusters: k={k}")
    if k < 1:
        raise ValueError("k must be >= 1")
    k = min(k, n)

    if method == "kmeans":
        from sklearn.cluster import KMeans

        labels = KMeans(
            n_clusters=k, n_init=n_init, random_state=seed
        ).fit_predict(Y)
    else:  # spectral
        from sklearn.cluster import SpectralClustering

        sc = SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",
            n_neighbors=min(k_eigen or k, n - 1) if (k_eigen or k) else max(2, k),
            assign_labels="kmeans",
            random_state=seed,
        )
        labels = sc.fit_predict(Y)

    series = pd.Series(labels + 1, index=emb.index, name="cluster")  # 1-based like R

    clust_slot = cc.netP.setdefault("clustering", {})
    clust_slot[type] = series
    if cc.verbose:
        print(
            f"[interaction] clustered {n} pathways into "
            f"{series.nunique()} group(s) ({method}, {type})"
        )
    return series


# --------------------------------------------------------------------------- #
# 4. plot                                                                     #
# --------------------------------------------------------------------------- #
def plot_net_embedding(
    cc,
    *,
    type: str = "functional",
    ax=None,
    figsize: tuple[float, float] = (6, 6),
    label: bool = True,
    font_size: int = 8,
    point_size: float = 60.0,
    title: Optional[str] = None,
):
    """Scatter of the pathway embedding colored by cluster (R ``netVisual_embedding``).

    Parameters
    ----------
    cc
        :class:`CellComm` object; requires ``cc.netP['embedding'][type]`` (from
        :func:`net_embedding`) and, for coloring, ``cc.netP['clustering'][type]``
        (from :func:`net_clustering`).
    type
        ``"functional"`` or ``"structural"``.
    ax
        Existing matplotlib axes to draw on; a new figure is created if ``None``.
    figsize
        Figure size when creating a new figure.
    label
        Annotate each point with its pathway name.
    font_size
        Font size for the pathway labels.
    point_size
        Scatter marker size.
    title
        Optional axes title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the embedding scatter.
    """
    import matplotlib.pyplot as plt

    if type not in ("functional", "structural"):
        raise ValueError("type must be 'functional' or 'structural'")
    emb = cc.netP.get("embedding", {}).get(type)
    if emb is None:
        raise RuntimeError(
            f"cc.netP['embedding']['{type}'] missing; run "
            f"net_embedding(cc, type='{type}') first."
        )
    Y = emb.to_numpy()
    names = list(emb.index)

    clusters = cc.netP.get("clustering", {}).get(type)
    if clusters is not None:
        clusters = clusters.reindex(names)
        labels = clusters.to_numpy()
        uniq = sorted(pd.unique(labels[~pd.isna(labels)]))
    else:
        labels = np.zeros(len(names))
        uniq = [0]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    try:
        from sceleto.data._colors import zeileis_26, godsnot_64

        base = zeileis_26 if len(uniq) <= 26 else godsnot_64
    except Exception:
        cmap = plt.get_cmap("tab20")
        base = [cmap(i % 20) for i in range(max(20, len(uniq)))]

    color_for = {c: base[i % len(base)] for i, c in enumerate(uniq)}

    for c in uniq:
        mask = labels == c
        ax.scatter(
            Y[mask, 0],
            Y[mask, 1],
            s=point_size,
            color=color_for[c],
            edgecolors="white",
            linewidths=0.5,
            label=f"cluster {int(c)}" if not pd.isna(c) else "n/a",
            zorder=2,
        )

    if label:
        for (x, y), name in zip(Y, names):
            ax.annotate(
                name,
                (x, y),
                fontsize=font_size,
                ha="center",
                va="bottom",
                xytext=(0, 4),
                textcoords="offset points",
                zorder=3,
            )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    if clusters is not None and len(uniq) > 1:
        ax.legend(fontsize=font_size, frameon=False, loc="best")
    ax.set_title(title if title is not None else f"{type} pathway similarity")
    ax.margins(0.1)
    fig.tight_layout()
    return fig
