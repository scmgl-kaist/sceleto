"""NMF communication-pattern analysis — a Python port of CellChat's ``selectK``
and ``identifyCommunicationPatterns`` (``analysis.R``).

CellChat learns *communication patterns* by non-negative matrix factorization
(NMF) of a cell-group × signaling-pathway matrix. Each pattern couples a set of
cell groups (which cells share it) to a set of pathways (which signals they use),
revealing coordinated sending ("outgoing") or receiving ("incoming") programs.

Pipeline
--------
1. :func:`select_k` — scan a range of ranks ``k`` and report clustering-stability
   metrics (cophenetic correlation + silhouette) so the user can pick the number
   of patterns, mirroring CellChat's ``selectK`` (which calls
   ``NMF::nmfEstimateRank``).
2. :func:`identify_communication_patterns` — factor the matrix at the chosen
   ``k`` into ``W`` (cell groups × patterns) and ``H`` (patterns × pathways),
   normalize as CellChat does, and store the result on the :class:`CellComm`
   object. Mirrors ``identifyCommunicationPatterns``.
3. :func:`plot_communication_patterns` — two-panel heatmap of ``W`` and ``Hᵀ``.

R → Python mapping
------------------
- **NMF engine (identical algorithm, not an approximation).** R uses
  ``NMF::nmf(data, rank=k, method='lee', seed='nndsvd')`` — the Lee & Seung
  multiplicative-update rule for the Frobenius/Euclidean objective, seeded with
  NNDSVD. sklearn's :class:`~sklearn.decomposition.NMF` with ``solver='mu'``,
  ``beta_loss='frobenius'`` implements *exactly* that same Lee & Seung
  multiplicative-update rule, and ``init='nndsvd'`` is the same NNDSVD seeding.
  This is a like-for-like reimplementation on a standard, trusted package — no
  NMF math is hand-rolled here. Exact factor values can differ from R only by
  convergence tolerance and floating-point ordering; the algorithm is the same
  and the recovered patterns are equivalent. ``random_state=seed`` makes it
  reproducible.
- **selectK metrics (legitimate reimplementation).** R's ``nmfEstimateRank``
  reports the *cophenetic correlation* and *consensus silhouette* of NMF
  consensus clustering over ``nrun`` restarts. We reproduce both with standard
  packages: for each ``k`` we run sklearn NMF ``n_runs`` times, build a
  connectivity/consensus matrix from each run's hard cluster assignment (argmax
  over pattern membership in ``W``), average it, and compute (a) the cophenetic
  correlation between the consensus dissimilarity and the cophenetic distances
  of its average-linkage dendrogram (:mod:`scipy.cluster.hierarchy`), and (b)
  the mean silhouette of the consensus clustering (:mod:`sklearn.metrics`).
  These are the same statistics NMF reports; only the factorization stays 100%
  sklearn and the metric code uses scipy/scikit-learn instead of the R ``NMF``
  package.

The cell-group × pathway matrix and its column-normalization follow CellChat
exactly (``apply(prob, c(1,3), sum)`` for outgoing / ``c(2,3)`` for incoming,
then divide each column by its max).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only, avoid import cycle
    import matplotlib.pyplot as plt
    from ._communication import CellComm


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _netP(cc: "CellComm") -> tuple[np.ndarray, list[str], list[str]]:
    """Pull the ``prob`` array, pathway names, and group names off ``cc``.

    Raises a clear :class:`ValueError` when the pathway-level communication has
    not been computed yet.
    """
    netP = getattr(cc, "netP", None)
    if not netP or "prob" not in netP:
        raise ValueError(
            "cc.netP['prob'] is missing. Run "
            "cc.compute_communication_pathway() before pattern analysis."
        )
    prob = np.asarray(netP["prob"], dtype=float)
    if prob.ndim != 3:
        raise ValueError(
            f"cc.netP['prob'] must be a 3-D (K, K, nPathway) array, got shape {prob.shape}"
        )
    pathways = [str(p) for p in netP.get("pathways", [])]
    if not pathways:
        pathways = [f"Pathway {p + 1}" for p in range(prob.shape[2])]
    group_names = [str(g) for g in getattr(cc, "group_names", [])]
    if not group_names:
        group_names = [f"Group {i + 1}" for i in range(prob.shape[0])]
    return prob, pathways, group_names


def _build_data_use(
    prob: np.ndarray,
    pattern: str,
    group_names: list[str],
    pathways: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Cell-group × pathway matrix, column-max normalized, zero-rows dropped.

    Reproduces CellChat's construction:

    - ``pattern="outgoing"`` → ``apply(prob, c(1,3), sum)`` — sum over the
      RECEIVER axis, giving how much each group SENDS via each pathway.
    - ``pattern="incoming"`` → ``apply(prob, c(2,3), sum)`` — sum over the
      SENDER axis, giving how much each group RECEIVES via each pathway.

    Then each pathway column is divided by its maximum (``sweep(...,2,max,'/')``)
    and all-zero cell-group rows are dropped (``data[rowSums(data)!=0,]``).

    Returns the normalized matrix and the surviving row (group) names.
    """
    if pattern == "outgoing":
        data = prob.sum(axis=1)  # sum over receiver → K_sender × nPathway
    elif pattern == "incoming":
        data = prob.sum(axis=0)  # sum over sender → K_receiver × nPathway
    else:
        raise ValueError(
            f"pattern must be 'outgoing' or 'incoming', got {pattern!r}"
        )

    # column-normalize by the per-pathway maximum. A pathway that is entirely
    # zero after thresholding has max 0; guard the division and leave it at 0
    # (it will be dropped as an uninformative column only if fully zero).
    col_max = data.max(axis=0)
    safe = np.where(col_max > 0, col_max, 1.0)
    data = data / safe[None, :]

    # drop cell-group rows that contribute nothing (rowSums == 0)
    row_keep = data.sum(axis=1) != 0
    data = data[row_keep]
    kept_names = [g for g, k in zip(group_names, row_keep) if k]

    if data.shape[0] == 0:
        raise ValueError(
            "The cell-group × pathway matrix is all zeros; there is no "
            f"{pattern} communication to factorize."
        )
    return data, kept_names


def _run_nmf(
    data: np.ndarray,
    k: int,
    *,
    random_state: Optional[int] = None,
    init: str = "nndsvd",
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """One NMF factorization ``data ≈ W @ H`` at rank ``k`` via sklearn.

    The factorization is done entirely by :class:`sklearn.decomposition.NMF`
    with ``solver='mu', beta_loss='frobenius'`` — the same Lee & Seung
    multiplicative-update rule as R's ``NMF::nmf(method='lee')`` — and
    ``init='nndsvd'`` matching R's ``seed='nndsvd'``. Returns ``(W, H)`` with
    ``W`` of shape ``(n_rows, k)`` and ``H`` of ``(k, n_cols)``.

    ``init`` is left as ``'nndsvd'`` for the single deterministic factorization
    (matching R's ``seed='nndsvd'``). The consensus scan in :func:`select_k`
    passes ``init='random'`` with a distinct ``random_state`` per run, because
    NNDSVD is deterministic and would make every restart identical — the
    standard consensus-NMF practice. The solver/objective is unchanged in both
    cases.

    Notes
    -----
    sklearn warns that ``solver='mu'`` cannot move exact zeros produced by plain
    ``init='nndsvd'`` off zero. We keep ``'nndsvd'`` to match R and silence only
    that specific, expected warning; ``'nndsvda'``/``'nndsvdar'`` are available
    to callers who prefer sklearn's zero-filling variants.
    """
    import warnings

    from sklearn.decomposition import NMF
    from sklearn.exceptions import ConvergenceWarning

    model = NMF(
        n_components=k,
        init=init,
        solver="mu",
        beta_loss="frobenius",
        max_iter=max_iter,
        random_state=random_state,
        tol=1e-4,
    )
    with warnings.catch_warnings():
        # the nndsvd+mu zero-update note and non-convergence at this small scale
        # are expected; don't spam the user's console with them
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        W = model.fit_transform(data)
    H = model.components_
    return W, H


def _scale_r1(x: np.ndarray) -> np.ndarray:
    """CellChat ``scaleMat(x, 'r1')`` — divide each ROW by its row sum."""
    rs = x.sum(axis=1, keepdims=True)
    return np.divide(x, rs, out=np.zeros_like(x), where=rs != 0)


def _scale_c1(x: np.ndarray) -> np.ndarray:
    """CellChat ``scaleMat(x, 'c1')`` — divide each COLUMN by its column sum."""
    cs = x.sum(axis=0, keepdims=True)
    return np.divide(x, cs, out=np.zeros_like(x), where=cs != 0)


# ---------------------------------------------------------------------------
# 1. identify_communication_patterns
# ---------------------------------------------------------------------------
def identify_communication_patterns(
    cc: "CellComm",
    pattern: str = "outgoing",
    k: Optional[int] = None,
    *,
    seed: int = 0,
    max_iter: int = 500,
    store: bool = True,
    verbose: bool = True,
) -> dict:
    """NMF communication patterns (CellChat ``identifyCommunicationPatterns``).

    Factorizes the cell-group × pathway matrix into ``k`` latent patterns. Each
    pattern links a group of cells (rows of ``W``) to a group of pathways (rows
    of ``H``), exposing coordinated ``outgoing`` (sending) or ``incoming``
    (receiving) signaling programs.

    Parameters
    ----------
    cc
        A :class:`CellComm` object after
        :meth:`CellComm.compute_communication_pathway` (so ``cc.netP['prob']``
        exists).
    pattern
        ``"outgoing"`` sums over the receiver axis (who *sends*);
        ``"incoming"`` sums over the sender axis (who *receives*).
    k
        Number of patterns (NMF rank). Run :func:`select_k` first to choose it.
        Must satisfy ``1 <= k <= min(n_groups, n_pathways)``.
    seed
        Random seed passed to NMF for reproducibility.
    max_iter
        Maximum NMF iterations.
    store
        If ``True`` (default), store the result on ``cc.patterns[pattern]``.
    verbose
        Print progress.

    Returns
    -------
    dict
        ``{"pattern", "k", "data", "W", "H", "cell", "signaling",
        "cell_pattern"}`` where

        - ``W`` : DataFrame, index = cell groups, columns = ``Pattern 1..k``;
          each row is row-normalized (a cell group's pattern memberships sum to
          1), matching CellChat ``scaleMat(W, 'r1')``.
        - ``H`` : DataFrame, index = ``Pattern 1..k``, columns = pathways; each
          column is column-normalized (a pathway's pattern memberships sum to
          1), matching CellChat ``scaleMat(H, 'c1')``.
        - ``cell`` / ``signaling`` : tidy long-form of ``W`` / ``H`` (the R
          ``res.pattern$cell`` / ``res.pattern$signaling`` tables).
        - ``cell_pattern`` : Series mapping each cell group to its dominant
          pattern (``argmax`` over ``W``).
        - ``data`` : the input cell-group × pathway matrix (pre-NMF).

    Raises
    ------
    ValueError
        If ``cc.netP['prob']`` is missing, the matrix is all zeros, or ``k`` is
        out of range (``k`` missing, ``k < 1``, or ``k`` larger than the number
        of usable groups/pathways).
    """
    prob, pathways, group_names = _netP(cc)
    data, kept_names = _build_data_use(prob, pattern, group_names, pathways)

    n_groups, n_pathways = data.shape
    k_max = min(n_groups, n_pathways)
    if k is None:
        raise ValueError(
            "k is required. Run select_k(cc, pattern=...) to choose the number "
            f"of patterns (valid range 2..{k_max})."
        )
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    if k > k_max:
        raise ValueError(
            f"k={k} exceeds the factorizable rank min(n_groups={n_groups}, "
            f"n_pathways={n_pathways})={k_max}. Choose k <= {k_max}."
        )

    k = int(k)
    W, H = _run_nmf(data, k, random_state=seed, max_iter=max_iter)

    # CellChat normalization: rows of W sum to 1, columns of H sum to 1.
    W = _scale_r1(W)
    H = _scale_c1(H)

    pattern_names = [f"Pattern {i + 1}" for i in range(k)]
    W_df = pd.DataFrame(W, index=kept_names, columns=pattern_names)
    H_df = pd.DataFrame(H, index=pattern_names, columns=pathways)

    # tidy long-form (R: data_W / data_H via as.table)
    cell_long = (
        W_df.reset_index()
        .melt(id_vars="index", var_name="Pattern", value_name="Contribution")
        .rename(columns={"index": "CellGroup"})
    )
    sig_long = (
        H_df.reset_index()
        .melt(id_vars="index", var_name="Signaling", value_name="Contribution")
        .rename(columns={"index": "Pattern"})
    )

    # dominant pattern per cell group (argmax over the row)
    if k >= 1 and W_df.shape[0] > 0:
        cell_pattern = W_df.idxmax(axis=1)
    else:  # pragma: no cover - guarded above
        cell_pattern = pd.Series(dtype=object)

    result = {
        "pattern": pattern,
        "k": k,
        "data": pd.DataFrame(data, index=kept_names, columns=pathways),
        "W": W_df,
        "H": H_df,
        "cell": cell_long,
        "signaling": sig_long,
        "cell_pattern": cell_pattern,
    }

    if store:
        if not hasattr(cc, "patterns") or getattr(cc, "patterns") is None:
            cc.patterns = {}
        cc.patterns[pattern] = result

    if verbose:
        print(
            f"[interaction] identified {k} {pattern} communication pattern(s) "
            f"over {n_groups} cell group(s) and {n_pathways} pathway(s)"
        )
    return result


# ---------------------------------------------------------------------------
# 2. select_k
# ---------------------------------------------------------------------------
def _consensus_matrix(labels_runs: list[np.ndarray]) -> np.ndarray:
    """Average connectivity (consensus) matrix over a list of hard clusterings.

    Entry ``(i, j)`` = fraction of runs in which samples ``i`` and ``j`` land in
    the same cluster. This is exactly NMF's consensus matrix.
    """
    n = len(labels_runs[0])
    consensus = np.zeros((n, n), dtype=float)
    for labels in labels_runs:
        same = labels[:, None] == labels[None, :]
        consensus += same
    consensus /= len(labels_runs)
    return consensus


def _cophenetic_correlation(consensus: np.ndarray) -> float:
    """Cophenetic correlation of the consensus matrix (NMF stability metric).

    Correlate the consensus *dissimilarity* (``1 - consensus``, condensed) with
    the cophenetic distances of its average-linkage hierarchical clustering. A
    value near 1 means the consensus clustering is crisp/stable at this rank.
    """
    from scipy.cluster.hierarchy import linkage, cophenet
    from scipy.spatial.distance import squareform

    n = consensus.shape[0]
    if n < 3:
        # cophenetic correlation is undefined for < 3 objects
        return float("nan")
    dissim = 1.0 - consensus
    np.fill_diagonal(dissim, 0.0)
    # numerical symmetry cleanup before squareform (it is strict)
    dissim = (dissim + dissim.T) / 2.0
    condensed = squareform(dissim, checks=False)
    if not np.any(condensed > 0):
        # perfectly consensual (all pairs always together): correlation ill-
        # defined (zero variance); report 1.0 as the crispest possible result
        return 1.0
    Z = linkage(condensed, method="average")
    coph_corr, _ = cophenet(Z, condensed)
    return float(coph_corr)


def _silhouette(consensus: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette of ``labels`` under the consensus dissimilarity.

    Uses ``1 - consensus`` as a precomputed distance, matching NMF's
    ``silhouette.consensus``. Returns NaN when silhouette is undefined
    (< 2 clusters or a cluster of size < 2 makes it degenerate).
    """
    from sklearn.metrics import silhouette_score

    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= len(labels):
        return float("nan")
    dissim = 1.0 - consensus
    np.fill_diagonal(dissim, 0.0)
    dissim = (dissim + dissim.T) / 2.0
    try:
        return float(
            silhouette_score(dissim, labels, metric="precomputed")
        )
    except Exception:  # pragma: no cover - degenerate clustering
        return float("nan")


def select_k(
    cc: "CellChat",
    pattern: str = "outgoing",
    k_range=range(2, 11),
    *,
    n_runs: int = 30,
    seed: int = 10,
    max_iter: int = 500,
    verbose: bool = True,
) -> pd.DataFrame:
    """Rank selection for NMF communication patterns (CellChat ``selectK``).

    For each candidate ``k`` this runs NMF ``n_runs`` times (varying the seed),
    forms the consensus clustering of cell groups over the runs, and reports two
    stability metrics — the same ones CellChat's ``selectK`` reads out of
    ``NMF::nmfEstimateRank``:

    - **cophenetic** : cophenetic correlation of the consensus matrix. Drops
      sharply at the ``k`` beyond which the factorization is no longer stable.
    - **silhouette** : mean silhouette of the consensus clustering.

    Pick the ``k`` just before cophenetic correlation begins to fall (a common
    NMF heuristic).

    Parameters
    ----------
    cc
        :class:`CellComm` object after
        :meth:`CellComm.compute_communication_pathway`.
    pattern
        ``"outgoing"`` or ``"incoming"``.
    k_range
        Iterable of candidate ranks (default ``range(2, 11)`` → 2..10). Values
        above ``min(n_groups, n_pathways)`` are skipped with a warning.
    n_runs
        NMF restarts per ``k`` (CellChat default 30). Use a smaller value for a
        quick scan.
    seed
        Base seed; run ``r`` uses ``seed + r`` for reproducibility.
    max_iter
        Maximum NMF iterations.
    verbose
        Print progress.

    Returns
    -------
    pd.DataFrame
        Columns ``k``, ``cophenetic``, ``silhouette`` (one row per usable ``k``),
        sorted by ``k``. Feed the chosen ``k`` to
        :func:`identify_communication_patterns`.

    Raises
    ------
    ValueError
        If ``cc.netP['prob']`` is missing, the matrix is all zeros, or no ``k``
        in ``k_range`` is usable (all exceed the factorizable rank).
    """
    prob, pathways, group_names = _netP(cc)
    data, kept_names = _build_data_use(prob, pattern, group_names, pathways)

    n_groups, n_pathways = data.shape
    k_max = min(n_groups, n_pathways)
    ks = sorted({int(k) for k in k_range})
    usable = [k for k in ks if 1 <= k <= k_max]
    skipped = [k for k in ks if k not in usable]
    if verbose and skipped:
        print(
            f"[interaction] skipping k={skipped}: exceed factorizable rank "
            f"min(n_groups={n_groups}, n_pathways={n_pathways})={k_max}"
        )
    if not usable:
        raise ValueError(
            f"No usable k in {list(ks)}: the factorizable rank is "
            f"min(n_groups={n_groups}, n_pathways={n_pathways})={k_max}. "
            "Provide k_range values within that bound."
        )

    if n_runs < 1:
        raise ValueError(f"n_runs must be >= 1, got {n_runs}")

    rows = []
    for k in usable:
        labels_runs = []
        for r in range(n_runs):
            # consensus runs need restart diversity → random init with a
            # distinct seed; the solver/objective is the same sklearn Lee-update
            W, H = _run_nmf(
                data, k, random_state=seed + r, init="random", max_iter=max_iter
            )
            # R NMF::nmfEstimateRank clusters the SAMPLES (= pathways = columns of
            # the input, = columns of H), driven by the coefficient matrix H —
            # NOT the cell groups (rows of W). Assign each pathway its dominant
            # pattern via H.argmax over patterns (axis 0).
            labels_runs.append(H.argmax(axis=0))
        consensus = _consensus_matrix(labels_runs)
        # representative clustering for the silhouette: the consensus itself,
        # clustered by average linkage into k groups
        rep_labels = _consensus_labels(consensus, k)
        coph = _cophenetic_correlation(consensus)
        sil = _silhouette(consensus, rep_labels)
        rows.append({"k": k, "cophenetic": coph, "silhouette": sil})
        if verbose:
            print(
                f"[interaction] k={k}: cophenetic={coph:.4f}, "
                f"silhouette={sil:.4f}"
            )

    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def _consensus_labels(consensus: np.ndarray, k: int) -> np.ndarray:
    """Cut the consensus dendrogram into ``k`` clusters (average linkage)."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    n = consensus.shape[0]
    if n <= k:
        # already at/below k objects: every object is its own (or a trivial)
        # cluster; hand back index labels
        return np.arange(n)
    dissim = 1.0 - consensus
    np.fill_diagonal(dissim, 0.0)
    dissim = (dissim + dissim.T) / 2.0
    condensed = squareform(dissim, checks=False)
    if not np.any(condensed > 0):
        return np.zeros(n, dtype=int)
    Z = linkage(condensed, method="average")
    return fcluster(Z, t=k, criterion="maxclust")


# ---------------------------------------------------------------------------
# 3. plot_communication_patterns
# ---------------------------------------------------------------------------
def plot_communication_patterns(
    cc: "CellChat",
    pattern: str = "outgoing",
    *,
    result: Optional[dict] = None,
    figsize: tuple[float, float] = (10, 6),
    cmap: str = "Spectral_r",
    title: Optional[str] = None,
) -> "plt.Figure":
    """Two-panel heatmap of the communication patterns (CellChat river view).

    Left panel: ``W`` (cell groups × patterns) — which cells drive each pattern.
    Right panel: ``Hᵀ`` (pathways × patterns) — which pathways define each
    pattern. Together they read like CellChat's ``identifyCommunicationPatterns``
    heatmap pair.

    Parameters
    ----------
    cc
        :class:`CellComm` object. Uses ``cc.patterns[pattern]`` unless a
        ``result`` dict is passed directly.
    pattern
        ``"outgoing"`` or ``"incoming"``.
    result
        A result dict from :func:`identify_communication_patterns`. If ``None``,
        read ``cc.patterns[pattern]`` (raising if it has not been computed).
    figsize
        Figure size.
    cmap
        Matplotlib colormap (default ``"Spectral_r"`` ≈ CellChat's reversed
        Spectral).
    title
        Overall figure title.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If no pattern result is available for ``pattern``.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if result is None:
        patterns = getattr(cc, "patterns", None)
        if not patterns or pattern not in patterns:
            raise ValueError(
                f"No {pattern!r} pattern result found. Run "
                f"identify_communication_patterns(cc, pattern={pattern!r}, k=...) "
                "first (or pass result=...)."
            )
        result = patterns[pattern]

    W = result["W"]
    H = result["H"]

    fig, axes = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1.2]}
    )

    # left: cell groups × patterns
    sns.heatmap(
        W, cmap=cmap, ax=axes[0], cbar_kws={"label": "Contribution"},
        linewidths=0.3, linecolor="white", vmin=0.0,
    )
    axes[0].set_title("Cell patterns", fontsize=10)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Cell group")
    axes[0].tick_params(axis="x", labelrotation=90)
    axes[0].tick_params(axis="y", labelrotation=0)

    # right: pathways × patterns (transpose of H)
    sns.heatmap(
        H.T, cmap=cmap, ax=axes[1], cbar_kws={"label": "Contribution"},
        linewidths=0.3, linecolor="white", vmin=0.0,
    )
    axes[1].set_title("Communication patterns", fontsize=10)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Signaling pathway")
    axes[1].tick_params(axis="x", labelrotation=90)
    axes[1].tick_params(axis="y", labelrotation=0)

    if title is None:
        title = f"{pattern.capitalize()} signaling patterns"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
