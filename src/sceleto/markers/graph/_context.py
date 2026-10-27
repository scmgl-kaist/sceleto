from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy import sparse


@dataclass(frozen=True)
class MarkerContext:
    """Cached matrices for marker pipeline.

    Notes
    -----
    - Matrices are (n_groups, n_genes) aligned to `groups` and `genes`.
    - `mean` is unnormalized group-wise mean expression.
    - `mean_norm` is per-gene normalized mean (0..1) by max across groups.
    - `n_expr` is number of expressing cells per gene per group (nnz).
    - `frac_expr` is fraction of expressing cells per gene per group (n_expr / n_cells_in_group).
    """
    groupby: str
    groups: List[str]          # length C
    genes: np.ndarray          # shape (G,)

    group_to_idx: Dict[str, int]

    mean: np.ndarray
    mean_norm: np.ndarray
    n_expr: np.ndarray
    frac_expr: np.ndarray

    # number of cells per group (aligned to `groups`)
    n_cells: np.ndarray        # shape (C,)

    # Graph-related caches (phase 1: only undirected PAGA edges)
    undirected_edges: Optional[List[Tuple[str, str]]] = None
    pos_df: Optional[pd.DataFrame] = None

    # Groups that had no PAGA edge after top-k trimming, and the fallback edges
    # that were added for them (empty when no fallback was needed).
    isolated_groups: Tuple[str, ...] = ()
    fallback_edges: Tuple[Tuple[str, str], ...] = ()

    # Source AnnData (stored for downstream convenience, e.g. .plot())
    adata: Optional[object] = None

    def get_group_idx(self, group: str) -> int:
        """Return row index for a group label."""
        return self.group_to_idx[str(group)]

    def to_mean_df(self) -> pd.DataFrame:
        """Convenience: convert `mean` matrix to DataFrame."""
        return pd.DataFrame(self.mean, index=self.groups, columns=self.genes)

    def to_mean_norm_df(self) -> pd.DataFrame:
        """Convenience: convert `mean_norm` matrix to DataFrame."""
        return pd.DataFrame(self.mean_norm, index=self.groups, columns=self.genes)

    def to_n_cells_series(self) -> pd.Series:
        """Return cell counts as a Series aligned to ctx.groups."""
        # Series index is cluster label, value is cell count
        return pd.Series(self.n_cells, index=self.groups, name="n_cells")


def get_groups(adata, groupby: str, exclude: Optional[List[str]] = None) -> List[str]:
    """Get group labels with a stable order."""
    if groupby not in adata.obs:
        raise KeyError(f"`groupby`='{groupby}' not found in adata.obs.")

    s = adata.obs[groupby]
    if isinstance(s.dtype, CategoricalDtype):
        groups = [str(x) for x in s.cat.categories]
    else:
        groups = [str(x) for x in pd.unique(s.astype(str))]

    if exclude:
        ex = set(map(str, exclude))
        groups = [g for g in groups if g not in ex]

    return groups


def top_k_edges_symmetric(con_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep top-k edges per node and symmetrize."""
    if con_df.shape[0] != con_df.shape[1]:
        raise ValueError("con_df must be square.")
    if not con_df.index.equals(con_df.columns):
        raise ValueError("con_df index/columns must match.")

    m = con_df.copy()
    np.fill_diagonal(m.values, 0.0)

    keep = np.zeros_like(m.values, dtype=bool)
    if k > 0:
        for i in range(m.shape[0]):
            row = m.values[i]
            idx = np.argsort(row)[::-1][:k]
            keep[i, idx] = row[idx] > 0

    keep = keep | keep.T
    return m.where(keep, other=0.0)


def build_context(
    adata,
    groupby: str,
    *,
    use_raw: bool = True,
    exclude: Optional[List[str]] = None,
    min_cells_per_group: int = 0,
    min_expr_cells_per_gene: int = 0,
    dtype=np.float32,
    k: Union[int, Literal["all"]] = 5,
    isolated: Literal["nearest", "all", "error"] = "nearest",
) -> MarkerContext:
    """Build MarkerContext with expression stats + cluster-pair graph.

    Parameters
    ----------
    k
        Cluster-pair graph topology:
        - int: keep top-k PAGA edges per node (requires PAGA in adata.uns).
        - "all": complete graph of all C(n, 2) pairs; cluster positions from
          UMAP centroids (requires adata.obsm['X_umap']).
    isolated
        What to do with a group that has no PAGA edge at all after top-k
        trimming (its kNN neighbors all lie inside the group, so PAGA
        connectivity to every other group is 0). Without an edge such a group
        gets no FC/delta and every gene scores 0. Only used when `k` is int.
        - "nearest": connect it to its `k` nearest groups by centroid distance
          in the embedding neighbors were built on (``use_rep``, else X_pca,
          else X_umap).
        - "all": connect it to every other group.
        - "error": raise.
        A warning lists the affected groups; see ``ctx.isolated_groups`` and
        ``ctx.fallback_edges``.
    """
    # context with expression values
    ctx = _build_expression_context(
        adata,
        groupby,
        use_raw=use_raw,
        exclude=exclude,
        min_cells_per_group=min_cells_per_group,
        min_expr_cells_per_gene=min_expr_cells_per_gene,
        dtype=dtype,
    )

    # context with graph structures
    if k == "all":
        undirected_edges, pos_df = _build_all_pairs_graph(
            adata,
            groupby=groupby,
            groups=ctx.groups,
        )
    elif isinstance(k, int):
        undirected_edges, pos_df = _extract_paga_undirected_edges(
            adata,
            groups=ctx.groups,  # enforce exact group order
            k=k,
        )
        undirected_edges, isolated_groups, fallback_edges = _connect_isolated_groups(
            adata,
            groupby=groupby,
            groups=ctx.groups,
            undirected_edges=undirected_edges,
            k=k,
            isolated=isolated,
        )
        ctx = replace(
            ctx,
            isolated_groups=tuple(isolated_groups),
            fallback_edges=tuple(fallback_edges),
        )
    else:
        raise ValueError(f"k must be int or 'all', got {k!r}")

    return replace(ctx, undirected_edges=undirected_edges, pos_df=pos_df, adata=adata)


def _build_expression_context(
    adata,
    groupby: str,
    *,
    use_raw: bool,
    exclude: Optional[List[str]],
    min_cells_per_group: int,
    min_expr_cells_per_gene: int,
    dtype,
) -> MarkerContext:
    """Internal: compute expression summary matrices."""
    from sceleto._expr import resolve_expression

    groups = get_groups(adata, groupby, exclude=exclude)
    group_to_idx = {g: i for i, g in enumerate(groups)}

    X, var_names, source = resolve_expression(adata)
    genes = var_names.to_numpy()

    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr()

    C = len(groups)
    G = X.shape[1]
    if len(genes) != G:
        raise ValueError(f"Gene dimension mismatch: len(var_names)={len(genes)} vs X.shape[1]={G}")

    mean = np.zeros((C, G), dtype=dtype)        # mean expression
    n_expr = np.zeros((C, G), dtype=dtype)      # number of expressing cells
    frac_expr = np.zeros((C, G), dtype=dtype)   # fraction of expressing cells

    n_cells_arr = np.zeros(C, dtype=np.int32)   # cell count

    obs_groups = adata.obs[groupby].astype(str).to_numpy()

    for i, g in enumerate(groups):
        mask = (obs_groups == g)
        n_cells = int(mask.sum())
        n_cells_arr[i] = n_cells

        if n_cells < min_cells_per_group:
            continue

        Xg = X[mask]

        expr_cells = Xg.getnnz(axis=0).astype(dtype, copy=False)
        n_expr[i] = expr_cells
        frac_expr[i] = expr_cells / max(n_cells, 1)

        m = np.asarray(Xg.mean(axis=0)).ravel().astype(dtype, copy=False)

        if min_expr_cells_per_gene > 0:
            m = m.copy()
            m[expr_cells < min_expr_cells_per_gene] = 0

        mean[i] = m

    max_per_gene = mean.max(axis=0)
    mean_norm = mean.copy()
    nz = max_per_gene > 0
    mean_norm[:, nz] /= max_per_gene[nz]

    return MarkerContext(
        groupby=groupby,
        groups=groups,
        genes=genes,
        group_to_idx=group_to_idx,
        mean=mean,
        mean_norm=mean_norm,
        n_expr=n_expr,
        frac_expr=frac_expr,
        n_cells=n_cells_arr,
        undirected_edges=None,
        pos_df=None,
    )


def _build_all_pairs_graph(
    adata,
    groupby: str,
    *,
    groups: List[str],
) -> Tuple[List[Tuple[str, str]], pd.DataFrame]:
    """Build complete cluster-pair graph (no PAGA) + UMAP centroid positions.

    All C(n, 2) cluster pairs become edges. Cluster positions are computed as
    the mean UMAP coordinate of cells in each cluster.

    Raises
    ------
    ValueError
        If adata.obsm['X_umap'] is missing (cannot compute centroids).
    """
    obsm = getattr(adata, "obsm", {})
    if "X_umap" not in obsm:
        raise ValueError(
            "k='all' requires adata.obsm['X_umap'] for cluster positions. "
            "Compute UMAP first (sc.tl.umap(adata))."
        )

    umap = np.asarray(obsm["X_umap"])
    if umap.shape[1] < 2:
        raise ValueError(
            f"adata.obsm['X_umap'] must have >=2 dims, got shape {umap.shape}."
        )

    obs_groups = adata.obs[groupby].astype(str).to_numpy()

    pos_rows = []
    for g in groups:
        mask = (obs_groups == g)
        if not mask.any():
            raise ValueError(f"Group '{g}' has no cells in adata.obs['{groupby}'].")
        pos_rows.append(umap[mask].mean(axis=0))

    pos_df = pd.DataFrame(np.vstack(pos_rows), index=groups)

    # Complete graph: all unique unordered pairs.
    undirected_edges = sorted(
        tuple(sorted((groups[i], groups[j])))
        for i in range(len(groups))
        for j in range(i + 1, len(groups))
    )

    return undirected_edges, pos_df


def _extract_paga_undirected_edges(
    adata,
    *,
    groups: List[str],
    k: int,
) -> Tuple[List[Tuple[str, str]], pd.DataFrame]:
    """Extract trimmed PAGA undirected edges aligned to provided `groups` order.

    Requires:
    - adata.uns['paga']['connectivities']
    - adata.uns['paga']['pos']
    """
    paga = adata.uns.get("paga", None)
    if paga is None or "connectivities" not in paga:
        raise ValueError("PAGA connectivities not found. Run sc.tl.paga(adata, ...) first.")

    if "pos" not in paga:
        raise ValueError("PAGA pos not found. Run sc.pl.paga(adata, show=False) after sc.tl.paga(...) to populate paga['pos'].")

    con = paga["connectivities"]
    con = con.toarray() if sparse.issparse(con) else np.asarray(con)

    con_df = pd.DataFrame(con, index=groups, columns=groups)

    trimmed_con_df = top_k_edges_symmetric(con_df, k=k)
    mat = trimmed_con_df.to_numpy()

    pos_df = pd.DataFrame(paga["pos"], index=groups)

    rows, cols = np.where((mat > 0) & (~np.eye(mat.shape[0], dtype=bool)))

    undirected_edges = sorted({
        tuple(sorted((str(trimmed_con_df.index[r]), str(trimmed_con_df.columns[c]))))
        for r, c in zip(rows, cols)
    })

    return undirected_edges, pos_df


def _group_centroids(
    adata,
    groupby: str,
    groups: List[str],
) -> Tuple[np.ndarray, str]:
    """Per-group centroids in the embedding neighbors were built on.

    Preference: ``adata.uns['neighbors']['params']['use_rep']`` -> X_pca -> X_umap.
    Returns (centroids (C, d), rep_name).
    """
    obsm = getattr(adata, "obsm", {})
    candidates: List[str] = []
    use_rep = adata.uns.get("neighbors", {}).get("params", {}).get("use_rep", None)
    if use_rep:
        candidates.append(str(use_rep))
    candidates += ["X_pca", "X_umap"]

    rep = next((r for r in candidates if r in obsm), None)
    if rep is None:
        raise ValueError(
            "Cannot compute cluster centroids for isolated-group fallback: none of "
            f"{candidates} found in adata.obsm. Set isolated='error' or k='all'."
        )

    emb = np.asarray(obsm[rep])
    obs_groups = adata.obs[groupby].astype(str).to_numpy()
    cent = np.vstack([emb[obs_groups == g].mean(axis=0) for g in groups])
    return cent, rep


def _connect_isolated_groups(
    adata,
    *,
    groupby: str,
    groups: List[str],
    undirected_edges: List[Tuple[str, str]],
    k: int,
    isolated: Literal["nearest", "all", "error"],
) -> Tuple[List[Tuple[str, str]], List[str], List[Tuple[str, str]]]:
    """Add fallback edges for groups left with degree 0 after PAGA top-k trimming.

    A group whose kNN neighbors all fall inside itself has PAGA connectivity 0
    to every other group. With no edge, FC/delta are never computed for it and
    every gene ties at score 0, so the ranking silently degrades to the
    alphabetical order of gene names. Here such groups get explicit edges
    instead (see `isolated` in :func:`build_context`).

    Returns (edges, isolated_groups, fallback_edges).
    """
    if isolated not in ("nearest", "all", "error"):
        raise ValueError(f"isolated must be 'nearest', 'all' or 'error', got {isolated!r}")

    degree = {g: 0 for g in groups}
    for a, b in undirected_edges:
        degree[a] += 1
        degree[b] += 1
    iso = [g for g in groups if degree[g] == 0]
    if not iso or len(groups) < 2:
        return undirected_edges, [], []

    if isolated == "error":
        raise ValueError(
            f"Groups {iso} have no PAGA edge after top-{k} trimming (PAGA connectivity "
            "to every other group is 0), so no marker can be scored for them. "
            "Use isolated='nearest'/'all', or k='all'."
        )

    fallback: set = set()
    if isolated == "all":
        for g in iso:
            for h in groups:
                if h != g:
                    fallback.add(tuple(sorted((g, h))))
        how = "every other group"
    else:
        cent, rep = _group_centroids(adata, groupby, groups)
        gidx = {g: i for i, g in enumerate(groups)}
        n_nb = min(k, len(groups) - 1)
        for g in iso:
            d = np.linalg.norm(cent - cent[gidx[g]], axis=1)
            d[gidx[g]] = np.inf
            for j in np.argsort(d)[:n_nb]:
                fallback.add(tuple(sorted((g, groups[j]))))
        how = f"the {n_nb} nearest group(s) by {rep} centroid distance"

    fallback_edges = sorted(fallback - set(undirected_edges))
    warnings.warn(
        f"Groups {iso} have no PAGA edge after top-{k} trimming; connected each to "
        f"{how} (isolated='{isolated}', {len(fallback_edges)} edge(s) added). "
        "Inspect ctx.fallback_edges; set isolated='error' to disallow.",
        UserWarning,
        stacklevel=3,
    )
    edges = sorted(set(undirected_edges) | set(fallback_edges))
    return edges, iso, fallback_edges
