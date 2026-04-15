from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd

from ._gene_filter import GeneFilter


@dataclass(frozen=True)
class BatchExpression:
    """Per-batch expression arrays for one resolution level."""

    mean: np.ndarray  # (n_groups, n_batches, n_genes)
    frac_expr: np.ndarray  # (n_groups, n_batches, n_genes)
    n_cells: np.ndarray  # (n_groups, n_batches) — cells per group/batch
    groups: List[str]
    batches: List[str]
    genes: np.ndarray
    group_to_idx: Dict[str, int]


@dataclass
class HierarchyRun:
    # Inputs / meta
    levels: List[str]
    params: Dict[str, Any]

    # Key artifacts
    icls_full_dict: Dict[str, str]
    icls_path_df: pd.DataFrame
    marker_rank_df: pd.DataFrame

    # Full (untruncated) ranked gene lists per leiden ID
    full_gene_lists: Dict[str, List[str]]

    # Expression contexts per resolution (groupby -> MarkerContext)
    contexts: Dict[str, Any]

    # Per-batch expression data (groupby -> BatchExpression); None if no batch_key
    batch_expression: Optional[Dict[str, BatchExpression]]

    # Batch key used (None if not provided)
    batch_key: Optional[str]

    def interactive_viewer(
        self,
        adata,
        *,
        save: str = "interactive_viewer.html",
        n_top: Optional[int] = None,
    ) -> None:
        """Generate an interactive HTML viewer for marker comparison.

        Parameters
        ----------
        adata
            AnnData with ``obs['icls']`` (set by hierarchy) and
            ``obsm['X_umap']``.
        save
            Output HTML file path.
        n_top
            Number of top markers per cluster. Defaults to the value
            used in the hierarchy run.
        """
        from ._viewer import build_interactive_html

        if n_top is None:
            n_top = self.params["n_top_markers"]

        build_interactive_html(
            adata=adata,
            icls_full_dict=self.icls_full_dict,
            full_gene_lists=self.full_gene_lists,
            n_top=n_top,
            save=save,
        )

    def compare_markers(
        self,
        icls: str,
        *,
        figsize=None,
        gene_filter: Optional[GeneFilter] = None,
        return_genes: bool = False,
    ):
        """Visualize top-N marker overlap across levels for a given icls."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        leiden_list = self.icls_path_df.set_index("icls").loc[icls, self.levels].tolist()
        n = self.params["n_top_markers"]

        sets = _build_gene_sets(leiden_list, self.full_gene_lists, n, gene_filter)
        union = sorted(set().union(*sets))

        df = pd.DataFrame(
            {lid: [1 if g in s else 0 for g in union] for lid, s in zip(leiden_list, sets)},
            index=union,
        ).sort_values(leiden_list, ascending=False).T

        if return_genes:
            return union

        from matplotlib.patches import Patch

        if figsize is None:
            figsize = (len(union) * 0.4, 2)

        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.get_cmap("Blues")
        sns.heatmap(
            df, cmap=cmap, linewidths=0.5, linecolor="black",
            cbar=False, xticklabels=True, ax=ax,
        )
        ax.set_title(f"Marker genes for path {icls}")
        ax.set_xlabel("")

        legend_handles = [
            Patch(facecolor=cmap(1.0), edgecolor="black", label="in top-N markers"),
            Patch(facecolor=cmap(0.0), edgecolor="black", label="not in top-N markers"),
        ]
        ax.legend(
            handles=legend_handles, loc="upper left",
            bbox_to_anchor=(1.0, 1.0), fontsize=7, frameon=False,
        )
        plt.close(fig)
        return fig

    def compare_markers_batch(
        self,
        icls: str,
        *,
        figsize=None,
        gene_filter: Optional[GeneFilter] = None,
        return_genes: bool = False,
        cap_percentile: float = 95.0,
    ):
        """Visualize top-N marker overlap with per-batch expression strips.

        Each strip in a (level, gene) cell encodes one batch:

          - grey  : batch has no cells in this cluster (no data)
          - white : batch has cells but mean expression is 0
          - red   : colored by mean / global_cap, clipped to [0, 1]

        The global cap is the ``cap_percentile``-th percentile of all positive
        active-batch means in the displayed cells, so a single outlier batch
        does not dominate the scale and cross-cell colors stay comparable.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.patches import Rectangle, Patch

        if self.batch_expression is None:
            raise ValueError(
                "No batch expression data. Re-run hierarchy() with batch_key."
            )

        leiden_list = (
            self.icls_path_df.set_index("icls").loc[icls, self.levels].tolist()
        )
        n = self.params["n_top_markers"]

        sets = _build_gene_sets(leiden_list, self.full_gene_lists, n, gene_filter)
        union = sorted(set().union(*sets))

        # sort genes by presence pattern
        presence_df = pd.DataFrame(
            {lid: [1 if g in s else 0 for g in union]
             for lid, s in zip(leiden_list, sets)},
            index=union,
        ).sort_values(leiden_list, ascending=False)
        union = presence_df.index.tolist()

        if return_genes:
            return union

        n_rows = len(leiden_list)
        n_cols = len(union)

        batch_data, global_cap = _collect_batch_values(
            leiden_list, union, self.batch_expression,
            cap_percentile=cap_percentile,
        )
        n_batches = len(next(iter(self.batch_expression.values())).batches)

        if figsize is None:
            figsize = (n_cols * 0.6, n_rows * 0.8 + 0.5)

        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.cm.Reds
        grey_color = "#cccccc"
        cell_w, cell_h = 1.0, 1.0
        strip_w = cell_w / n_batches

        for i, lid in enumerate(leiden_list):
            y = n_rows - 1 - i
            for j, gene in enumerate(union):
                if gene in sets[i]:
                    means_sorted, active_sorted = batch_data[i][j]
                    for b in range(n_batches):
                        if not active_sorted[b]:
                            facecolor = grey_color
                        elif means_sorted[b] == 0:
                            facecolor = "white"
                        else:
                            if global_cap > 0:
                                norm_val = min(means_sorted[b] / global_cap, 1.0)
                            else:
                                norm_val = 0.0
                            facecolor = cmap(norm_val)
                        ax.add_patch(Rectangle(
                            (j * cell_w + b * strip_w, y * cell_h),
                            strip_w, cell_h,
                            facecolor=facecolor, edgecolor="none",
                        ))
                ax.add_patch(Rectangle(
                    (j * cell_w, y * cell_h), cell_w, cell_h,
                    facecolor="none", edgecolor="black", linewidth=0.5,
                ))

        ax.set_xlim(0, n_cols * cell_w)
        ax.set_ylim(0, n_rows * cell_h)
        ax.set_xticks([j * cell_w + cell_w / 2 for j in range(n_cols)])
        ax.set_xticklabels(union, rotation=90, ha="center", fontsize=8)
        ax.set_yticks([i * cell_h + cell_h / 2 for i in range(n_rows)])
        ax.set_yticklabels(leiden_list[::-1], fontsize=9)
        ax.set_title(f"Marker genes for path {icls} (per-batch)")

        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=global_cap or 1.0),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label(f"Mean expression\n(cap = {cap_percentile:.0f}p)")

        legend_handles = [
            Patch(facecolor=grey_color, edgecolor="black", label="not in cluster"),
            Patch(facecolor="white", edgecolor="black", label="zero expression"),
        ]
        ax.legend(
            handles=legend_handles, loc="upper left",
            bbox_to_anchor=(1.0, -0.05), fontsize=7, frameon=False,
        )

        plt.tight_layout()
        plt.close(fig)
        return fig


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_gene_sets(
    leiden_list: List[str],
    full_gene_lists: Dict[str, List[str]],
    n: int,
    gene_filter: Optional[GeneFilter],
) -> List[set]:
    """Build top-N gene sets per leiden ID, optionally filtered."""
    sets = []
    for lid in leiden_list:
        genes = full_gene_lists[lid]
        if gene_filter is not None:
            genes = gene_filter.filter(genes)
        sets.append(set(genes[:n]))
    return sets


def _collect_batch_values(
    leiden_list: List[str],
    union: List[str],
    batch_expression: Dict[str, BatchExpression],
    *,
    cap_percentile: float = 95.0,
) -> Tuple[List[List[Tuple[np.ndarray, np.ndarray]]], float]:
    """Collect sorted per-batch values and a global normalization cap."""
    n_rows = len(leiden_list)
    n_cols = len(union)
    n_batches = len(next(iter(batch_expression.values())).batches)

    raw_vals = np.zeros((n_rows, n_cols, n_batches), dtype=np.float32)
    active = np.zeros((n_rows, n_batches), dtype=bool)

    for i, lid in enumerate(leiden_list):
        groupby, group_name = lid.split("@", 1)
        be = batch_expression[groupby]
        g_idx = be.group_to_idx[group_name]
        active[i] = be.n_cells[g_idx] > 0
        gene_indices = {g: int(k) for k, g in enumerate(be.genes)}

        for j, gene in enumerate(union):
            if gene in gene_indices:
                raw_vals[i, j] = be.mean[g_idx, :, gene_indices[gene]]

    batch_data: List[List[Tuple[np.ndarray, np.ndarray]]] = []
    for i in range(n_rows):
        row: List[Tuple[np.ndarray, np.ndarray]] = []
        act_i = active[i]
        for j in range(n_cols):
            means_j = raw_vals[i, j]
            order = np.lexsort((-means_j, ~act_i))
            row.append((means_j[order], act_i[order]))
        batch_data.append(row)

    pool = raw_vals[np.broadcast_to(active[:, None, :], raw_vals.shape)]
    pool = pool[pool > 0]
    if pool.size > 0:
        cap = float(np.percentile(pool, cap_percentile))
        if cap <= 0.0:
            cap = float(pool.max())
    else:
        cap = 0.0

    return batch_data, cap


def _compute_batch_expression(adata, ctx, batch_key):
    """Compute per-batch expression statistics for one resolution level."""
    from scipy import sparse

    from sceleto._expr import resolve_expression
    X, _, _ = resolve_expression(adata)
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr()

    groups = ctx.groups
    group_to_idx = ctx.group_to_idx
    genes = ctx.genes
    batches = sorted(adata.obs[batch_key].astype(str).unique().tolist())

    n_groups, n_batches, n_genes = len(groups), len(batches), len(genes)
    mean = np.zeros((n_groups, n_batches, n_genes), dtype=np.float32)
    frac_expr = np.zeros((n_groups, n_batches, n_genes), dtype=np.float32)
    n_cells_arr = np.zeros((n_groups, n_batches), dtype=np.int32)

    obs_groups = adata.obs[ctx.groupby].astype(str).to_numpy()
    obs_batches = adata.obs[batch_key].astype(str).to_numpy()

    for g_name, g_idx in group_to_idx.items():
        g_mask = obs_groups == g_name
        for b_idx, b_name in enumerate(batches):
            mask = g_mask & (obs_batches == b_name)
            n_cells = int(mask.sum())
            n_cells_arr[g_idx, b_idx] = n_cells
            if n_cells == 0:
                continue
            Xsub = X[mask]
            mean[g_idx, b_idx] = np.asarray(Xsub.mean(axis=0)).ravel()
            frac_expr[g_idx, b_idx] = Xsub.getnnz(axis=0) / n_cells

    return BatchExpression(
        mean=mean, frac_expr=frac_expr, n_cells=n_cells_arr,
        groups=groups, batches=batches, genes=genes, group_to_idx=group_to_idx,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def hierarchy(
    adata: Any,
    markers_list: Sequence[Any],
    *,
    min_cells_for_path: Optional[int] = None,
    n_top_markers: int = 10,
    gene_filter: Optional[GeneFilter] = None,
    batch_key: Optional[str] = None,
) -> HierarchyRun:
    """Run cross-resolution hierarchy pipeline.

    Combines three resolution levels of marker outputs into icls
    (integration cell lineage strings) and prepares marker comparison.

    Parameters
    ----------
    min_cells_for_path
        Paths with fewer cells are reassigned to their major neighbor path
        via kNN connectivities. Default: ``int(adata.shape[0] * 0.005)``.
    """
    from scipy import sparse

    markers_list = list(markers_list)

    if min_cells_for_path is None:
        min_cells_for_path = max(int(adata.shape[0] * 0.005), 1)

    g0, g1, g2 = [mo.ctx.groupby for mo in markers_list]

    # 1) Ensure categorical dtype
    for g in (g0, g1, g2):
        if not pd.api.types.is_categorical_dtype(adata.obs[g]):
            adata.obs[g] = adata.obs[g].astype("category")
        adata.obs[g] = adata.obs[g].cat.set_categories(
            adata.obs[g].cat.categories, ordered=True,
        )

    # 2) Build per-cell path strings
    adata.obs["path"] = (
        f"{g0}@" + adata.obs[g0].astype(str)
        + "|" + f"{g1}@" + adata.obs[g1].astype(str)
        + "|" + f"{g2}@" + adata.obs[g2].astype(str)
    )

    # 3) Make path categorical with cartesian product order
    path_categories = [
        f"{g0}@{x}|{g1}@{y}|{g2}@{z}"
        for x in adata.obs[g0].cat.categories
        for y in adata.obs[g1].cat.categories
        for z in adata.obs[g2].cat.categories
    ]
    adata.obs["path"] = pd.Categorical(
        adata.obs["path"], categories=path_categories, ordered=True,
    )

    # 4) Identify small paths and reassign their cells to neighbor paths
    small_paths = (
        adata.obs["path"].value_counts()
        .loc[lambda s: s < min_cells_for_path].index
    )
    small_mask = adata.obs["path"].isin(small_paths).to_numpy()

    if small_mask.any():
        conn = adata.obsp["connectivities"]
        if not sparse.issparse(conn):
            conn = sparse.csr_matrix(conn)
        else:
            conn = conn.tocsr()

        # path labels for non-small cells; NA for small-path cells
        path_arr = adata.obs["path"].to_numpy(dtype=object, copy=True)
        path_arr[small_mask] = None

        small_indices = np.where(small_mask)[0]
        for idx in small_indices:
            row = conn[idx]
            nbr_indices = row.indices
            nbr_paths = path_arr[nbr_indices]
            # keep only neighbors with valid (non-small) paths
            valid = nbr_paths[nbr_paths != None]  # noqa: E711
            if len(valid) > 0:
                values, counts = np.unique(valid, return_counts=True)
                path_arr[idx] = values[counts.argmax()]

        adata.obs["path"] = pd.Categorical(
            path_arr, categories=path_categories, ordered=True,
        )

    present = set(adata.obs["path"].dropna().unique())
    new_categories = [c for c in path_categories if c in present]
    adata.obs["path"] = adata.obs["path"].cat.set_categories(
        new_categories, ordered=True,
    )

    # Build icls mapping
    icls_full_dict: Dict[str, str] = {
        str(i): path for i, path in enumerate(adata.obs["path"].cat.categories)
    }
    path_to_key = {v: k for k, v in icls_full_dict.items()}
    adata.obs["icls"] = adata.obs["path"].map(path_to_key).astype("string")

    # Build path dataframe
    df_icls_path = pd.DataFrame(
        pd.Series(icls_full_dict), columns=["icls_full"],
    )
    df_icls_path[g0] = [x.split("|")[0] for x in df_icls_path["icls_full"]]
    df_icls_path[g1] = [x.split("|")[1] for x in df_icls_path["icls_full"]]
    df_icls_path[g2] = [x.split("|")[2] for x in df_icls_path["icls_full"]]
    df_icls_path["root"] = df_icls_path[g0]
    df_icls_path = df_icls_path.reset_index(names="icls")

    # Build marker rank table and full gene lists
    rows: List[List[Any]] = []
    full_gene_lists: Dict[str, List[str]] = {}

    for level_key, mo in zip([g0, g1, g2], markers_list):
        for k, v in mo.markers.items():
            leiden_id = f"{level_key}@{k}"
            full_gene_lists[leiden_id] = list(v)
            genes = gene_filter.filter(v) if gene_filter is not None else v
            for i, gene in enumerate(genes[:n_top_markers]):
                rows.append([level_key, leiden_id, gene, i + 1])

    df_marker_rank = pd.DataFrame(
        rows, columns=["resolution", "leiden", "gene", "rank"],
    )

    # Build contexts dict
    contexts = {mo.ctx.groupby: mo.ctx for mo in markers_list}

    # Compute batch expression if requested
    batch_expression = None
    if batch_key is not None:
        batch_expression = {
            mo.ctx.groupby: _compute_batch_expression(adata, mo.ctx, batch_key)
            for mo in markers_list
        }

    return HierarchyRun(
        levels=[str(g0), str(g1), str(g2)],
        params={
            "min_cells_for_path": int(min_cells_for_path),
            "n_top_markers": int(n_top_markers),
        },
        icls_full_dict=icls_full_dict,
        icls_path_df=df_icls_path,
        marker_rank_df=df_marker_rank,
        full_gene_lists=full_gene_lists,
        contexts=contexts,
        batch_expression=batch_expression,
        batch_key=batch_key,
    )
