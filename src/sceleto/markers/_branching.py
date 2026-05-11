"""Branching marker computation on a HierarchyRun.

Branching markers explain the *splits* in a resolution hierarchy tree. For
each branching point (a parent node), find genes that distinguish one child
from its siblings, weighted by within-cluster rank and global IDF across
icls (unique path strings).

Score formula:
    final_score = exclusivity * (1 + rank_score) * log1p(idf)
        exclusivity = my_present - avg_sibling
        rank_score  = 1 / (rank + 1)
        idf         = global IDF across icls

This module operates on top of an existing :class:`HierarchyRun` (from
:func:`sceleto.markers.hierarchy`). It does not consider batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._gene_filter import GeneFilter
from ._hierarchy import HierarchyRun


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BranchingResult:
    """Output of :func:`branching_markers`.

    Attributes
    ----------
    hr
        The input :class:`HierarchyRun`.
    icls_gene_presence_df
        Per-icls table of (gene × level) rank/presence.
    gene_freq_df
        Global gene frequency / IDF across icls.
    score_df
        Merged table combining presence and IDF, used by traversal.
    tree_root
        Nested ``Dict[str, {"children": ..., "icls_indices": ..., "level_name": ...}]``
        representing the hierarchy tree.
    icls_to_path
        Mapping ``icls_id → [g0@x, g1@y, g2@z]``.
    markers
        ``{branch_name: [(gene, score, rank, exclusivity), ...]}`` — top-N
        markers per branch.
    params
        Computation parameters.
    """

    hr: HierarchyRun
    icls_gene_presence_df: pd.DataFrame
    gene_freq_df: pd.DataFrame
    score_df: pd.DataFrame
    tree_root: Dict[str, Any]
    icls_to_path: Dict[int, List[str]]
    markers: Dict[str, List[Tuple[str, float, float, float]]]
    params: Dict[str, Any]

    def print_tree(self, gene_filter: Optional[GeneFilter] = None) -> None:
        """Print the hierarchical tree with branching markers."""
        _print_tree(
            self.tree_root, self.score_df, tuple(self.hr.levels),
            gene_filter=gene_filter,
            n_top=self.params["n_top"],
            min_exclusivity=self.params["min_exclusivity"],
        )

    def get_markers(self, branch_name: str) -> List[Tuple[str, float, float, float]]:
        """Get marker list for a specific branch (path part)."""
        return self.markers.get(branch_name, [])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def branching_markers(
    hr: HierarchyRun,
    *,
    n_top: int = 5,
    min_exclusivity: float = 0.1,
    gene_filter: Optional[GeneFilter] = None,
    verbose: bool = True,
) -> BranchingResult:
    """Compute branching markers from a HierarchyRun.

    Parameters
    ----------
    hr
        :class:`HierarchyRun` from :func:`sceleto.markers.hierarchy`. Must have
        ``icls_full_dict`` and ``marker_rank_df``.
    n_top
        Number of top markers per branch.
    min_exclusivity
        Minimum (my_present − avg_sibling) to consider a gene.
    gene_filter
        Optional :class:`GeneFilter` to drop genes from consideration.
    verbose
        If True, print the tree with markers during traversal.

    Returns
    -------
    :class:`BranchingResult`
    """
    levels = tuple(hr.levels)
    if len(levels) != 3:
        raise ValueError(
            f"branching_markers currently supports 3-level hierarchies; got {len(levels)}"
        )

    # 1) Build icls × (gene × level) presence
    icls_gene_presence = _build_icls_gene_presence(hr)

    # 2) Compute global IDF across icls
    gene_freq = _compute_idf(icls_gene_presence)

    # 3) Build score_df
    score_df = pd.merge(
        icls_gene_presence, gene_freq.reset_index(), how="left", on="gene",
    )
    score_df["icls"] = score_df["icls"].astype(int)

    # 4) Build tree from icls paths
    tree_root, icls_to_path = _build_tree(hr)

    # 5) Traverse and collect markers
    if verbose:
        markers = _print_tree(
            tree_root, score_df, levels,
            gene_filter=gene_filter, n_top=n_top, min_exclusivity=min_exclusivity,
        )
    else:
        markers = _collect_markers(
            tree_root, score_df, levels,
            gene_filter=gene_filter, n_top=n_top, min_exclusivity=min_exclusivity,
        )

    return BranchingResult(
        hr=hr,
        icls_gene_presence_df=icls_gene_presence,
        gene_freq_df=gene_freq.reset_index(),
        score_df=score_df,
        tree_root=tree_root,
        icls_to_path=icls_to_path,
        markers=markers,
        params={"n_top": int(n_top), "min_exclusivity": float(min_exclusivity)},
    )


# ---------------------------------------------------------------------------
# Internal — score_df / tree construction
# ---------------------------------------------------------------------------


def _build_icls_gene_presence(hr: HierarchyRun) -> pd.DataFrame:
    """Per icls (unique path), build a (gene × level) rank/presence table."""
    g0, g1, g2 = hr.levels
    pieces: List[pd.DataFrame] = []
    for k, v in hr.icls_full_dict.items():
        l0, l1, l2 = v.split("|")
        piv = hr.marker_rank_df[
            hr.marker_rank_df["leiden"].isin([l0, l1, l2])
        ].pivot(index="gene", columns="resolution", values="rank")
        piv = piv.reindex(columns=[g0, g1, g2])
        df_binary = piv.notna().astype("int8")

        df = pd.merge(piv, df_binary, left_index=True, right_index=True, how="left")
        df.columns = [
            "rank_0", "rank_1", "rank_2",
            "present_0", "present_1", "present_2",
        ]
        df["n_levels"] = df["present_0"] + df["present_1"] + df["present_2"]
        df = pd.merge(
            pd.Series([k] * df.shape[0], name="icls"),
            df.reset_index(), how="left", left_index=True, right_index=True,
        )
        df = df.sort_values("n_levels", ascending=False)
        pieces.append(df)
    return pd.concat(pieces, axis=0).reset_index(drop=True)


def _compute_idf(icls_gene_presence: pd.DataFrame) -> pd.DataFrame:
    """Global DF/IDF across icls."""
    present_cols = [c for c in icls_gene_presence.columns if c.startswith("present_")]
    if present_cols:
        present_any = (
            icls_gene_presence[present_cols].fillna(False).astype(bool).any(axis=1)
        )
        df_use = icls_gene_presence.loc[present_any, ["icls", "gene"]].copy()
    else:
        df_use = icls_gene_presence[["icls", "gene"]].copy()

    N_icls = df_use["icls"].nunique()
    gene_freq = (
        df_use.groupby("gene")["icls"].nunique()
        .rename("df_global_icls").reset_index()
    )
    gene_freq["frac_icls"] = gene_freq["df_global_icls"] / N_icls
    gene_freq["idf_global_icls"] = (
        np.log((N_icls + 1) / (gene_freq["df_global_icls"] + 1)) + 1.0
    )
    gene_freq = (
        gene_freq.set_index("gene")
        .sort_values(["df_global_icls", "idf_global_icls"], ascending=[False, True])
    )
    gene_freq.columns = ["n_icls", "frac_icls", "idf_icls"]
    return gene_freq


def _build_tree(hr: HierarchyRun) -> Tuple[Dict[str, Any], Dict[int, List[str]]]:
    """Build nested dict tree from icls paths."""
    tree_root: Dict[str, Any] = {}
    icls_to_path: Dict[int, List[str]] = {}

    for icls_idx, path_str in hr.icls_full_dict.items():
        parts = path_str.split("|")
        current_level = tree_root
        path_list: List[str] = []
        for part in parts:
            path_list.append(part)
            if part not in current_level:
                current_level[part] = {
                    "children": {}, "icls_indices": [], "level_name": part,
                }
            current_level[part]["icls_indices"].append(int(icls_idx))
            current_level = current_level[part]["children"]
        icls_to_path[int(icls_idx)] = path_list

    return tree_root, icls_to_path


# ---------------------------------------------------------------------------
# Internal — branching scoring (per node)
# ---------------------------------------------------------------------------


def _get_node_stats(
    icls_indices: List[int],
    score_df: pd.DataFrame,
    rank_col: str,
    present_col: str,
) -> pd.DataFrame:
    """Aggregate gene stats for a set of icls indices.

    For each gene, take best (min) rank and max presence across the group.
    """
    subset = score_df[score_df["icls"].isin(icls_indices)].copy()
    subset[rank_col] = subset[rank_col].fillna(100)
    subset[present_col] = subset[present_col].fillna(0)

    return subset.groupby("gene").agg({
        rank_col: "min",
        present_col: "max",
        "idf_icls": "first",
    })


def _find_branching_markers(
    children_dict: Dict[str, Dict],
    score_df: pd.DataFrame,
    target_level_suffix: str,
    *,
    gene_filter: Optional[GeneFilter] = None,
    n_top: int = 5,
    min_exclusivity: float = 0.1,
) -> Dict[str, List[Tuple[str, float, float, float]]]:
    """Find markers distinguishing sibling nodes at a branching point.

    Returns ``{child_name: [(gene, score, rank, exclusivity), ...]}``.
    """
    if not children_dict:
        return {}

    rank_col = f"rank_{target_level_suffix}"
    present_col = f"present_{target_level_suffix}"

    children_stats = {
        name: _get_node_stats(node["icls_indices"], score_df, rank_col, present_col)
        for name, node in children_dict.items()
    }

    results: Dict[str, List[Tuple[str, float, float, float]]] = {}
    for target_child, target_stats in children_stats.items():
        siblings = [n for n in children_stats if n != target_child]
        scores: List[Tuple[str, float, float, float]] = []

        for gene in target_stats.index:
            my_present = target_stats.loc[gene, present_col]
            if my_present == 0:
                continue

            my_rank = target_stats.loc[gene, rank_col]
            idf = target_stats.loc[gene, "idf_icls"]

            sibling_present = [
                children_stats[sib].loc[gene, present_col]
                if gene in children_stats[sib].index else 0
                for sib in siblings
            ]
            avg_sibling = float(np.mean(sibling_present)) if siblings else 0.0
            exclusivity = my_present - avg_sibling
            if exclusivity <= min_exclusivity:
                continue

            if gene_filter is not None and not gene_filter(gene):
                continue

            rank_score = 1.0 / (my_rank + 1.0)
            final_score = exclusivity * (1 + rank_score) * np.log1p(idf)
            scores.append((gene, float(final_score), float(my_rank), float(exclusivity)))

        scores.sort(key=lambda x: x[1], reverse=True)
        results[target_child] = scores[:n_top]

    return results


# ---------------------------------------------------------------------------
# Internal — tree traversal
# ---------------------------------------------------------------------------


def _suffix_for_node(node: Dict[str, Any], levels: Tuple[str, str, str]) -> str:
    """Determine which level suffix (0/1/2) the children of this node belong to."""
    g0, g1, g2 = levels
    first_child = next(iter(node))
    if g1 in first_child:
        return "1"
    if g2 in first_child:
        return "2"
    return "0"


def _print_tree(
    node: Dict[str, Dict],
    score_df: pd.DataFrame,
    levels: Tuple[str, str, str],
    *,
    gene_filter: Optional[GeneFilter] = None,
    n_top: int = 5,
    min_exclusivity: float = 0.1,
    depth: int = 0,
    all_markers: Optional[Dict[str, List[Tuple[str, float, float, float]]]] = None,
) -> Dict[str, List[Tuple[str, float, float, float]]]:
    """Recursively print the tree, accumulating branching markers."""
    if all_markers is None:
        all_markers = {}

    if depth == 0:
        print("Hierarchical Marker Tree")
        print("=" * 30)

    if not node:
        return all_markers

    suffix = _suffix_for_node(node, levels)
    markers = _find_branching_markers(
        node, score_df, suffix,
        gene_filter=gene_filter, n_top=n_top, min_exclusivity=min_exclusivity,
    )

    indent = "    " * depth
    for child_name, child_node in node.items():
        marker_str = ""
        if child_name in markers and markers[child_name]:
            top_genes = [m[0] for m in markers[child_name]]
            marker_str = f" :: {', '.join(top_genes)}"
            all_markers[child_name] = markers[child_name]

        icls_info = ""
        if not child_node["children"]:
            icls_info = f" (icls {child_node['icls_indices']})"

        print(f"{indent}├── {child_name}{icls_info}{marker_str}")

        if child_node["children"]:
            _print_tree(
                child_node["children"], score_df, levels,
                gene_filter=gene_filter, n_top=n_top, min_exclusivity=min_exclusivity,
                depth=depth + 1, all_markers=all_markers,
            )

    return all_markers


def _collect_markers(
    node: Dict[str, Dict],
    score_df: pd.DataFrame,
    levels: Tuple[str, str, str],
    *,
    gene_filter: Optional[GeneFilter] = None,
    n_top: int = 5,
    min_exclusivity: float = 0.1,
    all_markers: Optional[Dict[str, List[Tuple[str, float, float, float]]]] = None,
) -> Dict[str, List[Tuple[str, float, float, float]]]:
    """Silent (no-print) version of tree traversal."""
    if all_markers is None:
        all_markers = {}
    if not node:
        return all_markers

    suffix = _suffix_for_node(node, levels)
    markers = _find_branching_markers(
        node, score_df, suffix,
        gene_filter=gene_filter, n_top=n_top, min_exclusivity=min_exclusivity,
    )

    for child_name, child_node in node.items():
        if child_name in markers and markers[child_name]:
            all_markers[child_name] = markers[child_name]
        if child_node["children"]:
            _collect_markers(
                child_node["children"], score_df, levels,
                gene_filter=gene_filter, n_top=n_top, min_exclusivity=min_exclusivity,
                all_markers=all_markers,
            )
    return all_markers
