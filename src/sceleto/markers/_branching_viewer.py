"""Interactive HTML viewer for BranchingResult.

Combines the existing :func:`build_interactive_html` style (icls UMAP +
marker comparison heatmap) with a bottom panel.

Bottom panel has two modes:
- ``mgr=None`` (default): 3 small resolution UMAPs annotated with branching
  markers per cluster.
- ``mgr=<MarkerGraphRun>``: edge-activation graph showing, per gene from the
  marker comparison heatmap x-axis, which edges in the marker graph the gene
  is active on (uses :func:`MarkerGraphRun.plot_gene_edges_fc` data).
  Gene is selected by clicking the heatmap column, clicking a chip in the
  gene strip, mouse wheel on the panel, or ←/→ arrow keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ._viewer import _assign_icls_colors, _build_compare_data


def build_branching_html(
    adata: Any,
    br: Any,
    save: str,
    *,
    n_top: int = 5,
    mgr: Optional[Any] = None,
) -> None:
    """Build interactive HTML viewer.

    Parameters
    ----------
    adata
        AnnData with ``obs['icls']`` (set by hierarchy()), ``obsm['X_umap']``,
        and ``obs[level]`` for each level in ``br.hr.levels``.
    br
        :class:`sceleto.markers.BranchingResult`.
    save
        Output HTML file path.
    n_top
        Number of top markers to display per branch.
    mgr
        Optional :class:`sceleto.markers.graph.MarkerGraphRun`. When provided,
        the bottom-right panel becomes an edge-activation graph (per-gene,
        from ``mgr.viz``) instead of 3 resolution UMAPs.
    """
    if "X_umap" not in adata.obsm:
        raise ValueError("adata.obsm['X_umap'] required")
    if "icls" not in adata.obs.columns:
        raise ValueError("adata.obs['icls'] required (set by hierarchy())")

    hr = br.hr
    levels: List[str] = list(hr.levels)
    if len(levels) != 3:
        raise ValueError(f"viewer expects 3-level hierarchy; got {len(levels)}")

    umap_xy = adata.obsm["X_umap"]

    # ── Top section data (mirrors existing _viewer.py) ───────────────
    df = pd.DataFrame(
        {"umap_x": umap_xy[:, 0], "umap_y": umap_xy[:, 1],
         "icls": adata.obs["icls"].values},
        index=adata.obs.index,
    )
    icls_counts = df["icls"].value_counts().to_dict()
    icls_cell_counts = {str(k): int(v) for k, v in icls_counts.items()}
    marker_data = _build_compare_data(
        hr.icls_full_dict, hr.full_gene_lists, n_top, icls_cell_counts,
    )
    icls_colors = _assign_icls_colors(adata)
    umap_json = df[["umap_x", "umap_y", "icls"]].to_dict(orient="list")
    centroids = df.groupby("icls")[["umap_x", "umap_y"]].median().reset_index()
    centroids_json = centroids.to_dict(orient="list")

    # ── Bottom section data: 3 resolution UMAPs ──────────────────────
    n_cells = umap_xy.shape[0]
    if n_cells > 30000:
        rng = np.random.default_rng(0)
        sub_idx = np.sort(rng.choice(n_cells, size=30000, replace=False))
    else:
        sub_idx = np.arange(n_cells)

    bottom_umap = {
        "umap_x": umap_xy[sub_idx, 0].astype(float).round(3).tolist(),
        "umap_y": umap_xy[sub_idx, 1].astype(float).round(3).tolist(),
    }

    bottom_cluster: Dict[str, List[str]] = {}
    bottom_centroids: Dict[str, Dict[str, List]] = {}
    for lvl in levels:
        if lvl not in adata.obs.columns:
            raise ValueError(f"adata.obs missing {lvl!r}")
        col = adata.obs[lvl].astype(str).values
        bottom_cluster[lvl] = [str(c) for c in col[sub_idx]]

        df_c = pd.DataFrame({
            "x": umap_xy[:, 0],
            "y": umap_xy[:, 1],
            "c": col.astype(str),
        })
        med = df_c.groupby("c", observed=True)[["x", "y"]].median()
        try:
            med = med.reindex(sorted(med.index, key=lambda s: float(s)))
        except (TypeError, ValueError):
            med = med.sort_index()
        bottom_centroids[lvl] = {
            "labels": med.index.astype(str).tolist(),
            "x":      med["x"].round(2).tolist(),
            "y":      med["y"].round(2).tolist(),
        }

    # icls_id → [cluster_l0, cluster_l1, cluster_l2]
    icls_to_clusters: Dict[str, List[str]] = {}
    for icls_id, path_str in hr.icls_full_dict.items():
        parts = path_str.split("|")
        cs: List[str] = []
        for p in parts:
            at = p.find("@")
            cs.append(p[at+1:] if at >= 0 else p)
        icls_to_clusters[str(icls_id)] = cs

    # Branching markers per branch (top-N gene names only)
    branching_markers_json: Dict[str, List[str]] = {}
    for branch, marker_list in br.markers.items():
        branching_markers_json[branch] = [g for g, _, _, _ in marker_list[:n_top]]

    # ── Edge graph data (only if mgr provided) ───────────────────────
    edge_data = None
    if mgr is not None:
        edge_data = _build_edge_data(adata, mgr, marker_data)

    # ── Compose HTML ─────────────────────────────────────────────────
    data_blob = {
        # Top section
        "umap":             umap_json,
        "marker_data":      marker_data,
        "icls_colors":      icls_colors,
        "centroids":        centroids_json,
        # Bottom section
        "levels":            levels,
        "bottom_umap":       bottom_umap,
        "bottom_cluster":    bottom_cluster,
        "bottom_centroids":  bottom_centroids,
        "icls_to_clusters":  icls_to_clusters,
        "branching_markers": branching_markers_json,
        # Edge-activation panel (None if mgr not provided)
        "edge_data":         edge_data,
        # Meta
        "n_top": int(n_top),
    }

    html = _HTML_TEMPLATE.replace(
        "/*__DATA__*/",
        json.dumps(data_blob, separators=(",", ":")),
    )
    Path(save).write_text(html, encoding="utf-8")


def _build_edge_data(adata: Any, mgr: Any, marker_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract edge-activation data from MarkerGraphRun for JS rendering.

    Node positions are computed as UMAP centroids of ``adata.obs[mgr.groupby]``
    (not mgr.viz.pos_dict / paga pos) so the edge graph aligns with the icls
    UMAP shown on the left.

    Filters mean_mat / gene_edge_fc to the union of marker genes that may
    appear in any heatmap (the marker_data x-axes), to keep payload small.
    """
    viz = mgr.viz
    if viz.mean_mat is None or viz.gene_edge_fc is None:
        raise ValueError("mgr.viz must have mean_mat and gene_edge_fc set.")

    nodes = list(viz.G.nodes())
    node_ids = [str(n) for n in nodes]
    node_idx = {n: i for i, n in enumerate(nodes)}
    node_sizes = [float(s) for s in viz.node_sizes]

    # Node positions = UMAP centroids of mgr.groupby (icls etc.), so the
    # graph layout matches the icls UMAP exactly.
    if "X_umap" not in adata.obsm:
        raise ValueError("adata.obsm['X_umap'] required for edge graph layout")
    groupby = mgr.groupby
    if groupby not in adata.obs.columns:
        raise ValueError(f"adata.obs[{groupby!r}] required for edge graph layout")
    umap_xy = adata.obsm["X_umap"]
    obs_g = adata.obs[groupby].astype(str).values
    cent_df = pd.DataFrame(
        {"x": umap_xy[:, 0], "y": umap_xy[:, 1], "g": obs_g}
    ).groupby("g", observed=True)[["x", "y"]].median()
    pos: List[List[float]] = []
    for n in nodes:
        key = str(n)
        if key not in cent_df.index:
            raise ValueError(f"group {key!r} (mgr node) not found in adata.obs[{groupby!r}]")
        pos.append([float(cent_df.loc[key, "x"]), float(cent_df.loc[key, "y"])])
    edges = [
        [node_idx[u], node_idx[v]] for (u, v) in viz.G.edges()
    ]
    edge_pair_to_idx = {(u, v): i for i, (u, v) in enumerate(viz.G.edges())}

    # Marker gene union across all clusters' heatmaps
    all_marker_genes: set = set()
    for d in marker_data.values():
        all_marker_genes.update(d.get("genes", []))

    mm_cols = set(viz.mean_mat.columns)
    genes = [g for g in all_marker_genes if g in mm_cols]

    # mean_mat (gene → per-node value)
    gene_mean: Dict[str, List[float]] = {}
    for g in genes:
        col = viz.mean_mat[g]
        vals = []
        for n in nodes:
            v = col.get(n, 0.0)
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = 0.0
            vals.append(round(v, 3))
        gene_mean[g] = vals

    # gene_edge_fc (gene → [[edge_idx, fc], ...])
    gene_edge_fc_out: Dict[str, List[List[float]]] = {}
    for g in genes:
        fc_map = viz.gene_edge_fc.get(g, {})
        items: List[List[float]] = []
        for (u, v), fc in fc_map.items():
            ei = edge_pair_to_idx.get((u, v))
            if ei is None:
                continue
            try:
                items.append([ei, round(float(fc), 3)])
            except (TypeError, ValueError):
                continue
        if items:
            gene_edge_fc_out[g] = items

    # FC bin defaults match plot_gene_edges_fc
    fc_bins = [0.0, 3.0, 4.0, 5.0, 6.0, float("inf")]
    fc_colors = ["lightgrey", "#9fdab8", "#57b8d0", "#1d7eb7", "#084081"]
    # JSON has no Infinity; sentinel handled in JS
    fc_bins_js = [b if np.isfinite(b) else None for b in fc_bins]

    # UMAP range (with small padding) so icls UMAP and edge graph share axes
    pad_x = (float(umap_xy[:, 0].max()) - float(umap_xy[:, 0].min())) * 0.03
    pad_y = (float(umap_xy[:, 1].max()) - float(umap_xy[:, 1].min())) * 0.03
    umap_range = {
        "x": [float(umap_xy[:, 0].min()) - pad_x, float(umap_xy[:, 0].max()) + pad_x],
        "y": [float(umap_xy[:, 1].min()) - pad_y, float(umap_xy[:, 1].max()) + pad_y],
    }

    return {
        "node_ids":     node_ids,
        "node_pos":     pos,
        "node_sizes":   node_sizes,
        "edges":        edges,
        "gene_mean":    gene_mean,
        "gene_edge_fc": gene_edge_fc_out,
        "fc_bins":      fc_bins_js,
        "fc_colors":    fc_colors,
        "bg_color":     "lightgrey",
        "bg_alpha":     0.25,
        "umap_range":   umap_range,
    }


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>sceleto Branching Marker Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #f5f5f5; color: #222; }
  #app {
    display: grid;
    grid-template-columns: 540px 1fr;
    grid-template-rows: 1fr 1fr;
    height: 100vh;
    gap: 4px;
    padding: 4px;
  }
  #panel-icls   { grid-column: 1; grid-row: 1 / span 2; background: white; border: 1px solid #ddd; padding: 8px;  overflow: hidden;
                  display: flex; align-items: center; justify-content: center; }
  #panel-marker { grid-column: 2; grid-row: 1;          background: white; border: 1px solid #ddd; padding: 12px; overflow: auto; }
  #panel-cross  { grid-column: 2; grid-row: 2;          background: white; border: 1px solid #ddd; padding: 4px;  overflow: auto;
                  display: flex; flex-direction: column; outline: none; }

  /* Fixed plot sizes — do not resize on window changes. */
  #icls-umap   { width: 520px; height: 520px; }
  #cross-umaps { display: flex; gap: 2px; }
  .cross-cell  { width: 320px; height: 320px; flex: 0 0 320px; }

  /* Edge-activation panel */
  #gene-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
    padding: 4px 6px;
    max-width: 540px;
    flex: 0 0 auto;
    border-bottom: 1px solid #eee;
  }
  .gene-chip {
    padding: 2px 6px; font-size: 10px;
    background: #f0f0f0; border: 1px solid #ccc; border-radius: 8px;
    cursor: pointer; white-space: nowrap; user-select: none;
    text-align: center;
  }
  .gene-chip:hover { background: #e0e0e0; }
  .gene-chip.selected { background: #3182bd; color: white; border-color: #1d6ea3; }
  #edge-graph { width: 520px; height: 520px; flex: 0 0 520px; }
  .edge-panel-hint { font-size: 10px; color: #999; padding: 2px 6px; }
  th.gene-th { cursor: pointer; }
  th.gene-th:hover { background: #d8e8f5; }
  th.gene-th.selected { background: #3182bd; color: white; }

  h2 { font-size: 13px; color: #555; margin-bottom: 6px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
  .placeholder { color: #aaa; font-size: 12px; text-align: center; margin-top: 30px; }

  table.heatmap { border-collapse: collapse; font-size: 11px; }
  table.heatmap th, table.heatmap td { text-align: center; border: 1px solid #ccc; padding: 3px 0; }
  table.heatmap th:first-child, table.heatmap td:first-child { padding: 3px 8px; white-space: nowrap; }
  table.heatmap th { background: #f0f0f0; position: sticky; top: 0; }
  td.present { background: #3182bd; color: white; }
  td.absent  { background: #f7f7f7; color: #bbb; }
  .legend { font-size: 11px; margin-top: 8px; color: #666; }
  .legend span { display: inline-block; width: 14px; height: 14px; vertical-align: middle; margin-right: 4px; border: 1px solid #ccc; }
  .info { font-size: 12px; color: #666; margin-bottom: 12px; }
</style>
</head>
<body>
<div id="app">
  <div id="panel-icls"><div id="icls-umap"></div></div>
  <div id="panel-marker">
    <h2>Marker Comparison</h2>
    <div class="placeholder" id="marker-placeholder">Click a path on the UMAP.</div>
    <div id="marker-content" style="display:none;"></div>
  </div>
  <div id="panel-cross" tabindex="0">
    <!-- mgr=None mode -->
    <div id="cross-umaps">
      <div id="cross-0" class="cross-cell"></div>
      <div id="cross-1" class="cross-cell"></div>
      <div id="cross-2" class="cross-cell"></div>
    </div>
    <!-- mgr=MarkerGraphRun mode -->
    <div id="gene-chips" style="display:none;"></div>
    <div id="edge-graph" style="display:none;"></div>
    <div class="edge-panel-hint" id="edge-hint" style="display:none;">click cell · click chip · wheel · ← →</div>
  </div>
</div>

<script>
const DATA = /*__DATA__*/;
const LEVELS = DATA.levels;
const N_TOP = DATA.n_top;

// ── Top: icls UMAP ────────────────────────────────────────────────
const ICLS_GROUPS = {};
const ICLS_SORTED = (function() {
  const u = DATA.umap;
  for (let i = 0; i < u.icls.length; i++) {
    const id = u.icls[i];
    if (!ICLS_GROUPS[id]) ICLS_GROUPS[id] = { x: [], y: [] };
    ICLS_GROUPS[id].x.push(u.umap_x[i]);
    ICLS_GROUPS[id].y.push(u.umap_y[i]);
  }
  return [...new Set(u.icls)].sort((a,b) => parseInt(a) - parseInt(b));
})();

function buildIclsTraces(highlight) {
  const traces = ICLS_SORTED.map(id => ({
    x: ICLS_GROUPS[id].x, y: ICLS_GROUPS[id].y,
    mode: "markers", type: "scattergl",
    marker: {
      color: DATA.icls_colors[id],
      size:    highlight === id ? 4   : 2,
      opacity: highlight == null ? 0.5 : (highlight === id ? 0.9 : 0.1),
    },
    hoverinfo: "skip", showlegend: false,
  }));
  // Invisible clickable hit-box at each centroid
  const cent = DATA.centroids;
  traces.push({
    x: cent.umap_x, y: cent.umap_y,
    mode: "markers", type: "scattergl",
    marker: { size: 22, color: "rgba(0,0,0,0)", line: { width: 0 } },
    customdata: cent.icls,
    hoverinfo: "text",
    hovertext: cent.icls.map(id => "path " + id),
    showlegend: false,
  });
  return traces;
}

const ICLS_LAYOUT = (function() {
  const cent = DATA.centroids;
  const outline = [[-0.6,0],[0.6,0],[0,-0.6],[0,0.6],[-0.4,-0.4],[0.4,-0.4],[-0.4,0.4],[0.4,0.4]];
  const annotations = [];
  for (let i = 0; i < cent.icls.length; i++) {
    const id = cent.icls[i];
    const cx = cent.umap_x[i], cy = cent.umap_y[i];
    for (const [dx, dy] of outline) {
      annotations.push({
        x: cx, y: cy, text: "<b>" + id + "</b>",
        showarrow: false, xshift: dx, yshift: dy,
        font: { size: 11, color: "#fff", family: "Arial, sans-serif" },
      });
    }
    annotations.push({
      x: cx, y: cy, text: "<b>" + id + "</b>",
      showarrow: false,
      font: { size: 11, color: "#000", family: "Arial, sans-serif" },
    });
  }
  return {
    title: { text: "path UMAP — click a number", font: { size: 13 } },
    xaxis: { title: "UMAP1", zeroline: false, showticklabels: false, showgrid: false },
    yaxis: { title: "UMAP2", zeroline: false, showticklabels: false, showgrid: false },
    showlegend: false,
    margin: { l: 30, r: 10, t: 30, b: 30, autoexpand: false },
    hovermode: "closest",
    annotations: annotations,
    plot_bgcolor: "white", paper_bgcolor: "white",
  };
})();

// Mode flag: edge-activation panel vs 3-UMAP panel.
const HAS_EDGE = (DATA.edge_data != null);

// Fixed-size plot — no resize on window changes; CSS sets 520x520.
let SELECTED_ICLS = null;
let SORTED_GENES = [];      // current heatmap x-axis order
let SELECTED_GENE = null;
Plotly.newPlot("icls-umap", buildIclsTraces(null), ICLS_LAYOUT, { responsive: false, displayModeBar: false });
document.getElementById("icls-umap").on("plotly_click", function(data) {
  if (!data || !data.points || !data.points.length) return;
  const pt = data.points[0];
  const icls = pt.customdata;
  if (icls == null || !DATA.marker_data[icls]) return;
  // Toggle off if same icls clicked
  SELECTED_ICLS = SELECTED_ICLS === icls ? null : icls;
  Plotly.react("icls-umap", buildIclsTraces(SELECTED_ICLS), ICLS_LAYOUT);
  onIclsSelected(SELECTED_ICLS);
});

// ── Top-right: Marker Comparison heatmap (mirrors _viewer.py) ─────
function renderMarkerComparison(icls) {
  const d = DATA.marker_data[icls];
  const placeholder = document.getElementById("marker-placeholder");
  const panel = document.getElementById("marker-content");
  placeholder.style.display = "none";
  panel.style.display = "block";

  const sortedGenes = [...d.genes].sort((a, b) => {
    for (let lv = 0; lv < d.levels.length; lv++) {
      const diff = d.presence[b][lv] - d.presence[a][lv];
      if (diff !== 0) return diff;
    }
    return a.localeCompare(b);
  });
  SORTED_GENES = sortedGenes;

  const cellW = 28, labelColW = 120;
  const tableW = labelColW + sortedGenes.length * cellW;

  let html = '<div class="info"><b>path ' + icls + '</b> &nbsp; (' + d.n_cells + ' cells)<br>'
           + d.levels.map(l => '<code>' + l + '</code>').join(' &rarr; ') + '</div>';
  html += '<table class="heatmap" style="width:' + tableW + 'px;"><thead><tr><th style="width:' + labelColW + 'px;"></th>';
  const thCls = HAS_EDGE ? ' class="gene-th"' : '';
  for (const g of sortedGenes) {
    html += '<th' + thCls + ' data-gene="' + g + '" style="writing-mode:vertical-rl; transform:rotate(180deg);">' + g + '</th>';
  }
  html += '</tr></thead><tbody>';
  for (let i = 0; i < d.levels.length; i++) {
    html += '<tr><td style="text-align:left; white-space:nowrap; font-weight:bold;">' + d.levels[i] + '</td>';
    for (const g of sortedGenes) {
      const v = d.presence[g][i];
      html += '<td class="' + (v ? 'present' : 'absent') + '"></td>';
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  html += '<div class="legend">'
        + '<span style="background:#3182bd;"></span> in top-' + N_TOP + ' &nbsp;'
        + '<span style="background:#f7f7f7;"></span> not in top-' + N_TOP
        + '</div>';
  panel.innerHTML = html;
  if (HAS_EDGE) {
    panel.querySelectorAll("th.gene-th").forEach(el => {
      el.addEventListener("click", () => setSelectedGene(el.dataset.gene));
    });
  }
}

// ── Bottom: 3 small UMAPs with marker annotations ─────────────────
function childrenAt(level, parentPath) {
  // parentPath: array of cluster names at levels [0..level-1]
  // returns: array of unique cluster names at the requested level
  const out = new Set();
  for (const [icls, cs] of Object.entries(DATA.icls_to_clusters)) {
    let ok = true;
    for (let i = 0; i < parentPath.length; i++) {
      if (cs[i] !== parentPath[i]) { ok = false; break; }
    }
    if (ok && cs[level] != null) out.add(cs[level]);
  }
  return Array.from(out);
}

const CROSS_COLORS = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#ffbf00",
                      "#a65628","#f781bf","#1b9e77","#d95f02","#7570b3","#e7298a",
                      "#66a61e","#e6ab02","#a6761d","#999999"];
function clusterColor(idx) { return CROSS_COLORS[idx % CROSS_COLORS.length]; }

function plotCrossUmap(divId, levelIdx, highlightSet, markerAnnotations) {
  // highlightSet:  Set<string> cluster names to highlight (or null = no highlight)
  // markerAnnotations: { cluster_name: [gene1, gene2, ...] }
  const lvl = LEVELS[levelIdx];
  const clusterArr = DATA.bottom_cluster[lvl];
  const ux = DATA.bottom_umap.umap_x;
  const uy = DATA.bottom_umap.umap_y;
  const n = ux.length;

  const uniq = Array.from(new Set(clusterArr));
  uniq.sort((a,b) => {
    const na = Number(a), nb = Number(b);
    if (!isNaN(na) && !isNaN(nb)) return na - nb;
    return a < b ? -1 : 1;
  });

  const hasHighlight = highlightSet && highlightSet.size > 0;
  const traces = [];
  for (let i = 0; i < uniq.length; i++) {
    const c = uniq[i];
    const xs = [], ys = [];
    for (let k = 0; k < n; k++) {
      if (clusterArr[k] === c) { xs.push(ux[k]); ys.push(uy[k]); }
    }
    let color, opacity, size;
    if (!hasHighlight) {
      color = clusterColor(i); opacity = 0.6; size = 2;
    } else if (highlightSet.has(c)) {
      color = clusterColor(i); opacity = 0.9; size = 3;
    } else {
      color = "#dddddd"; opacity = 0.18; size = 1.5;
    }
    traces.push({
      x: xs, y: ys, mode: "markers", type: "scattergl",
      name: c,
      marker: { color: color, size: size, opacity: opacity, line: { width: 0 } },
      hoverinfo: "skip", showlegend: false,
    });
  }

  // Cluster number labels + marker annotation text (always on top)
  const cent = DATA.bottom_centroids[lvl];
  const annotations = [];
  const numOutline = [[-0.4,0],[0.4,0],[0,-0.4],[0,0.4]];

  // Directional offsets for marker annotation text. xanchor/yanchor pin the
  // box edge to the offset point so the box is pushed away from the cluster.
  const R = 45;   // cardinal distance
  const D = 35;   // diagonal distance (~ R/√2)
  // Index order: 0=N, 1=E, 2=S, 3=W, 4=NE, 5=SE, 6=SW, 7=NW
  const POSITIONS = [
    { ax:  0, ay: -R, xanchor: "center", yanchor: "bottom" },  // N
    { ax:  R, ay:  0, xanchor: "left",   yanchor: "middle" },  // E
    { ax:  0, ay:  R, xanchor: "center", yanchor: "top"    },  // S
    { ax: -R, ay:  0, xanchor: "right",  yanchor: "middle" },  // W
    { ax:  D, ay: -D, xanchor: "left",   yanchor: "bottom" },  // NE
    { ax:  D, ay:  D, xanchor: "left",   yanchor: "top"    },  // SE
    { ax: -D, ay:  D, xanchor: "right",  yanchor: "top"    },  // SW
    { ax: -D, ay: -D, xanchor: "right",  yanchor: "bottom" },  // NW
  ];
  // Diagonal target angles in data coords (CCW from +x). +y is up.
  const DIAG_ANGLES = {
    4:  Math.PI / 4,        // NE  (+x +y)
    5: -Math.PI / 4,        // SE  (+x -y)
    6: -3 * Math.PI / 4,    // SW  (-x -y)
    7:  3 * Math.PI / 4,    // NW  (-x +y)
  };

  // ── Build annotated cluster list ─────────────────────────────────
  const annotated = [];
  for (let i = 0; i < cent.labels.length; i++) {
    const lbl = cent.labels[i];
    if (markerAnnotations && markerAnnotations[lbl] && markerAnnotations[lbl].length) {
      annotated.push({ lbl, idx: i, cx: cent.x[i], cy: cent.y[i] });
    }
  }
  function angDiff(a, b) {
    let d = Math.abs(a - b);
    if (d > Math.PI) d = 2 * Math.PI - d;
    return d;
  }

  // ── Assign positions: cardinals by axis-extreme, diagonals by angle ──
  const posByLabel = {};
  if (annotated.length > 0) {
    const remaining = annotated.slice();

    // pick(extractor, compare) → splice the cluster maximizing `extractor`
    // (compare reverses for min). Returns the picked cluster or null.
    function pickExtreme(scoreFn) {
      if (remaining.length === 0) return null;
      let bestI = 0, bestS = scoreFn(remaining[0]);
      for (let i = 1; i < remaining.length; i++) {
        const s = scoreFn(remaining[i]);
        if (s > bestS) { bestS = s; bestI = i; }
      }
      return remaining.splice(bestI, 1)[0];
    }

    // W = smallest x (use −cx as score)
    let pick = pickExtreme(a => -a.cx); if (pick) posByLabel[pick.lbl] = 3;
    // E = largest x
    pick = pickExtreme(a =>  a.cx);    if (pick) posByLabel[pick.lbl] = 1;
    // N = largest y
    pick = pickExtreme(a =>  a.cy);    if (pick) posByLabel[pick.lbl] = 0;
    // S = smallest y
    pick = pickExtreme(a => -a.cy);    if (pick) posByLabel[pick.lbl] = 2;

    // Diagonals: pick by angle from remaining group center
    if (remaining.length > 0) {
      let gcx = 0, gcy = 0;
      for (const a of remaining) { gcx += a.cx; gcy += a.cy; }
      gcx /= remaining.length; gcy /= remaining.length;
      for (const a of remaining) {
        a.theta = Math.atan2(a.cy - gcy, a.cx - gcx);
      }
      const diagOrder = [7, 4, 5, 6];  // NW, NE, SE, SW
      for (const pIdx of diagOrder) {
        if (remaining.length === 0) break;
        const target = DIAG_ANGLES[pIdx];
        let bestI = 0, bestDiff = angDiff(remaining[0].theta, target);
        for (let i = 1; i < remaining.length; i++) {
          const d = angDiff(remaining[i].theta, target);
          if (d < bestDiff) { bestDiff = d; bestI = i; }
        }
        posByLabel[remaining[bestI].lbl] = pIdx;
        remaining.splice(bestI, 1);
      }
      // Anything left (>8 clusters): cycle through 0..7
      let fallback = 0;
      for (const a of remaining) {
        posByLabel[a.lbl] = fallback++ % POSITIONS.length;
      }
    }
  }
  const POS_BY_LABEL = posByLabel;

  // Pass 1: cluster number labels (drawn first → lower layer)
  for (let i = 0; i < cent.labels.length; i++) {
    const lbl = cent.labels[i];
    const cx = cent.x[i], cy = cent.y[i];
    const dimmed = hasHighlight && !highlightSet.has(lbl);
    const numColor = dimmed ? "#bbb" : "#000";

    for (const [dx, dy] of numOutline) {
      annotations.push({
        x: cx, y: cy, text: "<b>" + lbl + "</b>",
        showarrow: false, xshift: dx, yshift: dy,
        font: { size: 10, color: "#fff", family: "Arial, sans-serif" },
      });
    }
    annotations.push({
      x: cx, y: cy, text: "<b>" + lbl + "</b>",
      showarrow: false,
      font: { size: 10, color: numColor, family: "Arial, sans-serif" },
    });
  }

  // Pass 2: marker annotation boxes (drawn last → top layer, never blocked by
  // background cluster numbers from other clusters)
  for (let i = 0; i < cent.labels.length; i++) {
    const lbl = cent.labels[i];
    if (!markerAnnotations || !markerAnnotations[lbl] || !markerAnnotations[lbl].length) continue;
    const cx = cent.x[i], cy = cent.y[i];
    const genes = markerAnnotations[lbl];
    const pIdx = POS_BY_LABEL[lbl] ?? 0;
    const pos = POSITIONS[pIdx];

    annotations.push({
      x: cx, y: cy,
      ax: pos.ax, ay: pos.ay,
      xanchor: pos.xanchor, yanchor: pos.yanchor,
      text: genes.join("<br>"),
      showarrow: true, arrowhead: 2, arrowsize: 0.7, arrowwidth: 1,
      arrowcolor: "#666",
      bgcolor: "rgba(255,255,255,0.95)",
      bordercolor: "#999", borderwidth: 0.5, borderpad: 2,
      font: { size: 9, color: "#222", family: "Arial, sans-serif" },
      align: "left",
    });
  }

  const layout = {
    title: { text: lvl, font: { size: 11, color: "#666" } },
    xaxis: { showticklabels: false, showgrid: false, zeroline: false, ticks: "" },
    yaxis: { showticklabels: false, showgrid: false, zeroline: false, ticks: "" },
    showlegend: false,
    // Small margins → larger data area. Annotation boxes that fall outside
    // the SVG bounds get clipped (acceptable trade-off for max UMAP size).
    // autoexpand:false locks the data area size on selection changes.
    margin: { l: 8, r: 8, t: 22, b: 8, autoexpand: false },
    hovermode: false,
    annotations: annotations,
    plot_bgcolor: "white", paper_bgcolor: "white",
  };
  Plotly.newPlot(divId, traces, layout, { responsive: false, displayModeBar: false, staticPlot: false });
}

function plotAllCross(highlightSets, annotationsByLevel) {
  for (let i = 0; i < 3; i++) {
    plotCrossUmap(
      "cross-" + i, i,
      highlightSets ? highlightSets[i] : null,
      annotationsByLevel ? annotationsByLevel[i] : null,
    );
  }
}

// ── Initial state ─────────────────────────────────────────────────
if (HAS_EDGE) {
  // Hide 3-UMAP mode; show edge-graph panel
  document.getElementById("cross-umaps").style.display = "none";
  document.getElementById("gene-chips").style.display = "flex";
  document.getElementById("edge-graph").style.display = "block";
  document.getElementById("edge-hint").style.display = "block";
} else {
  plotAllCross(null, null);
}

// ── Edge-activation panel ─────────────────────────────────────────
// fc_bins entries may be null (representing +Infinity from JSON).
function fcBinIdx(fc) {
  const bins = DATA.edge_data.fc_bins;
  const nBins = bins.length - 1;
  for (let i = 0; i < nBins; i++) {
    const lo = bins[i] == null ? -Infinity : bins[i];
    const hi = bins[i + 1] == null ? Infinity : bins[i + 1];
    if (fc >= lo && fc < hi) return i;
  }
  return nBins - 1;
}

function renderGeneChips() {
  const div = document.getElementById("gene-chips");
  if (!SORTED_GENES.length) { div.innerHTML = ""; return; }
  const ed = DATA.edge_data;
  div.innerHTML = SORTED_GENES.map(g => {
    const has = (g in ed.gene_mean);
    const sel = g === SELECTED_GENE ? " selected" : "";
    const dim = has ? "" : " style=\"opacity:0.4;\"";
    return '<span class="gene-chip' + sel + '" data-gene="' + g + '"' + dim + '>' + g + '</span>';
  }).join("");
  div.querySelectorAll(".gene-chip").forEach(el => {
    el.addEventListener("click", () => setSelectedGene(el.dataset.gene));
  });
}

function highlightGeneSelection() {
  document.querySelectorAll("#gene-chips .gene-chip").forEach(el => {
    el.classList.toggle("selected", el.dataset.gene === SELECTED_GENE);
  });
  document.querySelectorAll("th.gene-th").forEach(el => {
    el.classList.toggle("selected", el.dataset.gene === SELECTED_GENE);
  });
  const chip = document.querySelector("#gene-chips .gene-chip.selected");
  if (chip) chip.scrollIntoView({ block: "nearest", inline: "center" });
  const th = document.querySelector("th.gene-th.selected");
  if (th) th.scrollIntoView({ block: "nearest", inline: "center" });
}

function renderEdgeGraph(gene) {
  const ed = DATA.edge_data;
  const div = document.getElementById("edge-graph");
  const vals = ed.gene_mean[gene] || ed.node_ids.map(() => 0);
  let vmin = Infinity, vmax = -Infinity;
  for (const v of vals) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
  if (!isFinite(vmin) || vmin === vmax) { vmin = 0; vmax = Math.max(1, vmax); }

  const fcMap = new Map();
  for (const [ei, fc] of (ed.gene_edge_fc[gene] || [])) fcMap.set(ei, fc);

  const nColors = ed.fc_colors.length;

  // Nodes: marker size scaled from node_sizes (matplotlib s ~ area; rough sqrt scaling)
  const nodeSize = ed.node_sizes.map(s => Math.max(14, Math.sqrt(s) * 1.4));
  const traces = [{
    x: ed.node_pos.map(p => p[0]),
    y: ed.node_pos.map(p => p[1]),
    mode: "markers", type: "scatter",
    marker: {
      color: vals,
      // Explicit white→black (matplotlib Greys convention; Plotly's named
      // "Greys" is the opposite, so we hard-code stops).
      colorscale: [[0, "rgb(255,255,255)"], [1, "rgb(0,0,0)"]],
      cmin: vmin, cmax: vmax,
      size: nodeSize,
      line: { color: "#000", width: 0.6 },
    },
    text: ed.node_ids,
    hovertemplate: "node %{text}<br>expr=%{marker.color:.2f}<extra></extra>",
    showlegend: false,
  }];

  // Edges as arrow annotations (one per edge). Colored by FC bin or grey background.
  const annotations = [];
  for (let i = 0; i < ed.edges.length; i++) {
    const [u, v] = ed.edges[i];
    const has = fcMap.has(i);
    const color = has ? ed.fc_colors[fcBinIdx(fcMap.get(i))] : ed.bg_color;
    const opacity = has ? 1.0 : ed.bg_alpha;
    annotations.push({
      x: ed.node_pos[v][0], y: ed.node_pos[v][1],   // arrowhead at target
      ax: ed.node_pos[u][0], ay: ed.node_pos[u][1], // tail at source
      xref: "x", yref: "y", axref: "x", ayref: "y",
      showarrow: true, arrowhead: 2, arrowsize: 1.0,
      arrowwidth: has ? 2.2 : 1.5,
      arrowcolor: color, opacity: opacity,
      standoff: 10, startstandoff: 8,
      text: "",
    });
  }

  // Node labels: white outline (8 directions) + black center for contrast
  const numOutline = [[-0.6,0],[0.6,0],[0,-0.6],[0,0.6],
                      [-0.45,-0.45],[0.45,-0.45],[-0.45,0.45],[0.45,0.45]];
  for (let i = 0; i < ed.node_ids.length; i++) {
    const cx = ed.node_pos[i][0], cy = ed.node_pos[i][1];
    const lbl = ed.node_ids[i];
    for (const [dx, dy] of numOutline) {
      annotations.push({
        x: cx, y: cy, text: "<b>" + lbl + "</b>",
        showarrow: false, xshift: dx, yshift: dy,
        font: { size: 10, color: "#fff", family: "Arial, sans-serif" },
      });
    }
    annotations.push({
      x: cx, y: cy, text: "<b>" + lbl + "</b>",
      showarrow: false,
      font: { size: 10, color: "#000", family: "Arial, sans-serif" },
    });
  }

  // FC legend
  const bins = ed.fc_bins;
  for (let b = 0; b < nColors; b++) {
    const lo = bins[b], hi = bins[b + 1];
    const label = (hi == null) ? ("≥ " + lo) : (lo + "–" + hi);
    annotations.push({
      xref: "paper", yref: "paper",
      x: 1.01, y: 0.95 - b * 0.06,
      text: '<span style="color:' + ed.fc_colors[b] + ';">■</span> ' + label,
      showarrow: false, xanchor: "left",
      font: { size: 10, color: "#333" },
    });
  }

  // Lock edge graph axes to UMAP data range so it aligns with the icls UMAP
  // (which autoranges to the same cells; their min/max match within padding).
  const xRange = DATA.edge_data.umap_range.x.slice();
  const yRange = DATA.edge_data.umap_range.y.slice();
  const layout = {
    title: { text: "gene: " + gene, font: { size: 12 } },
    xaxis: { showticklabels: false, showgrid: false, zeroline: false, ticks: "", range: xRange },
    yaxis: { showticklabels: false, showgrid: false, zeroline: false, ticks: "", range: yRange },
    showlegend: false,
    margin: { l: 8, r: 90, t: 26, b: 8, autoexpand: false },
    hovermode: "closest",
    annotations: annotations,
    plot_bgcolor: "white", paper_bgcolor: "white",
  };
  Plotly.react(div, traces, layout, { responsive: false, displayModeBar: false });
}

function setSelectedGene(gene) {
  if (!gene) return;
  SELECTED_GENE = gene;
  renderEdgeGraph(gene);
  highlightGeneSelection();
}

function stepGene(delta) {
  if (!SORTED_GENES.length) return;
  let idx = SORTED_GENES.indexOf(SELECTED_GENE);
  if (idx < 0) idx = 0;
  const next = idx + delta;
  if (next < 0 || next >= SORTED_GENES.length) return;  // clamp at endpoints
  setSelectedGene(SORTED_GENES[next]);
}

function clearEdgePanel() {
  SORTED_GENES = [];
  SELECTED_GENE = null;
  document.getElementById("gene-chips").innerHTML = "";
  Plotly.purge("edge-graph");
}

if (HAS_EDGE) {
  const xPanel = document.getElementById("panel-cross");
  xPanel.addEventListener("wheel", function(e) {
    if (!SORTED_GENES.length) return;
    e.preventDefault();
    stepGene(e.deltaY > 0 ? 1 : -1);
  }, { passive: false });
  xPanel.addEventListener("mouseenter", () => xPanel.focus());
  document.addEventListener("keydown", function(e) {
    if (!SORTED_GENES.length) return;
    if (e.key === "ArrowRight") { stepGene(1); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { stepGene(-1); e.preventDefault(); }
  });
}

// ── Click handler: update marker panel + bottom panel ─────────────
function onIclsSelected(iclsId) {
  if (iclsId == null) {
    document.getElementById("marker-placeholder").style.display = "block";
    document.getElementById("marker-content").style.display = "none";
    if (HAS_EDGE) clearEdgePanel();
    else plotAllCross(null, null);
    return;
  }
  renderMarkerComparison(iclsId);

  if (HAS_EDGE) {
    // Reset to first gene per phase 11 design (option a)
    renderGeneChips();
    if (SORTED_GENES.length) {
      // Prefer a gene that has edge-fc data; fall back to first
      const ed = DATA.edge_data;
      let first = SORTED_GENES.find(g => g in ed.gene_mean) || SORTED_GENES[0];
      setSelectedGene(first);
    }
    return;
  }

  const cs = DATA.icls_to_clusters[String(iclsId)];
  if (!cs) return;
  const [c0, c1, c2] = cs;

  // Determine highlight sets per level
  // level 0 (leiden_1.0): just ancestor c0
  // level 1 (leiden_2.0): all l2 children under c0
  // level 2 (leiden_4.0): all l4 children under (c0, c1)
  const l1_children = childrenAt(1, [c0]);
  const l2_children = childrenAt(2, [c0, c1]);

  const highlightSets = [
    new Set([c0]),
    new Set(l1_children),
    new Set(l2_children),
  ];

  // Marker annotations per UMAP
  // level 0: ancestor's branching markers (single)
  // level 1: each l2 child's branching markers
  // level 2: each l4 child's branching markers
  const annLvl0 = {};
  const m0 = DATA.branching_markers[LEVELS[0] + "@" + c0];
  if (m0) annLvl0[c0] = m0;

  const annLvl1 = {};
  for (const c of l1_children) {
    const m = DATA.branching_markers[LEVELS[1] + "@" + c];
    if (m) annLvl1[c] = m;
  }

  const annLvl2 = {};
  for (const c of l2_children) {
    const m = DATA.branching_markers[LEVELS[2] + "@" + c];
    if (m) annLvl2[c] = m;
  }

  plotAllCross(highlightSets, [annLvl0, annLvl1, annLvl2]);
}
</script>
</body>
</html>
"""
