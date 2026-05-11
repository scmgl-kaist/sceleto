"""Interactive HTML viewer for BranchingResult.

Renders a hierarchical tree on the left, 3 UMAPs (one per resolution) in the
middle, and a marker detail panel on the right. Clicking a tree node:
- Highlights the corresponding cluster in the matching UMAP
- Populates the marker detail panel with branching markers + scores

Standalone single-file HTML output, no external assets besides Plotly CDN.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def build_branching_html(
    adata: Any,
    br: Any,  # BranchingResult — typed via duck typing to avoid circular import
    save: str,
    *,
    n_top: int = 5,
) -> None:
    """Build interactive HTML viewer for branching markers.

    Parameters
    ----------
    adata
        AnnData with ``obsm['X_umap']`` and ``obs[level]`` for each level in
        ``br.hr.levels``.
    br
        :class:`sceleto.markers.BranchingResult`.
    save
        Output HTML file path.
    n_top
        Number of top markers to display per branch.
    """
    if "X_umap" not in adata.obsm:
        raise ValueError("adata.obsm['X_umap'] required for viewer")

    levels: List[str] = list(br.hr.levels)
    if len(levels) != 3:
        raise ValueError(f"viewer expects 3-level hierarchy; got {len(levels)}")

    # ── Data extraction ──────────────────────────────────────────────
    umap_xy = adata.obsm["X_umap"]
    n_cells = umap_xy.shape[0]

    # Subsample for plotting speed if too large
    if n_cells > 30000:
        rng = np.random.default_rng(0)
        sub_idx = rng.choice(n_cells, size=30000, replace=False)
        sub_idx.sort()
    else:
        sub_idx = np.arange(n_cells)

    umap_x = umap_xy[sub_idx, 0].astype(float).round(3).tolist()
    umap_y = umap_xy[sub_idx, 1].astype(float).round(3).tolist()

    cluster_per_level: Dict[str, List[str]] = {}
    centroids_per_level: Dict[str, Dict[str, List]] = {}
    for lvl in levels:
        if lvl not in adata.obs.columns:
            raise ValueError(f"adata.obs missing {lvl!r}")
        col = adata.obs[lvl].astype(str).values
        cluster_per_level[lvl] = [str(c) for c in col[sub_idx]]

        # Centroid (median) per cluster at this level (use full data, not subsample)
        df_c = pd.DataFrame({
            "x": umap_xy[:, 0],
            "y": umap_xy[:, 1],
            "c": col.astype(str),
        })
        med = df_c.groupby("c", observed=True)[["x", "y"]].median()
        # Numeric sort if possible
        try:
            med = med.reindex(sorted(med.index, key=lambda s: float(s)))
        except (TypeError, ValueError):
            med = med.sort_index()
        centroids_per_level[lvl] = {
            "labels": med.index.astype(str).tolist(),
            "x":      med["x"].round(2).tolist(),
            "y":      med["y"].round(2).tolist(),
        }

    # ── Tree serialization ───────────────────────────────────────────
    tree_json = _serialize_tree(br.tree_root)

    # ── Markers JSON ─────────────────────────────────────────────────
    markers_json: Dict[str, List[Dict[str, Any]]] = {}
    for branch, marker_list in br.markers.items():
        markers_json[branch] = [
            {
                "gene": g,
                "score": float(s),
                "rank": int(r),
                "exclusivity": float(e),
            }
            for g, s, r, e in marker_list[:n_top]
        ]

    # ── Cluster-level mapping for UMAP highlight ─────────────────────
    levels_index = {lvl: i for i, lvl in enumerate(levels)}

    # icls_id → [cluster_at_l0, cluster_at_l1, cluster_at_l2]
    # path string format: "leiden_1.0@3|leiden_2.0@5|leiden_4.0@7" → ["3","5","7"]
    icls_to_clusters: Dict[str, List[str]] = {}
    for icls_id, path_str in br.hr.icls_full_dict.items():
        parts = path_str.split("|")
        clusters_at_levels: List[str] = []
        for p in parts:
            at = p.find("@")
            clusters_at_levels.append(p[at+1:] if at >= 0 else p)
        icls_to_clusters[str(icls_id)] = clusters_at_levels

    # ── Compose HTML ─────────────────────────────────────────────────
    data_blob = {
        "levels":           levels,
        "umap_x":           umap_x,
        "umap_y":           umap_y,
        "cluster":          {lvl: cluster_per_level[lvl] for lvl in levels},
        "centroids":        centroids_per_level,
        "tree":             tree_json,
        "markers":          markers_json,
        "params":           dict(br.params),
        "level_index":      levels_index,
        "icls_to_clusters": icls_to_clusters,
    }

    html = _HTML_TEMPLATE.replace(
        "/*__DATA__*/",
        json.dumps(data_blob, separators=(",", ":")),
    )

    with open(save, "w") as f:
        f.write(html)


def _serialize_tree(tree_root: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Recursively convert nested tree dict to JSON-friendly list."""
    out: List[Dict[str, Any]] = []
    for name, node in tree_root.items():
        out.append({
            "name":         name,
            "n_icls":       len(node["icls_indices"]),
            "icls_indices": [int(i) for i in node["icls_indices"]],
            "children":     _serialize_tree(node["children"]),
        })
    return out


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>sceleto Branching Marker Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #fafafa; color: #222; }
  #app { display: grid; grid-template-columns: 320px 1fr 340px; height: 100vh; }
  #tree-panel { padding: 12px; overflow-y: auto; border-right: 1px solid #ddd; background: white; }
  #main-panel { display: flex; flex-direction: column; padding: 8px; min-width: 0; }
  #umaps { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; flex: 1 1 auto; min-height: 0; }
  .umap-box { background: white; border: 1px solid #eee; min-height: 0; }
  #path-panel { padding: 12px; overflow-y: auto; border-left: 1px solid #ddd; background: white; font-size: 11px; }
  #paths .path-row { padding: 3px 4px; border-bottom: 1px solid #f0f0f0; font-family: 'Courier New', monospace; }
  #paths .path-row .icls-tag { display: inline-block; min-width: 40px; color: #888; }
  h2 { font-size: 13px; color: #555; margin-bottom: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
  h3 { font-size: 12px; color: #777; margin: 8px 0 4px; }
  .tree-node { cursor: pointer; padding: 2px 4px; border-radius: 3px; font-size: 12px; user-select: none; }
  .tree-node:hover { background: #f0f0f0; }
  .tree-node.selected { background: #ffe066; font-weight: 600; }
  .tree-node.has-markers::before { content: "● "; color: #3182bd; }
  .tree-children { margin-left: 14px; border-left: 1px dashed #ccc; padding-left: 8px; }
  table.markers { width: 100%; font-size: 11px; border-collapse: collapse; margin-top: 6px; }
  table.markers th { background: #f0f0f0; padding: 4px; text-align: left; border-bottom: 1px solid #ccc; }
  table.markers td { padding: 3px 4px; border-bottom: 1px solid #f0f0f0; }
  table.markers td.gene { font-family: 'Courier New', monospace; font-weight: 600; color: #3182bd; }
  .placeholder { color: #aaa; font-size: 12px; text-align: center; margin-top: 30px; }
  .info { font-size: 11px; color: #888; margin-bottom: 6px; }
  .level-label { display: inline-block; font-size: 10px; padding: 2px 5px; background: #eee; border-radius: 3px; margin-bottom: 4px; }
</style>
</head>
<body>
<div id="app">
  <div id="tree-panel">
    <h2>Hierarchy Tree</h2>
    <div class="info">● marker available · click to highlight</div>
    <div id="tree"></div>
  </div>
  <div id="main-panel">
    <div id="umaps">
      <div id="umap-0" class="umap-box"></div>
      <div id="umap-1" class="umap-box"></div>
      <div id="umap-2" class="umap-box"></div>
    </div>
  </div>
  <div id="path-panel">
    <h2>Paths</h2>
    <div id="paths"><div class="placeholder">Click a tree node or cluster number</div></div>
  </div>
</div>

<script>
const DATA = /*__DATA__*/;
const LEVELS = DATA.levels;
const COLORS = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#ffff33","#a65628","#f781bf",
                "#999999","#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854","#ffd92f","#e5c494",
                "#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666"];

function colorFor(idx) { return COLORS[idx % COLORS.length]; }

function parseBranch(branchName) {
  // "leiden_1.0@3" → ["leiden_1.0", "3"]
  const at = branchName.indexOf("@");
  if (at < 0) return [null, null];
  return [branchName.substring(0, at), branchName.substring(at+1)];
}

// ── Tree rendering ────────────────────────────────────────────────
let SELECTED_BRANCH = null;

function renderNode(node, container, depth) {
  const div = document.createElement("div");
  div.className = "tree-node";
  if (DATA.markers[node.name]) div.classList.add("has-markers");
  // Leaf only: show icls id. Internal: just name (no n count).
  let label;
  if (!node.children || node.children.length === 0) {
    label = `${node.name} (icls=${node.icls_indices.join(",")})`;
  } else {
    label = node.name;
  }
  div.textContent = label;
  div.dataset.branch = node.name;
  div.addEventListener("click", (e) => {
    e.stopPropagation();
    selectBranch(node.name);
  });
  container.appendChild(div);

  if (node.children && node.children.length) {
    const childDiv = document.createElement("div");
    childDiv.className = "tree-children";
    for (const c of node.children) renderNode(c, childDiv, depth+1);
    container.appendChild(childDiv);
  }
}

function renderTree() {
  const root = document.getElementById("tree");
  root.innerHTML = "";
  for (const n of DATA.tree) renderNode(n, root, 0);
}

// ── Path panel ────────────────────────────────────────────────────
function renderPaths(branchName, iclsIndices) {
  const panel = document.getElementById("paths");
  if (!iclsIndices.length) {
    panel.innerHTML = `<div class="placeholder">No paths for<br><b>${branchName}</b></div>`;
    return;
  }
  let html = `<div class="level-label">${branchName}</div>`;
  html += `<div style="margin-bottom:6px; color:#666;">${iclsIndices.length} path(s)</div>`;
  const sorted = iclsIndices.slice().sort((a,b) => a - b);
  for (const i of sorted) {
    const clusters = DATA.icls_to_clusters[String(i)] || [];
    const parts = LEVELS.map((lvl, k) => `${lvl}@${clusters[k] ?? "?"}`);
    html += `<div class="path-row"><span class="icls-tag">${i}</span>${parts.join(" | ")}</div>`;
  }
  panel.innerHTML = html;
}

// ── UMAP plotting ─────────────────────────────────────────────────
function plotUmap(divId, levelIdx, highlightSet) {
  // highlightSet: Set of cluster strings to highlight at this level
  //               (null/undefined → no highlight, show full coloring)
  const lvl = LEVELS[levelIdx];
  const clusterArr = DATA.cluster[lvl];
  const x = DATA.umap_x;
  const y = DATA.umap_y;
  const n = x.length;

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
      if (clusterArr[k] === c) { xs.push(x[k]); ys.push(y[k]); }
    }
    let color, opacity, size;
    if (!hasHighlight) {
      color = colorFor(i); opacity = 0.7; size = 2.5;
    } else if (highlightSet.has(c)) {
      color = colorFor(i); opacity = 0.95; size = 4;
    } else {
      color = "#cccccc"; opacity = 0.2; size = 2;
    }
    traces.push({
      x: xs, y: ys,
      mode: "markers",
      type: "scattergl",
      name: c,
      marker: { color: color, size: size, opacity: opacity, line: { width: 0 } },
      hoverinfo: "skip",
      showlegend: false,
      customdata: c,
    });
  }

  // Invisible click hit-box at each cluster centroid (numbers are drawn as
  // annotations below to keep them on top of all traces).
  const cent = DATA.centroids[lvl];
  const annotations = [];
  if (cent && cent.labels.length) {
    traces.push({
      x: cent.x, y: cent.y,
      mode: "markers",
      type: "scatter",
      marker: { size: 22, color: "rgba(0,0,0,0)", line: { width: 0 } },
      hoverinfo: "text",
      hovertext: cent.labels.map(c => `${lvl}@${c}`),
      customdata: cent.labels,
      showlegend: false,
    });

    // Build outline + main text via layout annotations (always on top)
    const outline = [[-0.6,0],[0.6,0],[0,-0.6],[0,0.6],[-0.4,-0.4],[0.4,-0.4],[-0.4,0.4],[0.4,0.4]];
    for (let i = 0; i < cent.labels.length; i++) {
      const lbl = cent.labels[i];
      const cx = cent.x[i], cy = cent.y[i];
      const dimmed = hasHighlight && !highlightSet.has(lbl);
      const mainColor = dimmed ? "#aaa" : "#000";
      for (const [dx, dy] of outline) {
        annotations.push({
          x: cx, y: cy,
          text: "<b>" + lbl + "</b>",
          showarrow: false,
          xshift: dx, yshift: dy,
          font: { size: 11, color: "#fff", family: "Arial, sans-serif" },
        });
      }
      annotations.push({
        x: cx, y: cy,
        text: "<b>" + lbl + "</b>",
        showarrow: false,
        font: { size: 11, color: mainColor, family: "Arial, sans-serif" },
      });
    }
  }

  const titleSuffix = hasHighlight
    ? ` :: {${Array.from(highlightSet).join(", ")}}`
    : "";
  const layout = {
    title: { text: lvl + titleSuffix, font: { size: 13 } },
    xaxis: { showticklabels: false, showgrid: false, zeroline: false, ticks: "" },
    yaxis: { showticklabels: false, showgrid: false, zeroline: false, ticks: "",
             scaleanchor: "x", scaleratio: 1 },
    margin: { l: 5, r: 5, t: 30, b: 5 },
    plot_bgcolor: "white",
    paper_bgcolor: "white",
    hovermode: "closest",
    annotations: annotations,
    showlegend: false,
  };
  Plotly.newPlot(divId, traces, layout, { displayModeBar: false, responsive: true });

  // Click handler — works for either centroid text or cell marker
  const plotDiv = document.getElementById(divId);
  plotDiv.on("plotly_click", (ev) => {
    if (!ev || !ev.points || !ev.points.length) return;
    const p = ev.points[0];
    const clusterName = p.customdata || p.data.name;
    if (clusterName == null) return;
    selectBranch(`${lvl}@${clusterName}`);
  });
}

function plotAllUmaps(highlightSets) {
  // highlightSets: [Set, Set, Set] or null for no highlight
  for (let i = 0; i < 3; i++) {
    plotUmap(`umap-${i}`, i, highlightSets ? highlightSets[i] : null);
  }
}

// Aggregate icls_indices across ALL tree nodes with the given branch name.
// e.g. leiden_4.0@8 may appear under multiple parents → we want all paths
// it belongs to, not just the clicked tree instance.
function findAllIclsForBranch(branchName) {
  const result = new Set();
  function walk(nodes) {
    for (const n of nodes) {
      if (n.name === branchName) {
        for (const i of n.icls_indices) result.add(i);
      }
      if (n.children && n.children.length) walk(n.children);
    }
  }
  walk(DATA.tree);
  return Array.from(result);
}

// Given icls indices, compute the 3 cluster sets (one per level) that share
// path with any of these icls. This produces cross-level highlight.
function clusterSetsFromIcls(iclsIndices) {
  const sets = [new Set(), new Set(), new Set()];
  for (const i of iclsIndices) {
    const clusters = DATA.icls_to_clusters[String(i)];
    if (!clusters) continue;
    for (let lvl = 0; lvl < 3; lvl++) {
      sets[lvl].add(clusters[lvl]);
    }
  }
  return sets;
}

// ── Selection handler ─────────────────────────────────────────────
function selectBranch(branchName) {
  SELECTED_BRANCH = branchName;
  document.querySelectorAll(".tree-node").forEach(el => {
    el.classList.toggle("selected", el.dataset.branch === branchName);
  });
  // Aggregate icls across ALL appearances of this branch name in tree
  const iclsIndices = findAllIclsForBranch(branchName);
  renderPaths(branchName, iclsIndices);
  const highlightSets = clusterSetsFromIcls(iclsIndices);
  plotAllUmaps(highlightSets);
}

// ── Init ──────────────────────────────────────────────────────────
renderTree();
plotAllUmaps(null);
</script>
</body>
</html>
"""
