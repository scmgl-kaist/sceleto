"""Interactive HTML viewer for BranchingResult.

Combines the existing :func:`build_interactive_html` style (icls UMAP +
marker comparison heatmap) with a bottom panel of 3 small resolution UMAPs
annotated with branching markers per cluster.

Layout (2x2 grid; left column merged):
    ┌─────────────┬───────────────────────┐
    │             │ Marker Comparison     │
    │  icls UMAP  ├───────────────────────┤
    │ (clickable) │ leiden_1.0 | 2.0 | 4.0│
    │             │ (annotated UMAPs,     │
    │             │  not clickable)       │
    └─────────────┴───────────────────────┘

On clicking an icls cluster in the top-left UMAP:
- top-right marker comparison heatmap updates (same as
  :class:`HierarchyRun.interactive_viewer`)
- bottom-right 3 small UMAPs update with branching markers shown as text:
    * leiden_1.0 UMAP: ancestor cluster's markers
    * leiden_2.0 UMAP: all leiden_2.0 children of l1 ancestor, each
      annotated with its own branching markers
    * leiden_4.0 UMAP: all leiden_4.0 children of l2 ancestor, similarly
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ._viewer import _assign_icls_colors, _build_compare_data


def build_branching_html(
    adata: Any,
    br: Any,
    save: str,
    *,
    n_top: int = 5,
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
        # Meta
        "n_top": int(n_top),
    }

    html = _HTML_TEMPLATE.replace(
        "/*__DATA__*/",
        json.dumps(data_blob, separators=(",", ":")),
    )
    Path(save).write_text(html, encoding="utf-8")


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
    grid-template-columns: 2fr 5fr;
    grid-template-rows: 1fr 1fr;
    height: 100vh;
    gap: 4px;
    padding: 4px;
  }
  #panel-icls   { grid-column: 1; grid-row: 1 / span 2; background: white; border: 1px solid #ddd; padding: 8px;  overflow: hidden;
                  display: flex; align-items: center; justify-content: center; }
  #panel-marker { grid-column: 2; grid-row: 1;          background: white; border: 1px solid #ddd; padding: 12px; overflow: auto; }
  #panel-cross  { grid-column: 2; grid-row: 2;          background: white; border: 1px solid #ddd; padding: 4px;  overflow: hidden; }

  /* Fixed plot sizes — do not resize on window changes. Small windows clip. */
  #icls-umap   { width: 520px; height: 520px; }
  #cross-umaps { display: flex; gap: 2px; }
  .cross-cell  { width: 320px; height: 320px; flex: 0 0 320px; }

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
  <div id="panel-cross">
    <div id="cross-umaps">
      <div id="cross-0" class="cross-cell"></div>
      <div id="cross-1" class="cross-cell"></div>
      <div id="cross-2" class="cross-cell"></div>
    </div>
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

// Fixed-size plot — no resize on window changes; CSS sets 520x520.
let SELECTED_ICLS = null;
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

  const cellW = 28, labelColW = 120;
  const tableW = labelColW + sortedGenes.length * cellW;

  let html = '<div class="info"><b>path ' + icls + '</b> &nbsp; (' + d.n_cells + ' cells)<br>'
           + d.levels.map(l => '<code>' + l + '</code>').join(' &rarr; ') + '</div>';
  html += '<table class="heatmap" style="width:' + tableW + 'px;"><thead><tr><th style="width:' + labelColW + 'px;"></th>';
  for (const g of sortedGenes) {
    html += '<th style="writing-mode:vertical-rl; transform:rotate(180deg);">' + g + '</th>';
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

// ── Initial state: 3 UMAPs with no highlight, no marker text ──────
plotAllCross(null, null);

// ── Click handler: update marker panel + bottom 3 UMAPs ───────────
function onIclsSelected(iclsId) {
  if (iclsId == null) {
    // Reset to initial state
    document.getElementById("marker-placeholder").style.display = "block";
    document.getElementById("marker-content").style.display = "none";
    plotAllCross(null, null);
    return;
  }
  renderMarkerComparison(iclsId);

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
