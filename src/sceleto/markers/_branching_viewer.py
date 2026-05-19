"""Interactive HTML viewer for HierarchyRun with edge-activation panel.

Layout (grid):
    ┌─────────────┬───────────────────────┐
    │             │ Marker Comparison     │
    │  icls UMAP  ├───────────────────────┤
    │ (clickable) │ Gene chips + per-gene │
    │             │ edge-activation graph │
    └─────────────┴───────────────────────┘

The bottom-right panel shows the marker graph (from ``mgr.viz.G``) with
nodes colored by per-gene expression and edges colored by per-gene FC
bin. Genes are selected by clicking a heatmap column, clicking a chip,
mouse wheel, or ←/→ keys. Selection clamps at endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ._viewer import _assign_icls_colors, _build_compare_data
from ._edge_panel import build_edge_data


def build_branching_html(
    adata: Any,
    hr: Any,
    save: str,
    mgr: Any,
    *,
    n_top: int = 5,
) -> None:
    """Build interactive HTML viewer with edge-activation bottom panel.

    Parameters
    ----------
    adata
        AnnData with ``obs['icls']`` (set by hierarchy()) and ``obsm['X_umap']``.
    hr
        :class:`sceleto.markers.HierarchyRun`.
    save
        Output HTML file path.
    mgr
        :class:`sceleto.markers.graph.MarkerGraphRun` whose graph drives the
        edge-activation panel.
    n_top
        Number of top markers to display per branch.
    """
    if "X_umap" not in adata.obsm:
        raise ValueError("adata.obsm['X_umap'] required")
    if "icls" not in adata.obs.columns:
        raise ValueError("adata.obs['icls'] required (set by hierarchy())")
    if mgr is None:
        raise ValueError("mgr is required (MarkerGraphRun for the edge-activation panel)")

    levels: List[str] = list(hr.levels)
    if len(levels) != 3:
        raise ValueError(f"viewer expects 3-level hierarchy; got {len(levels)}")

    umap_xy = adata.obsm["X_umap"]

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

    edge_data = build_edge_data(adata, mgr, marker_data)

    data_blob = {
        "umap":         umap_json,
        "marker_data":  marker_data,
        "icls_colors":  icls_colors,
        "centroids":    centroids_json,
        "levels":       levels,
        "edge_data":    edge_data,
        "n_top":        int(n_top),
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
<title>sceleto Hierarchy Viewer</title>
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

  #icls-umap   { width: 520px; height: 520px; }

  /* Edge-activation panel */
  #gene-chips {
    display: flex; flex-wrap: wrap;
    gap: 3px; padding: 4px 6px;
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
    <div id="gene-chips"></div>
    <div id="edge-graph"></div>
    <div class="edge-panel-hint">click cell · click chip · wheel · ← →</div>
  </div>
</div>

<script>
const DATA = /*__DATA__*/;
const N_TOP = DATA.n_top;

let SELECTED_ICLS = null;
let SORTED_GENES = [];
let SELECTED_GENE = null;

// ── icls UMAP ─────────────────────────────────────────────────────
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

Plotly.newPlot("icls-umap", buildIclsTraces(null), ICLS_LAYOUT, { responsive: false, displayModeBar: false });
document.getElementById("icls-umap").on("plotly_click", function(data) {
  if (!data || !data.points || !data.points.length) return;
  const pt = data.points[0];
  const icls = pt.customdata;
  if (icls == null || !DATA.marker_data[icls]) return;
  SELECTED_ICLS = SELECTED_ICLS === icls ? null : icls;
  Plotly.react("icls-umap", buildIclsTraces(SELECTED_ICLS), ICLS_LAYOUT);
  onIclsSelected(SELECTED_ICLS);
});

// ── Marker Comparison heatmap ─────────────────────────────────────
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
  for (const g of sortedGenes) {
    html += '<th class="gene-th" data-gene="' + g + '" style="writing-mode:vertical-rl; transform:rotate(180deg);">' + g + '</th>';
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
  panel.querySelectorAll("th.gene-th").forEach(el => {
    el.addEventListener("click", () => setSelectedGene(el.dataset.gene));
  });
}

// ── Edge-activation panel ─────────────────────────────────────────
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

  const nodeSize = ed.node_sizes.map(s => Math.max(14, Math.sqrt(s) * 1.4));
  const traces = [{
    x: ed.node_pos.map(p => p[0]),
    y: ed.node_pos.map(p => p[1]),
    mode: "markers", type: "scatter",
    marker: {
      color: vals,
      colorscale: [[0, "rgb(255,255,255)"], [1, "rgb(0,0,0)"]],
      cmin: vmin, cmax: vmax,
      size: nodeSize,
      line: { color: "#000", width: 0.6 },
    },
    text: ed.node_ids,
    hovertemplate: "node %{text}<br>expr=%{marker.color:.2f}<extra></extra>",
    showlegend: false,
  }];

  const annotations = [];
  for (let i = 0; i < ed.edges.length; i++) {
    const [u, v] = ed.edges[i];
    const has = fcMap.has(i);
    const color = has ? ed.fc_colors[fcBinIdx(fcMap.get(i))] : ed.bg_color;
    const opacity = has ? 1.0 : ed.bg_alpha;
    annotations.push({
      x: ed.node_pos[v][0], y: ed.node_pos[v][1],
      ax: ed.node_pos[u][0], ay: ed.node_pos[u][1],
      xref: "x", yref: "y", axref: "x", ayref: "y",
      showarrow: true, arrowhead: 2, arrowsize: 1.0,
      arrowwidth: has ? 2.2 : 1.5,
      arrowcolor: color, opacity: opacity,
      standoff: 10, startstandoff: 8,
      text: "",
    });
  }

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

  const xRange = ed.umap_range.x.slice();
  const yRange = ed.umap_range.y.slice();
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
  if (next < 0 || next >= SORTED_GENES.length) return;
  setSelectedGene(SORTED_GENES[next]);
}

function clearEdgePanel() {
  SORTED_GENES = [];
  SELECTED_GENE = null;
  document.getElementById("gene-chips").innerHTML = "";
  Plotly.purge("edge-graph");
}

(function() {
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
})();

function onIclsSelected(iclsId) {
  if (iclsId == null) {
    document.getElementById("marker-placeholder").style.display = "block";
    document.getElementById("marker-content").style.display = "none";
    clearEdgePanel();
    return;
  }
  renderMarkerComparison(iclsId);
  renderGeneChips();
  if (SORTED_GENES.length) {
    const ed = DATA.edge_data;
    const first = SORTED_GENES.find(g => g in ed.gene_mean) || SORTED_GENES[0];
    setSelectedGene(first);
  }
}
</script>
</body>
</html>
"""
