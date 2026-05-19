"""Shared helper for the edge-activation panel.

The panel renders, per gene, the marker graph (mgr.viz.G) with:
- nodes colored by mean expression
- edges colored by FC bin (with arrow direction)

``build_edge_data`` extracts the JS payload (filtered to the gene union
that appears in any heatmap cluster, to keep size small).

Used by both the non-batch hierarchy viewer (_branching_viewer.py) and the
batch hierarchy viewer (_viewer.py:build_interactive_html_batch).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def build_edge_data(adata: Any, mgr: Any, marker_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract edge-activation data from MarkerGraphRun for JS rendering.

    Node positions = UMAP centroids of ``adata.obs[mgr.groupby]`` so the
    graph aligns with the icls UMAP shown alongside.

    Filters mean_mat / gene_edge_fc to the union of marker genes that may
    appear in any heatmap (the marker_data x-axes), to keep payload small.

    Parameters
    ----------
    adata
        AnnData with ``obsm['X_umap']`` and ``obs[mgr.groupby]``.
    mgr
        :class:`sceleto.markers.graph.MarkerGraphRun`. ``mgr.viz`` must have
        ``mean_mat``, ``gene_edge_fc``, ``G``, ``node_sizes``.
    marker_data
        Per-icls heatmap data (from ``_build_compare_data`` or batch variant).
        Used to compute the gene union for payload filtering.
    """
    viz = mgr.viz
    if viz.mean_mat is None or viz.gene_edge_fc is None:
        raise ValueError("mgr.viz must have mean_mat and gene_edge_fc set.")

    if "X_umap" not in adata.obsm:
        raise ValueError("adata.obsm['X_umap'] required for edge graph layout")
    groupby = mgr.groupby
    if groupby not in adata.obs.columns:
        raise ValueError(f"adata.obs[{groupby!r}] required for edge graph layout")

    nodes = list(viz.G.nodes())
    node_ids = [str(n) for n in nodes]
    node_idx = {n: i for i, n in enumerate(nodes)}
    node_sizes = [float(s) for s in viz.node_sizes]

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

    all_marker_genes: set = set()
    for d in marker_data.values():
        all_marker_genes.update(d.get("genes", []))

    mm_cols = set(viz.mean_mat.columns)
    genes = [g for g in all_marker_genes if g in mm_cols]

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

    fc_bins = [0.0, 3.0, 4.0, 5.0, 6.0, float("inf")]
    fc_colors = ["lightgrey", "#9fdab8", "#57b8d0", "#1d7eb7", "#084081"]
    fc_bins_js = [b if np.isfinite(b) else None for b in fc_bins]

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


# JavaScript snippet for rendering the edge panel.
# Expects the following globals to exist already in the page:
# - DATA.edge_data (built by build_edge_data)
# - SORTED_GENES, SELECTED_GENE (let bindings)
# - Markup: #gene-chips, #edge-graph, #panel-cross (the panel; tabindex=0)
# - On heatmap render, gene <th> elements get class="gene-th" and data-gene=<name>
# Call setSelectedGene(g) to drive the panel; stepGene(±1) for prev/next.
EDGE_PANEL_JS = r"""
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
  if (!xPanel) return;
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
"""


# CSS for the edge-activation panel (chip strip + graph div + heatmap gene-th).
EDGE_PANEL_CSS = """
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
"""
