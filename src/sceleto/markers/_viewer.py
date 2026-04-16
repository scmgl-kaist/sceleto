"""Interactive HTML viewer for HierarchyRun.

Generates a self-contained HTML file with a Plotly UMAP scatter and
a click-to-view marker comparison heatmap per icls.
No additional Python dependencies required (Plotly loaded via CDN).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _assign_icls_colors(adata: Any) -> Dict[str, str]:
    """Get or assign scanpy-style categorical colors for icls."""
    import scanpy as sc

    key = "icls"
    if f"{key}_colors" not in adata.uns:
        sc.pl.umap(adata, color=key, show=False)
    colors_list = adata.uns[f"{key}_colors"]
    obs_col = adata.obs[key]
    if hasattr(obs_col, "cat"):
        categories = list(obs_col.cat.categories)
    else:
        categories = sorted(obs_col.unique().tolist(), key=lambda x: int(x) if str(x).isdigit() else x)
    return {cat: color for cat, color in zip(categories, colors_list)}


def _build_compare_data(
    icls_full_dict: Dict[str, str],
    full_gene_lists: Dict[str, List[str]],
    n_top: int,
    icls_cell_counts: Dict[str, int],
) -> Dict[str, Any]:
    """Pre-compute compare_markers data for all icls."""
    compare_data = {}

    for icls_id, full_path in icls_full_dict.items():
        parts = full_path.split("|")

        sets = []
        for lid in parts:
            if lid in full_gene_lists:
                sets.append(set(full_gene_lists[lid][:n_top]))
            else:
                sets.append(set())

        union = sorted(set().union(*sets))
        if not union:
            continue

        presence = {}
        for gene in union:
            presence[gene] = [1 if gene in s else 0 for s in sets]

        compare_data[icls_id] = {
            "path": full_path,
            "levels": parts,
            "genes": union,
            "presence": presence,
            "n_cells": icls_cell_counts.get(icls_id, 0),
        }

    return compare_data


def build_interactive_html(
    adata: Any,
    icls_full_dict: Dict[str, str],
    full_gene_lists: Dict[str, List[str]],
    n_top: int,
    save: str,
) -> None:
    """Build and write the interactive HTML viewer.

    Parameters
    ----------
    adata
        AnnData with ``obs['icls']`` and ``obsm['X_umap']``.
    icls_full_dict
        Mapping from icls id to full path string.
    full_gene_lists
        Mapping from leiden id to ranked gene list.
    n_top
        Number of top markers to display.
    save
        Output file path.
    """
    # ── data extraction ──────────────────────────────────────────
    umap_coords = adata.obsm["X_umap"]
    df = pd.DataFrame(
        {"umap_x": umap_coords[:, 0], "umap_y": umap_coords[:, 1],
         "icls": adata.obs["icls"].values},
        index=adata.obs.index,
    )

    # cell counts per icls
    icls_counts = df["icls"].value_counts().to_dict()
    icls_cell_counts = {str(k): int(v) for k, v in icls_counts.items()}

    # compare_markers data
    marker_data = _build_compare_data(
        icls_full_dict, full_gene_lists, n_top, icls_cell_counts,
    )

    # colors
    colors = _assign_icls_colors(adata)

    # umap json
    umap_json = df[["umap_x", "umap_y", "icls"]].to_dict(orient="list")

    # centroids
    centroids = df.groupby("icls")[["umap_x", "umap_y"]].median().reset_index()
    centroids_json = centroids.to_dict(orient="list")

    # ── HTML ─────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>sceleto Interactive Marker Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #f5f5f5; }}
  #container {{ display: flex; height: 100vh; }}
  #umap-panel {{ width: 550px; min-width: 400px; padding: 10px; display: flex; align-items: center; justify-content: center; }}
  #marker-panel {{
    flex: 1; padding: 16px; overflow-y: auto;
    background: white; border-left: 1px solid #ddd;
    box-shadow: -2px 0 8px rgba(0,0,0,0.05);
  }}
  #marker-panel h2 {{ font-size: 15px; margin-bottom: 8px; color: #333; }}
  #marker-panel .info {{ font-size: 12px; color: #666; margin-bottom: 12px; }}
  #marker-panel .placeholder {{
    color: #999; font-size: 13px; margin-top: 40px; text-align: center;
  }}
  table.heatmap {{ border-collapse: collapse; font-size: 11px; }}
  table.heatmap th, table.heatmap td {{
    text-align: center; border: 1px solid #ccc; padding: 3px 0;
  }}
  table.heatmap th:first-child, table.heatmap td:first-child {{
    padding: 3px 8px; white-space: nowrap;
  }}
  table.heatmap th {{ background: #f0f0f0; position: sticky; top: 0; }}
  td.present {{ background: #3182bd; color: white; }}
  td.absent {{ background: #f7f7f7; color: #bbb; }}
  .legend {{ font-size: 11px; margin-top: 8px; color: #666; }}
  .legend span {{
    display: inline-block; width: 14px; height: 14px;
    vertical-align: middle; margin-right: 4px; border: 1px solid #ccc;
  }}
</style>
</head>
<body>
<div id="container">
  <div id="umap-panel">
    <div id="umap-plot" style="width:520px; height:520px;"></div>
  </div>
  <div id="marker-panel">
    <h2>Marker Comparison</h2>
    <div class="placeholder">Click an icls cluster on the UMAP to view markers.</div>
    <div id="marker-content" style="display:none;"></div>
  </div>
</div>

<script>
const umapData = {json.dumps(umap_json)};
const markerData = {json.dumps(marker_data)};
const centroids = {json.dumps(centroids_json)};
const iclsColors = {json.dumps(colors)};

const iclsSet = [...new Set(umapData.icls)];
const traces = [];

const groups = {{}};
for (let i = 0; i < umapData.icls.length; i++) {{
  const id = umapData.icls[i];
  if (!groups[id]) groups[id] = {{x: [], y: []}};
  groups[id].x.push(umapData.umap_x[i]);
  groups[id].y.push(umapData.umap_y[i]);
}}

for (const id of iclsSet.sort((a,b) => parseInt(a) - parseInt(b))) {{
  traces.push({{
    x: groups[id].x,
    y: groups[id].y,
    mode: 'markers',
    type: 'scattergl',
    name: 'icls ' + id,
    marker: {{ color: iclsColors[id], size: 2, opacity: 0.5 }},
    hoverinfo: 'skip',
    showlegend: false,
  }});
}}

traces.push({{
  x: centroids.umap_x,
  y: centroids.umap_y,
  mode: 'markers',
  type: 'scattergl',
  marker: {{ size: 22, color: 'rgba(0,0,0,0)', line: {{ width: 0 }} }},
  customdata: centroids.icls,
  hoverinfo: 'text',
  hovertext: centroids.icls.map(id => 'icls ' + id),
  showlegend: false,
}});

const outlineOffsets = [[-0.8,0],[0.8,0],[0,-0.8],[0,0.8]];
const annotations = [];
for (let i = 0; i < centroids.icls.length; i++) {{
  const id = centroids.icls[i];
  const cx = centroids.umap_x[i];
  const cy = centroids.umap_y[i];
  for (const [dx, dy] of outlineOffsets) {{
    annotations.push({{
      x: cx, y: cy,
      text: '<b>' + id + '</b>',
      showarrow: false,
      xshift: dx, yshift: dy,
      font: {{ size: 11, color: '#fff', family: 'Arial, sans-serif' }},
    }});
  }}
  annotations.push({{
    x: cx, y: cy,
    text: '<b>' + id + '</b>',
    showarrow: false,
    font: {{ size: 11, color: '#000', family: 'Arial, sans-serif' }},
  }});
}}

const layout = {{
  title: 'UMAP — icls (click a label)',
  xaxis: {{ title: 'UMAP1', zeroline: false }},
  yaxis: {{ title: 'UMAP2', zeroline: false }},
  showlegend: false,
  margin: {{ l: 50, r: 10, t: 40, b: 40 }},
  hovermode: 'closest',
  annotations: annotations,
}};

Plotly.newPlot('umap-plot', traces, layout, {{responsive: true}});

document.getElementById('umap-plot').on('plotly_click', function(data) {{
  const pt = data.points[0];
  const icls = pt.customdata;
  if (!icls || !markerData[icls]) return;

  const d = markerData[icls];
  const panel = document.getElementById('marker-content');
  const placeholder = document.querySelector('.placeholder');
  placeholder.style.display = 'none';
  panel.style.display = 'block';

  let html = '<div class="info">';
  html += '<b>icls ' + icls + '</b> &nbsp; (' + d.n_cells + ' cells)<br>';
  html += d.levels.map(l => '<code>' + l + '</code>').join(' &rarr; ');
  html += '</div>';

  const sortedGenes = [...d.genes].sort((a, b) => {{
    for (let lv = 0; lv < d.levels.length; lv++) {{
      const diff = d.presence[b][lv] - d.presence[a][lv];
      if (diff !== 0) return diff;
    }}
    return a.localeCompare(b);
  }});

  const cellW = 28;
  const labelColW = 120;
  const tableW = labelColW + sortedGenes.length * cellW;
  html += '<table class="heatmap" style="width:' + tableW + 'px;"><thead><tr><th style="width:' + labelColW + 'px;"></th>';
  for (const g of sortedGenes) {{
    html += '<th style="writing-mode:vertical-rl; transform:rotate(180deg);">' + g + '</th>';
  }}
  html += '</tr></thead><tbody>';

  for (let i = 0; i < d.levels.length; i++) {{
    html += '<tr><td style="text-align:left; white-space:nowrap; font-weight:bold;">' + d.levels[i] + '</td>';
    for (const g of sortedGenes) {{
      const v = d.presence[g][i];
      html += '<td class="' + (v ? 'present' : 'absent') + '"></td>';
    }}
    html += '</tr>';
  }}
  html += '</tbody></table>';

  html += '<div class="legend">';
  html += '<span style="background:#3182bd;"></span> in top-{n_top} markers &nbsp;';
  html += '<span style="background:#f7f7f7;"></span> not in top-{n_top}';
  html += '</div>';

  panel.innerHTML = html;
}});
</script>
</body>
</html>
"""

    Path(save).write_text(html, encoding="utf-8")
    print(f"Interactive viewer saved → {save}")


# ---------------------------------------------------------------------------
# Batch viewer
# ---------------------------------------------------------------------------

def _build_batch_compare_data(
    icls_full_dict: Dict[str, str],
    full_gene_lists: Dict[str, List[str]],
    n_top: int,
    icls_cell_counts: Dict[str, int],
    batch_expression: Dict[str, Any],
) -> Dict[str, Any]:
    """Pre-compute batch compare data for all icls."""
    batches = list(next(iter(batch_expression.values())).batches)
    n_batches = len(batches)
    compare_data = {}

    for icls_id, full_path in icls_full_dict.items():
        leiden_list = full_path.split("|")

        sets = []
        for lid in leiden_list:
            if lid in full_gene_lists:
                sets.append(set(full_gene_lists[lid][:n_top]))
            else:
                sets.append(set())

        union_set = set().union(*sets)
        if not union_set:
            continue

        # Sort genes by presence pattern (same as compare_markers_batch)
        union = sorted(union_set)
        presence_df = pd.DataFrame(
            {lid: [1 if g in s else 0 for g in union]
             for lid, s in zip(leiden_list, sets)},
            index=union,
        ).sort_values(leiden_list, ascending=False)
        union = presence_df.index.tolist()

        presence = {g: [1 if g in s else 0 for s in sets] for g in union}

        n_rows = len(leiden_list)
        n_cols = len(union)
        raw_vals = np.zeros((n_rows, n_cols, n_batches), dtype=np.float32)
        active = np.zeros((n_rows, n_batches), dtype=bool)

        for i, lid in enumerate(leiden_list):
            groupby, group_name = lid.split("@", 1)
            be = batch_expression[groupby]
            if group_name not in be.group_to_idx:
                continue
            g_idx = be.group_to_idx[group_name]
            active[i] = be.n_cells[g_idx] > 0
            gene_to_idx = {g: int(k) for k, g in enumerate(be.genes)}
            for j, gene in enumerate(union):
                if gene in gene_to_idx:
                    raw_vals[i, j] = be.mean[g_idx, :, gene_to_idx[gene]]

        # Per-cell max normalization
        batch_vals = []
        for i in range(n_rows):
            act_i = active[i]
            row = []
            for j in range(n_cols):
                means_j = raw_vals[i, j].copy()
                active_means = means_j[act_i]
                cell_max = (float(active_means.max())
                            if act_i.any() and active_means.max() > 0 else 1.0)
                means_j /= cell_max
                # Sort: active first, descending value; inactive last
                order = sorted(
                    range(n_batches),
                    key=lambda b: (not bool(act_i[b]), -float(means_j[b])),
                )
                # Each entry: [batch_name, value_or_null]
                cell_vals = [
                    [batches[b], None if not act_i[b] else round(float(means_j[b]), 4)]
                    for b in order
                ]
                row.append(cell_vals)
            batch_vals.append(row)

        compare_data[icls_id] = {
            "path": full_path,
            "levels": leiden_list,
            "genes": union,
            "presence": presence,
            "batches": batches,
            "batch_vals": batch_vals,
            "n_cells": icls_cell_counts.get(icls_id, 0),
        }

    return compare_data


def build_interactive_html_batch(
    adata: Any,
    icls_full_dict: Dict[str, str],
    full_gene_lists: Dict[str, List[str]],
    batch_expression: Dict[str, Any],
    n_top: int,
    save: str,
) -> None:
    """Build and write the interactive batch HTML viewer.

    Parameters
    ----------
    adata
        AnnData with ``obs['icls']`` and ``obsm['X_umap']``.
    icls_full_dict
        Mapping from icls id to full path string.
    full_gene_lists
        Mapping from leiden id to ranked gene list.
    batch_expression
        Per-batch expression data from HierarchyRun.
    n_top
        Number of top markers to display.
    save
        Output file path.
    """
    # ── data extraction ──────────────────────────────────────────
    umap_coords = adata.obsm["X_umap"]
    df = pd.DataFrame(
        {"umap_x": umap_coords[:, 0], "umap_y": umap_coords[:, 1],
         "icls": adata.obs["icls"].values},
        index=adata.obs.index,
    )
    icls_counts = df["icls"].value_counts().to_dict()
    icls_cell_counts = {str(k): int(v) for k, v in icls_counts.items()}

    marker_data = _build_batch_compare_data(
        icls_full_dict, full_gene_lists, n_top, icls_cell_counts, batch_expression,
    )
    colors = _assign_icls_colors(adata)
    umap_json = df[["umap_x", "umap_y", "icls"]].to_dict(orient="list")
    centroids = df.groupby("icls")[["umap_x", "umap_y"]].median().reset_index()
    centroids_json = centroids.to_dict(orient="list")

    # ── HTML ─────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>sceleto Batch Marker Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #f5f5f5; }}
  #container {{ display: flex; height: 100vh; }}
  #umap-panel {{ width: 550px; min-width: 400px; padding: 10px; display: flex; align-items: center; justify-content: center; }}
  #marker-panel {{
    flex: 1; padding: 16px; overflow: auto;
    background: white; border-left: 1px solid #ddd;
    box-shadow: -2px 0 8px rgba(0,0,0,0.05);
  }}
  #marker-panel h2 {{ font-size: 15px; margin-bottom: 8px; color: #333; }}
  #marker-panel .info {{ font-size: 12px; color: #666; margin-bottom: 12px; }}
  #marker-panel .placeholder {{
    color: #999; font-size: 13px; margin-top: 40px; text-align: center;
  }}
  table.heatmap {{ border-collapse: collapse; font-size: 11px; }}
  table.heatmap th, table.heatmap td {{
    text-align: center; padding: 0;
  }}
  table.heatmap th:first-child {{ padding: 3px 8px; white-space: nowrap; }}
  table.heatmap td:first-child {{ padding: 3px 8px; white-space: nowrap; text-align: left; font-weight: bold; }}
  table.heatmap th {{ background: #f0f0f0; position: sticky; top: 0; z-index: 1; }}
  .legend-row {{ display: flex; align-items: center; gap: 12px; font-size: 11px; color: #666; margin-top: 10px; flex-wrap: wrap; }}
  .legend-swatch {{ display: inline-block; width: 14px; height: 14px; vertical-align: middle; border: 1px solid #aaa; margin-right: 3px; }}
  .batch-legend {{ display: flex; flex-wrap: wrap; gap: 6px; font-size: 11px; margin-top: 8px; }}
  .batch-legend-item {{ display: flex; align-items: center; gap: 3px; white-space: nowrap; }}
  .batch-dot {{ width: 10px; height: 10px; border-radius: 2px; border: 1px solid #aaa; flex-shrink: 0; }}
</style>
</head>
<body>
<div id="container">
  <div id="umap-panel">
    <div id="umap-plot" style="width:520px; height:520px;"></div>
  </div>
  <div id="marker-panel">
    <h2>Batch Marker Comparison</h2>
    <div class="placeholder">Click an icls cluster on the UMAP to view markers.</div>
    <div id="marker-content" style="display:none;"></div>
  </div>
</div>

<script>
const umapData = {json.dumps(umap_json)};
const markerData = {json.dumps(marker_data)};
const centroids = {json.dumps(centroids_json)};
const iclsColors = {json.dumps(colors)};

// ── UMAP plot ──────────────────────────────────────────────────
const iclsSet = [...new Set(umapData.icls)];
const groups = {{}};
for (let i = 0; i < umapData.icls.length; i++) {{
  const id = umapData.icls[i];
  if (!groups[id]) groups[id] = {{x: [], y: []}};
  groups[id].x.push(umapData.umap_x[i]);
  groups[id].y.push(umapData.umap_y[i]);
}}

const traces = [];
for (const id of iclsSet.sort((a,b) => parseInt(a) - parseInt(b))) {{
  traces.push({{
    x: groups[id].x, y: groups[id].y,
    mode: 'markers', type: 'scattergl',
    name: 'icls ' + id,
    marker: {{ color: iclsColors[id], size: 2, opacity: 0.5 }},
    hoverinfo: 'skip', showlegend: false,
  }});
}}

traces.push({{
  x: centroids.umap_x, y: centroids.umap_y,
  mode: 'markers', type: 'scattergl',
  marker: {{ size: 22, color: 'rgba(0,0,0,0)', line: {{ width: 0 }} }},
  customdata: centroids.icls,
  hoverinfo: 'text',
  hovertext: centroids.icls.map(id => 'icls ' + id),
  showlegend: false,
}});

const outlineOffsets = [[-0.8,0],[0.8,0],[0,-0.8],[0,0.8]];
const annotations = [];
for (let i = 0; i < centroids.icls.length; i++) {{
  const id = centroids.icls[i];
  const cx = centroids.umap_x[i], cy = centroids.umap_y[i];
  for (const [dx, dy] of outlineOffsets) {{
    annotations.push({{ x: cx, y: cy, text: '<b>' + id + '</b>', showarrow: false,
      xshift: dx, yshift: dy, font: {{ size: 11, color: '#fff', family: 'Arial, sans-serif' }} }});
  }}
  annotations.push({{ x: cx, y: cy, text: '<b>' + id + '</b>', showarrow: false,
    font: {{ size: 11, color: '#000', family: 'Arial, sans-serif' }} }});
}}

Plotly.newPlot('umap-plot', traces, {{
  title: 'UMAP — icls (click a label)',
  xaxis: {{ title: 'UMAP1', zeroline: false }},
  yaxis: {{ title: 'UMAP2', zeroline: false }},
  showlegend: false,
  margin: {{ l: 50, r: 10, t: 40, b: 40 }},
  hovermode: 'closest',
  annotations: annotations,
}}, {{responsive: true}});

// ── Batch heatmap rendering ────────────────────────────────────
function redsColor(v) {{
  // Approximate matplotlib Reds colormap
  const r = Math.round(255 - v * 152);
  const g = Math.round(245 - v * 245);
  const b = Math.round(240 - v * 227);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function renderBatchHeatmap(d) {{
  const {{ genes, levels, batches, batch_vals, presence, n_cells }} = d;
  const CELL_H = 22;
  const LABEL_W = 130;
  const cellW = 26;
  const tableW = LABEL_W + genes.length * cellW;

  let html = '<div class="info">';
  html += `<b>icls ${{d.path.split('|').join(' → ')}}</b> &nbsp; (${{n_cells}} cells)<br>`;
  html += levels.map(l => `<code>${{l}}</code>`).join(' &rarr; ');
  html += '</div>';

  html += `<table class="heatmap" style="width:${{tableW}}px;"><thead><tr>`;
  html += `<th style="width:${{LABEL_W}}px;"></th>`;
  for (const g of genes) {{
    html += `<th style="width:${{cellW}}px;"><div style="writing-mode:vertical-rl; transform:rotate(180deg); padding:2px 0;">${{g}}</div></th>`;
  }}
  html += '</tr></thead><tbody>';

  for (let i = 0; i < levels.length; i++) {{
    html += `<tr><td>${{levels[i]}}</td>`;
    for (let j = 0; j < genes.length; j++) {{
      const gene = genes[j];
      const inSet = presence[gene][i];
      if (!inSet) {{
        html += `<td style="background:#f5f5f5; border:1px solid #e0e0e0; height:${{CELL_H}}px;"></td>`;
      }} else {{
        const vals = batch_vals[i][j]; // [[batch_name, value_or_null], ...]
        let strips = `<div style="display:flex; height:${{CELL_H}}px; width:${{cellW}}px;">`;
        for (let b = 0; b < vals.length; b++) {{
          const [bname, v] = vals[b];
          let bg;
          if (v === null) {{
            bg = '#cccccc';
          }} else if (v === 0) {{
            bg = 'white';
          }} else {{
            bg = redsColor(v);
          }}
          const tip = v === null ? `${{bname}}: no cells` : `${{bname}}: ${{v.toFixed(3)}}`;
          strips += `<div style="flex:1; background:${{bg}};" title="${{tip}}"></div>`;
        }}
        strips += '</div>';
        html += `<td style="padding:0; border:1px solid #444;">${{strips}}</td>`;
      }}
    }}
    html += '</tr>';
  }}
  html += '</tbody></table>';

  // Legend
  html += '<div class="legend-row">';
  html += '<span><span class="legend-swatch" style="background:#cccccc;"></span>no cells in batch</span>';
  html += '<span><span class="legend-swatch" style="background:white;"></span>expr = 0</span>';
  html += `<span><span class="legend-swatch" style="background:${{redsColor(0.7)}};"></span>mean expr (per-cell max = 1)</span>`;
  html += '<span><span class="legend-swatch" style="background:#f5f5f5; border:1px solid #e0e0e0;"></span>not in top-{n_top}</span>';
  html += '</div>';

  // Batch color index
  const batchColors = batches.map((_, b) => {{
    const hue = Math.round(b * 360 / batches.length);
    return `hsl(${{hue}}, 60%, 60%)`;
  }});
  html += '<div class="batch-legend">';
  for (let b = 0; b < batches.length; b++) {{
    html += `<div class="batch-legend-item"><div class="batch-dot" style="background:${{batchColors[b]}};"></div>${{batches[b]}}</div>`;
  }}
  html += '</div>';

  return html;
}}

// ── Click handler ──────────────────────────────────────────────
document.getElementById('umap-plot').on('plotly_click', function(data) {{
  const pt = data.points[0];
  const icls = pt.customdata;
  if (!icls || !markerData[icls]) return;

  const panel = document.getElementById('marker-content');
  document.querySelector('.placeholder').style.display = 'none';
  panel.style.display = 'block';
  panel.innerHTML = renderBatchHeatmap(markerData[icls]);
}});
</script>
</body>
</html>
"""

    Path(save).write_text(html, encoding="utf-8")
    print(f"Batch interactive viewer saved → {save}")
