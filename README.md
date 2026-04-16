# sceleto

A backbone package for single-cell RNA-seq analysis — marker discovery, gene network, and cell type annotation.

## Install

```bash
pip install git+https://github.com/scmgl-kaist/sceleto.git
```

## Quick Start

```python
import scanpy as sc
import sceleto as scl

adata = sc.read_h5ad("your_data.h5ad")

# Graph-based marker detection (PAGA + auto FC threshold)
result = scl.markers.marker(adata, groupby="cell_type")

# Per-group marker genes
result.markers  # {'T_cell': ['CD3D', 'CD3E', ...], 'B_cell': ['MS4A1', ...], ...}

# Dotplot
scl.dotplot(adata, result.markers, "cell_type")

# Cross-resolution hierarchy (3 leiden levels → icls)
m1 = scl.markers.marker(adata, groupby="leiden_0.1")
m2 = scl.markers.marker(adata, groupby="leiden_0.5")
m3 = scl.markers.marker(adata, groupby="leiden_1.0")
res = scl.markers.hierarchy(adata, [m1, m2, m3])

res.compare_markers("0")                          # static heatmap
res.interactive_viewer(adata, save="viewer.html")  # interactive HTML
```

## Modules

- **markers** — graph-based and simple marker gene discovery
- **annotation** — cell type annotation (custom LR transfer + pangeapy wrapper)
- **network** — gene network construction and visualization

## Requirements

Python >= 3.10, scanpy >= 1.10, anndata >= 0.10, numpy >= 1.24, pandas >= 2.0, scipy >= 1.11, scikit-learn >= 1.3, matplotlib >= 3.7, seaborn >= 0.13