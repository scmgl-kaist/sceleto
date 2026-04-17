# sceleto

Single-cell RNA-seq analysis toolkit — marker discovery, gene network, and cell type annotation.

## Install

```bash
pip install git+https://github.com/scmgl-kaist/sceleto.git
```

## Modules

| Module | Description |
|--------|-------------|
| `sceleto.markers` | Graph-based marker gene discovery; cross-resolution hierarchy |
| `sceleto.network` | Gene co-expression network construction and visualization |
| `sceleto.annotation` | Cell type annotation with LR transfer and pangeapy integration |
| `sceleto.dotplot` | Publication-ready dotplots |

## Requirements

Python ≥ 3.10, scanpy ≥ 1.10