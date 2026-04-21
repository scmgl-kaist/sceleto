# sceleto

[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue)](https://sceleto.readthedocs.io/en/latest/)

Single-cell RNA-seq analysis toolkit — marker discovery, gene network, and cell type annotation.

Documentation: https://sceleto.readthedocs.io/en/latest/

## Install

```bash
pip install git+https://github.com/scmgl-kaist/sceleto.git
```
If the update isn't reflected even after reinstalling, try:
```bash
pip install --force-reinstall git+https://github.com/scmgl-kaist/sceleto.git
```

## Modules

| Module | Description |
|--------|-------------|
| `sceleto.markers` | Graph-based marker gene discovery; cross-resolution hierarchy |
| `sceleto.network` | Gene co-expression network construction and visualization |
| `sceleto.annotation` | Cell type annotation with LR transfer and pangeapy integration |

## Requirements

Python ≥ 3.10, scanpy ≥ 1.10