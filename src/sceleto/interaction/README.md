# sceleto.interaction

Cell–cell communication inference — a Python port of [CellChat](https://github.com/jinworks/CellChat)
(Jin et al., *Nat. Commun.* 2021) and its spatial extension, built directly on `AnnData`.

The port reproduces CellChat's core numerics to R parity: identical communication
probabilities, permutation p-values, pathway aggregation, network centrality,
manifold embedding/clustering, and communication-pattern (NMF) decomposition. The
bundled CellChatDB (human, mouse, zebrafish) ships with the package, so no external
download or R runtime is required.

## Quick start

The entry point is `CellComm`, constructed from an `AnnData` and a `groupby` column
in `adata.obs` holding the cell-group labels. Expression is expected
log-normalized (e.g. `sc.pp.normalize_total` + `sc.pp.log1p`).

```python
import scanpy as sc
import sceleto.interaction as ix

# adata: log-normalized AnnData with a cell-group column in adata.obs
adata = sc.read_h5ad("pbmc.h5ad")

cc = ix.CellComm(adata, groupby="cell_type", species="human")

# core pipeline
cc.identify_overexpressed()                 # over-expressed genes / ligand-receptor pairs
cc.compute_communication(nboot=100, n_jobs=4)  # LR-level probability + permutation p-values
cc.aggregate()                              # cell-group × cell-group aggregated network
cc.compute_communication_pathway()          # collapse LR pairs into signaling pathways

# rank signaling pathways by overall information flow
ranking = cc.rank_net()
print(ranking.head())
#   pathway_name      flow
# 0         TGFb  0.921354
# 1          TNF  0.577123
# 2          IL6  0.576402
```

The pipeline is chainable — each step returns `self`:

```python
cc = (
    ix.CellComm(adata, groupby="cell_type")
    .identify_overexpressed()
    .compute_communication(nboot=100, n_jobs=4)
    .aggregate()
    .compute_communication_pathway()
)
```

`compute_communication` runs the bootstrap permutation test in parallel
(`n_jobs`); results are deterministic and independent of `n_jobs` for a given
`seed`.

## Accessing results

```python
cc.prob        # LR-level probability tensor        (n_groups, n_groups, n_LR)
cc.pval        # permutation p-values               (n_groups, n_groups, n_LR)
cc.weight      # aggregated interaction strength     (n_groups, n_groups)
cc.count       # aggregated interaction count        (n_groups, n_groups)
cc.pathways    # list of significant signaling pathways
cc.group_names # cell-group labels, in matrix order
```

## Visualization

```python
# aggregated cell-group network
ix.plot_network_circle(cc.weight, cc.group_names)   # circular network
ix.plot_heatmap(cc.weight, cc.group_names)          # heatmap

# pathway-level
ix.plot_rank_net(cc.rank_net())                     # pathway information-flow ranking
ix.plot_aggregate(cc, signaling="TGFb")             # network for one pathway
ix.plot_signaling_role_scatter(cc)                  # sender vs. receiver role scatter

# ligand-receptor detail
ix.plot_bubble(cc.subset_communication())           # LR bubble plot
```

Other plotters: `plot_chord`, `plot_chord_gene`, `plot_contribution`,
`plot_signaling_role_heatmap`, `plot_signaling_role_network`,
`plot_communication_patterns` / `_dot` / `_river`, `plot_net_embedding` /
`_zoom`.

## Downstream analyses

```python
# signaling-role centrality (outgoing / incoming / mediator / influencer)
ix.plot_signaling_role_network(cc, signaling="TGFb")   # roles for one pathway
role = ix.signaling_role_pathway(cc, pattern="outgoing")  # pathway × cell-group matrix
ix.plot_signaling_role_heatmap(role)

# communication patterns (NMF) — outgoing or incoming
ix.select_k(cc, pattern="outgoing")                 # choose the number of patterns
ix.identify_communication_patterns(cc, pattern="outgoing", k=3)

# signaling-pathway similarity manifold
ix.compute_net_similarity(cc, type="functional")
ix.net_embedding(cc, type="functional")
ix.net_clustering(cc, type="functional")
```

## Comparing conditions

`MultiCellComm` holds several `CellComm` objects (e.g. control vs. disease) and
supports differential comparison of interaction strength and pathway flow.

```python
mcc = ix.MultiCellComm([cc_ctrl, cc_dis], names=["control", "disease"])
```

## Spatial data

Pass spatial coordinates to enable the spatially-constrained probability model
(distance-limited signaling), matching CellChat v2's spatial mode:

```python
cc = ix.CellComm(
    adata, groupby="cell_type",
    coordinates=adata.obsm["spatial"],
    spatial_factors={"ratio": 1.0, "tol": 5.0},
)
```

## Database

```python
ix.list_cellcommdb()                     # available databases
db = ix.load_cellcommdb("human")         # human / mouse / zebrafish
db = db.subset(annotation="Secreted Signaling")   # e.g. secreted-signaling only
cc = ix.CellComm(adata, groupby="cell_type", db=db)
```

## Reference

Jin, S. et al. Inference and analysis of cell-cell communication using CellChat.
*Nat. Commun.* **12**, 1088 (2021).
