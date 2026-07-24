"""Tests for the ``sceleto.interaction`` (CellChat port) submodule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")

from sceleto import interaction as I


# --------------------------------------------------------------------------
# database
# --------------------------------------------------------------------------
@pytest.mark.parametrize("species", ["human", "mouse", "zebrafish"])
def test_load_db(species):
    db = I.load_cellcommdb(species)
    assert len(db.interaction) > 1000
    assert {"ligand", "receptor", "pathway_name", "annotation"} <= set(db.interaction.columns)
    assert len(db.pathways) > 50


def test_db_bad_species():
    with pytest.raises(ValueError):
        I.load_cellcommdb("elephant")


def test_complex_resolution():
    db = I.load_cellcommdb("human")
    # TGFbR1_R2 is a known complex → two subunits
    subs = db.resolve_genes("TGFbR1_R2")
    assert subs == ["TGFBR1", "TGFBR2"]
    # a single gene resolves to itself
    assert db.resolve_genes("TGFB1") == ["TGFB1"]


def test_subset_by_annotation():
    db = I.load_cellcommdb("human")
    sub = db.subset(annotation="Secreted Signaling")
    assert set(sub.interaction["annotation"].unique()) == {"Secreted Signaling"}
    with pytest.raises(ValueError):
        db.subset(annotation="NopeSignaling")


# --------------------------------------------------------------------------
# gene mapper — cross-species + symbol-update robustness
# --------------------------------------------------------------------------
def test_gene_mapper_direct_and_alias():
    db = I.load_cellcommdb("human")
    gm = I.GeneMapper.from_db(db)
    # direct
    assert gm.resolve("TGFB1") == "TGFB1"
    # case-insensitive (mouse-cased query vs human DB)
    assert gm.resolve("Tgfb1") == "TGFB1"
    # unknown
    assert gm.resolve("NOTAGENE") is None


def test_gene_mapper_bulk():
    db = I.load_cellcommdb("human")
    gm = I.GeneMapper.from_db(db)
    res = gm.map(["TGFB1", "Tgfb1", "NOTAGENE"])
    assert res.mapping["TGFB1"] == "TGFB1"
    assert res.mapping["Tgfb1"] == "TGFB1"
    assert "NOTAGENE" in res.unmatched
    assert res.n_matched == 2


# --------------------------------------------------------------------------
# end-to-end pipeline
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def toy_adata():
    import scanpy as sc
    from anndata import AnnData

    rng = np.random.default_rng(0)
    db = I.load_cellcommdb("human")
    extra = ["TGFB1", "TGFBR1", "TGFBR2", "IL6", "IL6R", "IL6ST",
             "TNF", "TNFRSF1A", "VEGFA", "FLT1", "KDR"]
    genes = sorted(set(db.signaling_genes()[:200]) | set(extra))
    n = 240
    ct = rng.choice(["Tcell", "Mono", "Fibro"], size=n)
    X = rng.gamma(0.3, 1.0, (n, len(genes))).astype(np.float32)
    gi = {g: i for i, g in enumerate(genes)}
    for g in ["TGFB1", "IL6", "TNF", "VEGFA"]:
        X[ct == "Fibro", gi[g]] += 4
    for g in ["TGFBR1", "TGFBR2", "IL6R", "TNFRSF1A", "FLT1", "KDR"]:
        X[ct == "Mono", gi[g]] += 4
    adata = AnnData(X=X, obs=pd.DataFrame({"cell_type": pd.Categorical(ct)}),
                    var=pd.DataFrame(index=genes))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def test_pipeline(toy_adata):
    cc = I.CellComm(toy_adata, "cell_type", species="human", verbose=False)
    cc.identify_overexpressed()
    assert cc.LRsig is not None and len(cc.LRsig) > 0
    cc.compute_communication(nboot=10, seed=1)
    K = len(cc.group_names)
    assert cc.net["prob"].shape[:2] == (K, K)
    cc.aggregate()
    assert cc.net["count"].shape == (K, K)
    cc.compute_communication_pathway()
    assert len(cc.netP["pathways"]) > 0

    df = cc.subset_communication()
    assert {"source", "target", "ligand", "receptor", "prob", "pval"} <= set(df.columns)
    # biological expectation: strongest signal is Fibro -> Mono
    top = df.sort_values("prob", ascending=False).iloc[0]
    assert top["source"] == "Fibro" and top["target"] == "Mono"


def test_pipeline_requires_order(toy_adata):
    cc = I.CellComm(toy_adata, "cell_type", verbose=False)
    with pytest.raises(RuntimeError):
        cc.compute_communication(nboot=5)  # before identify_overexpressed


def test_bad_groupby(toy_adata):
    with pytest.raises(KeyError):
        I.CellComm(toy_adata, "no_such_column", verbose=False)


# --------------------------------------------------------------------------
# analysis + viz smoke tests
# --------------------------------------------------------------------------
def test_centrality_and_roles(toy_adata):
    cc = I.CellComm(toy_adata, "cell_type", verbose=False)
    cc.identify_overexpressed().compute_communication(nboot=10).aggregate()
    cen = I.compute_centrality(cc.net["weight"], cc.group_names)
    assert set(cen.columns) >= {"outdeg", "indeg", "flowbet", "info"}
    roles = I.signaling_role(cen)
    assert "role" in roles.columns
    # Fibro should be the top sender
    assert cen["outdeg"].idxmax() == "Fibro"


def test_viz_smoke(toy_adata):
    cc = I.CellComm(toy_adata, "cell_type", verbose=False)
    cc.identify_overexpressed().compute_communication(nboot=10).aggregate()
    df = cc.subset_communication()
    assert I.plot_network_circle(cc.net["weight"], cc.group_names) is not None
    assert I.plot_heatmap(cc.net["weight"], cc.group_names) is not None
    assert I.plot_chord(cc.net["weight"], cc.group_names) is not None
    if len(df):
        assert I.plot_bubble(df.head(20)) is not None


# --------------------------------------------------------------------------
# CRITICAL + IMPORTANT features (filter, rank, contribution, NMF, manifold)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cc_full(toy_adata):
    """A CellChat run through the full inference pipeline for feature tests."""
    cc = I.CellComm(toy_adata, "cell_type", verbose=False)
    cc.identify_overexpressed().compute_communication(nboot=10, seed=1)
    cc.compute_communication_pathway().aggregate()
    return cc


def test_filter_communication(toy_adata):
    cc = I.CellComm(toy_adata, "cell_type", verbose=False)
    cc.identify_overexpressed().compute_communication(nboot=10)
    # filtering with a huge min_cells zeros everything
    cc.filter_communication(min_cells=10_000)
    assert (cc.net["prob"] > 0).sum() == 0


def test_rank_net(cc_full):
    rk = cc_full.rank_net()
    assert list(rk.columns) == ["pathway_name", "flow"]
    # sorted descending
    assert (rk["flow"].to_numpy()[:-1] >= rk["flow"].to_numpy()[1:]).all()


def test_contribution(cc_full):
    top = cc_full.rank_net().iloc[0]["pathway_name"]
    con = cc_full.contribution(top)
    assert abs(con["contribution"].sum() - 1.0) < 1e-9
    with pytest.raises(ValueError):
        cc_full.contribution("NOT_A_PATHWAY")


def test_signaling_role_pathway(cc_full):
    rm = I.signaling_role_pathway(cc_full, pattern="outgoing")
    assert rm.shape[0] == len(cc_full.netP["pathways"])
    assert rm.shape[1] == len(cc_full.group_names)
    with pytest.raises(ValueError):
        I.signaling_role_pathway(cc_full, pattern="nonsense")


def test_nmf_patterns(cc_full):
    n_path = len(cc_full.netP["pathways"])
    if n_path < 2:
        pytest.skip("need >=2 pathways")
    k = min(2, n_path, len(cc_full.group_names))
    res = I.identify_communication_patterns(cc_full, pattern="outgoing", k=k, verbose=False)
    assert res["W"].shape[1] == k
    assert res["H"].shape[0] == k


def test_manifold(cc_full):
    n_path = len(cc_full.netP["pathways"])
    if n_path < 3:
        pytest.skip("manifold needs >=3 pathways")
    sim = I.compute_net_similarity(cc_full, type="functional")
    assert sim.shape == (n_path, n_path)


# --------------------------------------------------------------------------
# spatial constraints
# --------------------------------------------------------------------------
def test_compute_region_distance():
    import numpy as np
    # two tight clusters far apart on a line
    coords = np.array([[0, 0], [1, 0], [0, 1],          # group A near origin
                       [100, 0], [101, 0], [100, 1]])   # group B ~100 units away
    groups = ["A", "A", "A", "B", "B", "B"]
    res = I.compute_region_distance(
        coords, groups, interaction_range=250, ratio=1.0, tol=5.0,
        k_min=1, contact_dependent=False,
    )
    d = res.d_spatial
    names = res.group_names
    ai, bi = names.index("A"), names.index("B")
    # within-range (250): A-B distance finite and ~100
    assert np.isfinite(d[ai, bi])
    assert 95 < d[ai, bi] < 105
    # symmetric
    assert np.isclose(d[ai, bi], d[bi, ai])


def test_compute_region_distance_out_of_range():
    import numpy as np
    coords = np.array([[0, 0], [1, 0], [500, 0], [501, 0]])
    groups = ["A", "A", "B", "B"]
    res = I.compute_region_distance(
        coords, groups, interaction_range=100, ratio=1.0, tol=5.0,
        k_min=1, contact_dependent=False,
    )
    names = res.group_names
    ai, bi = names.index("A"), names.index("B")
    # ~500 units apart, beyond interaction_range=100 → NaN
    assert np.isnan(res.d_spatial[ai, bi])


def test_spatial_probability_factor():
    import numpy as np
    d = np.array([[10.0, 200.0], [200.0, 10.0]])
    P = I.spatial_probability_factor(d, scale_distance=0.01)
    # closer pair (smaller distance) → larger probability weight
    assert P[0, 1] == P[1, 0]  # symmetric
    # diagonal set to max
    assert P[0, 0] == P.max()
    assert (P >= 0).all()


def test_spatial_end_to_end():
    import numpy as np
    import scanpy as sc
    from anndata import AnnData
    rng = np.random.default_rng(0)
    db = I.load_cellcommdb("human").subset(annotation="Secreted Signaling")
    extra = ["TGFB1", "TGFBR1", "TGFBR2", "IL6", "IL6R", "IL6ST"]
    genes = sorted(set(db.signaling_genes()[:150]) | set(extra))
    n = 120
    # two spatially separated groups
    coords = np.vstack([rng.normal([0, 0], 5, (n // 2, 2)),
                        rng.normal([50, 50], 5, (n // 2, 2))])
    ct = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    X = rng.gamma(0.3, 1.0, (n, len(genes))).astype(np.float32)
    adata = AnnData(X=X, obs=pd.DataFrame({"ct": pd.Categorical(ct)}),
                    var=pd.DataFrame(index=genes))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    cc = I.CellComm(adata, "ct", db=db, coordinates=coords,
                    spatial_factors={"ratio": 1.0, "tol": 5.0}, verbose=False)
    assert cc.is_spatial
    cc.identify_overexpressed()
    cc.compute_communication(nboot=5, seed=1, interaction_range=250,
                             contact_dependent=False)
    K = len(cc.group_names)
    assert cc.net["prob"].shape[:2] == (K, K)


# --------------------------------------------------------------------------
# multi-dataset comparison
# --------------------------------------------------------------------------
def _toy_cc(seed, adata):
    cc = I.CellComm(adata, "cell_type", verbose=False)
    cc.identify_overexpressed().compute_communication(nboot=5, seed=seed)
    cc.compute_communication_pathway().aggregate()
    return cc


def test_multicellchat(toy_adata):
    cc1 = _toy_cc(1, toy_adata)
    cc2 = _toy_cc(2, toy_adata)
    mcc = I.MultiCellComm([cc1, cc2], names=["A", "B"])

    ci = mcc.compare_interactions("count")
    assert list(ci.columns) == ["dataset", "value"]
    assert set(ci["dataset"]) == {"A", "B"}

    diff = mcc.diff_interaction("weight", comparison=(0, 1))
    K = len(cc1.group_names)
    assert diff.shape == (K, K)

    rn = mcc.rank_net(measure="weight", do_stat=True)
    assert {"name", "contribution", "group"} <= set(rn.columns)
    assert "pvalues" in rn.columns  # do_stat=True


def test_multicellchat_requires_two(toy_adata):
    with pytest.raises(ValueError):
        I.MultiCellComm([_toy_cc(1, toy_adata)], names=["only"])


def test_multicellchat_group_mismatch(toy_adata):
    cc1 = _toy_cc(1, toy_adata)
    # a second dataset with a DIFFERENT (renamed) group set but still multi-group
    a2 = toy_adata.copy()
    remap = {"Tcell": "Tcell2", "Mono": "Mono", "Fibro": "Fibro"}
    a2.obs["cell_type"] = pd.Categorical(
        [remap[c] for c in a2.obs["cell_type"].astype(str)]
    )
    cc2 = _toy_cc(2, a2)
    with pytest.raises(ValueError):
        I.MultiCellComm([cc1, cc2], names=["A", "B"])


# --------------------------------------------------------------------------
# remaining visualizations (viz2) smoke tests
# --------------------------------------------------------------------------
def test_viz2_smoke(cc_full):
    # role scatter / network work off aggregate + centrality
    assert I.plot_signaling_role_scatter(cc_full) is not None
    assert I.plot_signaling_role_network(cc_full) is not None
    assert I.plot_chord_gene(cc_full) is not None
    # aggregate dispatcher on the top-ranked pathway
    top = cc_full.rank_net().iloc[0]["pathway_name"]
    assert I.plot_aggregate(cc_full, top) is not None


def test_viz2_patterns(cc_full):
    n_path = len(cc_full.netP["pathways"])
    if n_path < 2:
        pytest.skip("need >=2 pathways")
    k = min(2, n_path, len(cc_full.group_names))
    res = I.identify_communication_patterns(cc_full, pattern="outgoing", k=k, verbose=False)
    assert I.plot_communication_patterns_river(cc_full, pattern="outgoing", result=res) is not None
    assert I.plot_communication_patterns_dot(cc_full, pattern="outgoing", result=res) is not None


def test_viz2_embedding_zoom(cc_full):
    n_path = len(cc_full.netP["pathways"])
    if n_path < 3:
        pytest.skip("manifold needs >=3 pathways")
    I.compute_net_similarity(cc_full, type="functional")
    I.net_embedding(cc_full, type="functional")
    I.net_clustering(cc_full, type="functional")
    assert I.plot_net_embedding_zoom(cc_full, type="functional") is not None
