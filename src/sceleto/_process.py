"""Processing functions ported from scjp."""

import numpy as np
import scanpy as sc


def _load_cc_genes():
    """Load cell cycle gene list from gene_categories.json."""
    import json
    from pathlib import Path
    path = Path(__file__).parent / "data" / "gene_categories.json"
    with open(path) as f:
        d = json.load(f)
    return d["Cell_Cycle"]


def remove_geneset(adata, geneset):
    """
    Remove the given geneset from adata.

    adata:AnnData,         REQUIRED | AnnData object.
    geneset:list,          REQUIRED | List of genes to be removed.
    """
    adata = adata[:, ~adata.var_names.isin(list(geneset))].copy()
    return adata


def sc_process(adata, pid='fspkuc', n_pcs=50):
    """
    Performs desired scanpy preprocessing according to the letters passed
    into the pid parameter.

    adata:AnnData,      REQUIRED | AnnData object.
    pid:str,            REQUIRED | A string made out of letters, each
                                   corresponding to a scanpy preprocessing
                                   function.
    n_pcs:int,      NOT REQUIRED | Number of PCs to be used for neighbor
                                   search. Default = 50

    Letters for pid and their corresponding function:

    n: normalise
    l: log
    f: filter hvg
    r: remove cc_genes
    s: scale
    p: pca
    k: knn_neighbors
    u: umap
    c: leiden clustering
    """
    if 'n' in pid:
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=10e4)
    if 'l' in pid:
        sc.pp.log1p(adata)
        adata.raw = adata
        print('adding raw...')
    if 'f' in pid:
        if adata.raw is None:
            adata.raw = adata
            print('adding raw...')
        sc.pp.filter_genes_dispersion(adata)
    if 'r' in pid:
        cc_genes = _load_cc_genes()
        adata = remove_geneset(adata, cc_genes)
        print('removing cc_genes...')
    if 's' in pid:
        sc.pp.scale(adata, max_value=10)
    if 'p' in pid:
        sc.pp.pca(adata)
    if 'k' in pid:
        sc.pp.neighbors(adata, n_pcs=n_pcs)
    if 'u' in pid:
        sc.tl.umap(adata)
    if 'c' in pid:
        sc.tl.leiden(adata)
    return adata


def read_process(
    adata,
    version,
    species='human',
    sample=None,
    define_var=True,
    call_doublet=True,
    write=True,
    min_n_counts=1000,
    min_n_genes=500,
    max_n_genes=7000,
    max_p_mito=0.5,
):
    """
    Used for reading and saving the data with desired cell filtration.

    adata:AnnData,            REQUIRED | AnnData object.
    version:str,              REQUIRED | The version of the h5ad file.
    species:str,              REQUIRED | The species which the data belongs
                                         to. Default = 'human'
    sample:str,               REQUIRED | Name of the sample the cells belong
                                         to. Default = None
    define_var:boolean,   NOT REQUIRED | Defines gene names if true and adds
                                         it to adata. Default = True
    call_doublet:boolean, NOT REQUIRED | Identifies doublet cells and stores
                                         it in adata. Default = True
    write:boolean,        NOT REQUIRED | Saves the file. Default = True
    min_n_counts:int,         REQUIRED | Minimum number of counts per cell.
                                         Default = 1000
    min_n_genes:int,          REQUIRED | Minimum number of genes per cell.
                                         Default = 500
    max_n_genes:int,          REQUIRED | Maximum number of genes per cell.
                                         Default = 7000
    max_p_mito:float,         REQUIRED | Maximum allowed ratio of
                                         mitochondrial genes. Default = 0.5
    """
    if sample:
        adata.obs['Sample'] = sample
    if define_var:
        adata.var['GeneName'] = list(adata.var.gene_ids.index)
        adata.var['EnsemblID'] = list(adata.var.gene_ids)
    adata.obs['n_counts'] = np.sum(adata.X, axis=1).A1
    adata.obs['n_genes'] = np.sum(adata.X > 0, axis=1).A1

    print('calculating mito... as species = {}'.format(species))
    if species == 'mouse':
        mito_genes = adata.var_names.str.startswith('mt-')
    elif species == 'human':
        mito_genes = adata.var_names.str.startswith('MT-')
    else:
        raise ValueError(f"Unknown species: {species}. Use 'human' or 'mouse'.")

    adata.obs['mito'] = (
        np.sum(adata.X[:, mito_genes], axis=1).A1
        / (np.sum(adata.X, axis=1).A1 + 1)
    )

    print(
        'filtering cells... higher than {} counts, more than {} and less '
        'than {} genes, less than {} p_mito...'.format(
            min_n_counts, min_n_genes, max_n_genes, max_p_mito
        )
    )
    # filter cells
    clist = []
    clist.append(np.array(adata.obs['n_counts'] > min_n_counts))
    clist.append(np.array(adata.obs['n_genes'] > min_n_genes))
    clist.append(np.array(adata.obs['n_genes'] < max_n_genes))
    clist.append(np.array(adata.obs['mito'] < max_p_mito))

    c = np.column_stack(clist).all(axis=1)
    adata = adata[c].copy()

    if call_doublet:
        import scrublet as scr
        print('calling doublets using scrublet...')
        scrub = scr.Scrublet(adata.X)
        doublet_scores, predicted_doublets = scrub.scrub_doublets(verbose=False)
        adata.obs['doublet_scores'] = doublet_scores
        adata.obs['predicted_doublets'] = predicted_doublets

    if write:
        print('writing output into write/%s%s_filtered.h5ad ...' % (version, sample))
        sc.write('%s%s_filtered' % (version, sample), adata)
    return adata
