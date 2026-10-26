from __future__ import annotations
from typing import Optional, Sequence, Union, List

def us(adata,gene,groups=None, show=False, exclude =None,figsize=None,**kwargs):
    """
    * 03/10/2022
    Create a umap using a list of genes.

    adata:AnnData,         REQUIRED | AnnData object.
    gene:list/str,         REQUIRED | List of genes to use for UMAP. A coma seperated string can be used instead of a list
    groups:str,        NOT REQUIRED | Restrict to a few categories in categorical observation annotation
    show:boolean,      NOT REQUIRED | Show the plot. Default = False.
    exclude:list,      NOT REQUIRED | List of genes to exclude. 
    figsize:float,     NOT REQUIRED | Figure size.
    """
    import scanpy as sc

    if isinstance(gene, str) and ',' in gene:
        gene = gene.split(',')
    # user-supplied cmap/color_map takes precedence over the OrRd default
    cmap = kwargs.pop('cmap', None) or kwargs.pop('color_map', None) or 'OrRd'
    kwargs.pop('color_map', None)
    if groups:
        sc.pl.umap(adata, color=gene, color_map=cmap, groups=groups, show=show, **kwargs)
    else:
        if exclude:
            groups = [x for x in set(adata.obs[gene]) if x not in exclude]
            sc.pl.umap(adata, color=gene, color_map=cmap, groups=groups, show=show, **kwargs)
        else:
            sc.pl.umap(adata, color=gene, color_map=cmap, show=show, **kwargs)