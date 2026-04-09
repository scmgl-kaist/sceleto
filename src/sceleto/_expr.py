"""Expression matrix resolution: auto-detect log1p-normalized data."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _check_log1p(X, n_sample: int = 500) -> bool:
    """Check if a matrix looks like log1p-normalized data.

    Criteria:
    - No negative values
    - Max value < 15 (log1p(10000) ≈ 9.2, generous upper bound)
    """
    if sparse.issparse(X):
        # sample a block for speed
        n = min(n_sample, X.shape[0])
        sample = X[:n].toarray()
    else:
        n = min(n_sample, X.shape[0])
        sample = np.asarray(X[:n])

    if np.any(sample < 0):
        return False
    if sample.max() > 15:
        return False
    return True


def resolve_expression(adata):
    """Determine the appropriate expression matrix (X or raw.X).

    Logic:
    1. If adata.X is log1p-normalized → use X
    2. Else if adata.raw exists and raw.X is log1p-normalized → use raw.X
    3. Else → raise ValueError

    Returns
    -------
    X : sparse or dense matrix
    var_names : pd.Index
    source : str ("X" or "raw")
    """
    # Try X first
    if _check_log1p(adata.X):
        return adata.X, adata.var_names, "X"

    # Try raw
    if adata.raw is not None and _check_log1p(adata.raw.X):
        return adata.raw.X, adata.raw.var_names, "raw"

    # Neither works
    raise ValueError(
        "Neither adata.X nor adata.raw.X appear to be log1p-normalized. "
        "sceleto requires log1p-transformed expression data. "
        "Tip: run sc.pp.normalize_total(adata) then sc.pp.log1p(adata), "
        "or store log1p data in adata.raw before scaling."
    )
