"""Expression-aggregation kernels for communication probability.

Faithful NumPy ports of CellChat's per-group expression math (``modeling.R``):

- :func:`tri_mean` — Tukey's trimean, the default group-summary statistic.
- :func:`group_mean` — per-group summary of a cell×gene matrix.
- :func:`complex_expr` — multi-subunit ligand/receptor expression via the
  geometric mean of subunit expression.
- :func:`lr_expr` — ligand/receptor expression (single gene or complex).
- :func:`coreceptor_factor` — co-activation / co-inhibition receptor modulation.
- :func:`agonist_factor` / :func:`antagonist_factor` — Hill-function cofactor terms.

Everything works on a ``genes × groups`` average-expression matrix indexed by
gene symbol (a :class:`pandas.DataFrame`), matching CellChat's ``data.use.avg``.
Genes absent from the matrix are treated as not expressed, exactly as the R code
does by intersecting against ``rownames(data.use)``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

_SUBUNIT_PREFIX = "subunit"
_COFACTOR_PREFIX = "cofactor"


# ---------------------------------------------------------------------------
# group summary statistics
# ---------------------------------------------------------------------------
def tri_mean(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Tukey's trimean: ``(Q1 + 2*median + Q3) / 4``.

    Equivalent to CellChat's ``triMean`` =
    ``mean(quantile(x, c(.25, .50, .50, .75)))``.

    Parameters
    ----------
    x
        Values.
    axis
        Axis to reduce.
    """
    # np.quantile is vastly faster than np.nanquantile (the nan version falls
    # back to a per-column apply_along_axis). Expression matrices here are
    # NaN-free (X/max of non-negative counts), so use the fast path when there
    # are no NaNs; the two are identical in that case.
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():
        q1, med, q3 = np.nanquantile(x, [0.25, 0.5, 0.75], axis=axis)
    else:
        q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75], axis=axis)
    return (q1 + 2.0 * med + q3) / 4.0


def _reducer(type: str, trim: float):
    """Return a ``(values, axis) -> summary`` reducer matching CellChat ``type``."""
    if type == "triMean":
        return lambda a, axis: tri_mean(a, axis=axis)
    if type == "median":
        return lambda a, axis: np.nanmedian(a, axis=axis)
    if type == "truncatedMean":
        return lambda a, axis: _trimmed_mean(a, trim, axis)
    if type == "thresholdedMean":
        return lambda a, axis: _thresholded_mean(a, trim, axis)
    raise ValueError(
        f"Unknown mean type {type!r}. "
        "Valid: 'triMean', 'truncatedMean', 'thresholdedMean', 'median'."
    )


def _trimmed_mean(a: np.ndarray, trim: float, axis: int) -> np.ndarray:
    from scipy.stats import trim_mean

    return trim_mean(a, proportiontocut=trim, axis=axis)


def _thresholded_mean(a: np.ndarray, trim: float, axis: int) -> np.ndarray:
    # CellChat thresholdedMean (modeling.R): a detection-rate gate — if the
    # fraction of NONZERO values along the axis is below `trim`, the summary is
    # 0; otherwise it is the plain mean of all values.
    #   percent = nnzero(x)/length(x); if (percent < trim) 0 else mean(x)
    a = np.asarray(a, dtype=float)
    n = a.shape[axis]
    frac_nonzero = np.count_nonzero(a, axis=axis) / n if n else np.zeros(a.shape[1 - axis])
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(a, axis=axis)
    return np.where(frac_nonzero < trim, 0.0, mean)


def group_mean(
    expr: np.ndarray,
    genes: Sequence[str],
    groups: pd.Categorical | pd.Series,
    *,
    type: str = "triMean",
    trim: float = 0.1,
) -> pd.DataFrame:
    """Per-group summary expression → ``genes × groups`` DataFrame.

    Parameters
    ----------
    expr
        Dense ``cells × genes`` matrix, already scaled by ``1/max`` upstream.
    genes
        Gene symbols labelling the columns of ``expr``.
    groups
        Per-cell group labels (a pandas Categorical or Series); the result
        columns follow the categorical level order.
    type, trim
        Summary statistic and trim fraction (see :func:`_reducer`).

    Returns
    -------
    pd.DataFrame
        Average expression indexed by gene, columns = group levels.
    """
    cat = groups if isinstance(groups, pd.Categorical) else pd.Categorical(groups)
    levels = list(cat.categories)
    reducer = _reducer(type, trim)
    codes = cat.codes
    out = np.empty((expr.shape[1], len(levels)), dtype=float)
    for j, _ in enumerate(levels):
        rows = codes == j
        block = expr[rows, :]
        out[:, j] = reducer(block, 0)
    return pd.DataFrame(out, index=list(genes), columns=levels)


# ---------------------------------------------------------------------------
# ligand / receptor / complex expression
# ---------------------------------------------------------------------------
def _geometric_mean(mat: np.ndarray, axis: int = 0) -> np.ndarray:
    """Geometric mean ``exp(mean(log(x)))`` along ``axis`` (NaNs skipped)."""
    with np.errstate(divide="ignore"):
        logs = np.log(mat)
    return np.exp(np.nanmean(logs, axis=axis))


def complex_expr(
    avg: pd.DataFrame,
    complex_names: Iterable[str],
    complex_table: pd.DataFrame,
) -> np.ndarray:
    """Expression of complexes = geometric mean of their subunits.

    Parameters
    ----------
    avg
        ``genes × groups`` average-expression matrix.
    complex_names
        Complex names to resolve (rows of ``complex_table``).
    complex_table
        The DB complex table, indexed by complex name, with ``subunit_*`` cols.

    Returns
    -------
    np.ndarray
        ``len(complex_names) × n_groups`` expression matrix.
    """
    n_groups = avg.shape[1]
    sub_cols = [c for c in complex_table.columns if c.startswith(_SUBUNIT_PREFIX)]
    rows = []
    for name in complex_names:
        if name in complex_table.index:
            subs = [str(complex_table.at[name, c]).strip() for c in sub_cols]
            subs = [s for s in subs if s and s in avg.index]
        else:
            subs = []
        if subs:
            rows.append(_geometric_mean(avg.loc[subs].to_numpy(), axis=0))
        else:
            rows.append(np.zeros(n_groups))
    return np.vstack(rows) if rows else np.empty((0, n_groups))


def lr_expr(
    genes_lr: Sequence[str],
    avg: pd.DataFrame,
    complex_table: pd.DataFrame,
) -> np.ndarray:
    """Ligand/receptor expression: single gene → its row, complex → geom. mean.

    Genes not present in ``avg`` and not a complex resolve to 0 (absent).

    Returns
    -------
    np.ndarray
        ``len(genes_lr) × n_groups`` expression matrix (row order = ``genes_lr``).
    """
    n_groups = avg.shape[1]
    out = np.zeros((len(genes_lr), n_groups), dtype=float)
    present = set(avg.index)
    complex_rows: list[int] = []
    complex_names: list[str] = []
    for i, g in enumerate(genes_lr):
        if g in present:
            out[i, :] = avg.loc[g].to_numpy()
        else:
            complex_rows.append(i)
            complex_names.append(g)
    if complex_names:
        cx = complex_expr(avg, complex_names, complex_table)
        for k, i in enumerate(complex_rows):
            out[i, :] = cx[k]
    return out


# ---------------------------------------------------------------------------
# cofactor modulation (co-receptor, agonist, antagonist)
# ---------------------------------------------------------------------------
def _cofactor_genes(name: str, cofactor_table: pd.DataFrame, present: set) -> list[str]:
    if not name or name not in cofactor_table.index:
        return []
    cols = [c for c in cofactor_table.columns if c.startswith(_COFACTOR_PREFIX)]
    genes = [str(cofactor_table.at[name, c]).strip() for c in cols]
    return [g for g in genes if g and g in present]


def coreceptor_factor(
    coreceptor_names: Sequence[str],
    avg: pd.DataFrame,
    cofactor_table: pd.DataFrame,
) -> np.ndarray:
    """Co-receptor modulation factor ``prod(1 + expr)`` per L-R pair.

    Returns 1 for pairs with no co-receptor. Matches
    ``computeExpr_coreceptor``: the receptor expression is later multiplied by
    the co-activation factor and divided by the co-inhibition factor.

    Returns
    -------
    np.ndarray
        ``len(coreceptor_names) × n_groups`` factor matrix.
    """
    n_groups = avg.shape[1]
    present = set(avg.index)
    out = np.ones((len(coreceptor_names), n_groups), dtype=float)
    for i, name in enumerate(coreceptor_names):
        genes = _cofactor_genes(str(name), cofactor_table, present)
        if genes:
            out[i, :] = np.prod(1.0 + avg.loc[genes].to_numpy(), axis=0)
    return out


def agonist_factor(
    agonist_name: str,
    avg: pd.DataFrame,
    cofactor_table: pd.DataFrame,
    *,
    Kh: float = 0.5,
    n: float = 1.0,
) -> np.ndarray:
    """Agonist term ``prod(1 + Hill(expr))`` over cofactor genes (per group).

    Returns a length-``n_groups`` vector (all ones if no agonist genes present).
    """
    present = set(avg.index)
    genes = _cofactor_genes(str(agonist_name), cofactor_table, present)
    if not genes:
        return np.ones(avg.shape[1])
    x = avg.loc[genes].to_numpy()
    hill = x**n / (Kh**n + x**n)
    return np.prod(1.0 + hill, axis=0)


def antagonist_factor(
    antagonist_name: str,
    avg: pd.DataFrame,
    cofactor_table: pd.DataFrame,
    *,
    Kh: float = 0.5,
    n: float = 1.0,
) -> np.ndarray:
    """Antagonist term ``prod(Kh^n / (Kh^n + expr^n))`` over cofactor genes.

    Returns a length-``n_groups`` vector (all ones if no antagonist genes present).
    """
    present = set(avg.index)
    genes = _cofactor_genes(str(antagonist_name), cofactor_table, present)
    if not genes:
        return np.ones(avg.shape[1])
    x = avg.loc[genes].to_numpy()
    return np.prod(Kh**n / (Kh**n + x**n), axis=0)
