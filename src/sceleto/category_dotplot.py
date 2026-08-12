"""Category dot-heatmap: a grouped dotplot colored per *category* level.

Reproduces cross-species / cross-condition marker panels (e.g. Sepp *et al.*
cerebellum Fig. 2B) where rows are split into ``groupby`` blocks (cell types),
each block holds one sub-row per ``category`` level (species / condition /
sample), the dot **color** is drawn from *that category's own colormap*, and the
dot **size** encodes fraction of expressing cells.

This is a thin wrapper around
:class:`PyComplexHeatmap.DotClustermapPlotter` — the same tool the reference
figure was made with — which natively supports per-``hue`` colormaps, row/column
splitting into blocks, and colored annotation sidebars.  ``PyComplexHeatmap`` is
an optional dependency: ``pip install "sceleto[dotplot]"``.

Usage
-----
>>> import sceleto as scl
>>> scl.category_dotplot(adata, ['CD3D', 'CD8A', 'MS4A1'],
...                      groupby='cell_type', category='species')
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from scipy import sparse

from .dotplot import _resolve_var_names, _check_log1p_normalized


_SEP = " | "

# sequential colormaps cycled across category levels (ref1: Oranges/Greens/Greys)
_DEFAULT_CMAPS = [
    "Oranges", "Greens", "Greys", "Purples", "Blues", "Reds",
    "YlOrBr", "BuGn", "RdPu", "PuBu",
]


def _levels(series: pd.Series, requested):
    if requested is not None:
        return [str(x) for x in requested]
    if isinstance(series.dtype, pd.CategoricalDtype):
        return [str(x) for x in series.cat.categories]
    return [str(x) for x in pd.unique(series.astype(str))]


def _seq_cmap(base_color):
    """Light-grey → *base_color* sequential colormap."""
    r, g, b = to_rgb(base_color)
    return LinearSegmentedColormap.from_list("_scl_cat", [(0.93, 0.93, 0.93), (r, g, b)])


def _resolve_category_cmaps(cat_levels, category_cmaps):
    if category_cmaps is None:
        return {c: _DEFAULT_CMAPS[i % len(_DEFAULT_CMAPS)]
                for i, c in enumerate(cat_levels)}
    if isinstance(category_cmaps, Mapping):
        missing = [c for c in cat_levels if c not in category_cmaps]
        if missing:
            raise ValueError(f"category_dotplot: no colormap for {missing}.")
        return {c: category_cmaps[c] for c in cat_levels}
    seq = list(category_cmaps)
    if len(seq) < len(cat_levels):
        raise ValueError(
            f"category_dotplot: need >= {len(cat_levels)} colormaps, got {len(seq)}."
        )
    return {c: seq[i] for i, c in enumerate(cat_levels)}


def _uns_colors(adata, col, levels):
    """{level: color} from ``adata.uns[f'{col}_colors']`` or ``None``."""
    key = f"{col}_colors"
    s = adata.obs[col]
    if key not in adata.uns or not isinstance(s.dtype, pd.CategoricalDtype):
        return None
    order = list(s.cat.categories.astype(str))
    colors = list(adata.uns[key])
    if len(colors) < len(order):
        return None
    lut = dict(zip(order, colors))
    return {lv: lut[lv] for lv in levels if lv in lut}


def category_dotplot(
    adata,
    var_names: Union[Sequence[str], Mapping[str, Sequence]],
    groupby: str,
    category: str,
    *,
    groups: Optional[Sequence[str]] = None,
    categories: Optional[Sequence[str]] = None,
    use_raw: bool = True,
    drop_empty: bool = False,
    max_scale: bool = False,
    standard_scale=None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    category_cmaps=None,
    group_colors=None,
    row_split: bool = True,
    row_split_gap: float = 1.5,
    col_split_gap: float = 0.3,
    sidebar: bool = True,
    show_group_label: bool = True,
    max_s: float = 130.0,
    min_s: float = 0.0,
    dot_size: str = "pct",
    size_range=None,
    grid: bool = False,
    legend_width: Optional[float] = None,
    cmap_legend_kws: Optional[dict] = None,
    color_legend_kws: Optional[dict] = None,
    dot_legend_kws: Optional[dict] = None,
    figsize: Optional[Tuple[float, float]] = None,
    show_rownames: bool = False,
    save: Optional[str] = None,
    show: bool = True,
    **kwargs,
):
    """Grouped dot-heatmap with per-``category`` colormaps (ref1 fig2B style).

    Rows = ``groupby`` blocks × ``category`` sub-rows; columns = genes
    (optionally bracket-split via a mapping).  Dot color is each category's own
    colormap applied to mean expression; dot size is fraction of expressing cells.

    Parameters
    ----------
    adata
        AnnData with log1p-normalized expression.
    var_names
        Gene list or ``{group_label: [gene, ...]}`` mapping.  A mapping splits
        the x-axis into gene-groups (``col_split``).
    groupby
        ``adata.obs`` column for the primary row blocks (e.g. cell type).
    category
        ``adata.obs`` column whose levels become colored sub-rows (e.g. species).
    groups, categories
        Optional ordered subsets of the ``groupby`` / ``category`` levels.
    use_raw
        Read expression from ``adata.raw.X`` (default) else ``adata.X``.
    drop_empty
        If ``False`` (default) show the full ``groupby``×``category`` grid
        (empty combos are blank rows); if ``True`` hide empty combos.
    max_scale
        If ``True``, color = per-gene max-normalized group mean → 0..1 (no
        negatives), like ``sceleto.dotplot(max_scale=True)``.  The max is taken
        *within each category* (each category has its own colormap, so its dots
        span its own 0..1), not across categories.  Default ``False``.
    standard_scale
        Color scaling of mean expression: ``"var"`` (z per gene) / ``"group"``
        (z per row) / ``None`` or ``False`` (default → **raw log1p mean**).
        Mutually exclusive with ``max_scale``.
    vmin, vmax
        Color limits.  ``None`` (default) auto-selects: data range for raw mean,
        ``0..1`` for ``max_scale``, ``-2..2`` for ``standard_scale``.
    category_cmaps
        Per-category colormaps: dict/list of matplotlib sequential colormap names
        (or Colormaps).  If ``None`` (default) and ``adata.uns[f'{category}_colors']``
        exists, a light→uns-color gradient is built per level; otherwise cycles
        Oranges/Greens/Greys/Purples/Blues/….
    group_colors
        ``{level: color}`` for the ``groupby`` sidebar.  Default uses
        ``adata.uns[f'{groupby}_colors']`` when present.
    row_split
        Split rows into ``groupby`` blocks (with ``row_split_gap``).
    sidebar
        Draw the left annotation bars (``groupby`` + ``category`` colors).
    show_group_label
        Write each ``groupby`` (cell type) name once, centred on its block, at
        the far left (default ``True``).
    max_s
        Largest dot area.
    min_s
        Smallest dot area (default ``0`` → a value at the low end of the size
        scale draws no dot, like ``scanpy.pl.dotplot``).  PyComplexHeatmap would
        otherwise floor it at ``max_s * 0.1``; raise ``min_s`` to keep low values
        faintly visible.
    dot_size
        ``"pct"`` (fraction expressing, default) or ``"mean"`` (mean expression).
    size_range
        ``(low, high)`` fixing the dot-size scale.  By default (``None``) the low
        end is anchored at 0 and, for ``dot_size="pct"``, the high end is the
        observed maximum rounded up to the next 10 % (the ``scanpy.pl.dotplot``
        convention) — so the largest dot = the top value in *this* plot and the
        size legend reads in round numbers.  Pass ``size_range=(0, 100)`` to make
        dot area an absolute fraction (comparable across plots); values are
        clipped to the range.
    grid
        Draw grid lines (default ``False``).
    legend_width
        Legend area width in mm; ``None`` (default) auto-sizes from the longest
        category label so legend text is not clipped.
    cmap_legend_kws, color_legend_kws, dot_legend_kws
        Overrides merged onto the built-in "pretty" defaults (rectangular
        colorbar ``extend='neither'``; discrete legends drawn without a box
        outline ``frameon=False``).
    figsize
        Manual ``(width, height)``; auto-sized (compact) otherwise.
    show_rownames
        Show the ``group | category`` label on every sub-row.
    save
        Path to save (PDF, dpi=300).
    show
        Call ``plt.show()``.
    **kwargs
        Forwarded to ``PyComplexHeatmap.DotClustermapPlotter`` (then to
        ``matplotlib.scatter``).  Dots get a thin outline by default
        (``edgecolors="black"``, ``linewidth=0.5``); override e.g. with
        ``edgecolors="none"`` / ``linewidth=0`` to remove it.  x tick labels
        rotate 90° by default (``xticklabels_kws={"labelrotation": 90}``);
        note PyComplexHeatmap honours ``labelrotation``, not ``rotation``.

    Returns
    -------
    ``(fig, DotClustermapPlotter)``.
    """
    try:
        import PyComplexHeatmap as pch
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "category_dotplot requires PyComplexHeatmap. "
            'Install with: pip install "sceleto[dotplot]"'
        ) from e

    # ── expression source & genes ─────────────────────────────────────
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True but adata.raw is None.")
        src, src_names = adata.raw, list(adata.raw.var_names)
    else:
        src, src_names = adata, list(adata.var_names)

    available = set(src_names)
    var_group_dict, genes = _resolve_var_names(var_names, available)
    if not genes:
        raise ValueError("category_dotplot: none of the provided genes are present.")
    genes = list(dict.fromkeys(genes))
    gene_to_split = None
    if var_group_dict is not None:
        gene_to_split = {}
        for label, gs in var_group_dict.items():
            for g in gs:
                gene_to_split.setdefault(g, str(label))

    for col in (groupby, category):
        if col not in adata.obs:
            raise ValueError(f"category_dotplot: obs column {col!r} not found.")

    row_levels = _levels(adata.obs[groupby], groups)
    cat_levels = _levels(adata.obs[category], categories)

    # ── per (group, category) mean & pct → long dataframe ─────────────
    gi = np.array([src_names.index(g) for g in genes])
    X = src.X[:, gi]
    _check_log1p_normalized(X, "adata.raw.X" if use_raw else "adata.X")

    rlab = adata.obs[groupby].astype(str).to_numpy()
    clab = adata.obs[category].astype(str).to_numpy()

    records, y_full = [], []
    for r in row_levels:
        for c in cat_levels:
            y = f"{r}{_SEP}{c}"
            y_full.append((y, r, c))
            m = (rlab == r) & (clab == c)
            if not m.any():
                continue
            sub = X[m]
            if sparse.issparse(sub):
                mean = np.asarray(sub.mean(0)).ravel()
                pct = np.asarray((sub > 0).sum(0)).ravel() / sub.shape[0] * 100
            else:
                sub = np.asarray(sub)
                mean = sub.mean(0)
                pct = (sub > 0).mean(0) * 100
            for k, g in enumerate(genes):
                records.append((y, r, c, g, float(pct[k]), float(mean[k])))

    if not records:
        raise ValueError("category_dotplot: no cells for the requested groups/categories.")
    df = pd.DataFrame(records, columns=["y", groupby, category, "gene", "pct", "mean"])

    # ── color scaling ─────────────────────────────────────────────────
    # default (max_scale=False, standard_scale falsy): raw log1p mean expression.
    if max_scale and standard_scale:
        raise ValueError("set only one of max_scale / standard_scale.")
    if max_scale:
        # per-gene max-normalized group mean → 0..1, no negatives (like scl.dotplot).
        # Normalize *within each category* (not globally): each category has its
        # own colormap, so its dots should span 0..1 against that category's own
        # max for the gene.
        gmax = df.groupby(["gene", category])["mean"].transform("max").to_numpy()
        df["color"] = np.where(gmax > 0, df["mean"].to_numpy() / gmax, 0.0)
        color_vmin = 0.0 if vmin is None else vmin
        color_vmax = 1.0 if vmax is None else vmax
    elif standard_scale in ("var", "group"):
        key = "gene" if standard_scale == "var" else "y"
        df["color"] = df.groupby(key)["mean"].transform(
            lambda v: (v - v.mean()) / (v.std() + 1e-9))
        color_vmin = -2.0 if vmin is None else vmin
        color_vmax = 2.0 if vmax is None else vmax
    elif not standard_scale:  # None / False → raw log1p mean
        df["color"] = df["mean"]
        color_vmin, color_vmax = vmin, vmax  # None → auto from data
    else:
        raise ValueError("standard_scale must be 'var', 'group', False, or None.")

    size_col = {"pct": "pct", "mean": "mean"}.get(dot_size)
    if size_col is None:
        raise ValueError("dot_size must be 'pct' or 'mean'.")

    # ── ordering ──────────────────────────────────────────────────────
    have = set(df["y"])
    y_order = [y for (y, r, c) in y_full if (y in have or not drop_empty)]
    ann = pd.DataFrame(index=y_order)
    ann[groupby] = [y.split(_SEP)[0] for y in y_order]
    ann[category] = [y.split(_SEP)[1] for y in y_order]

    # category colormaps: prefer adata.uns[f'{category}_colors'] (build a
    # light→uns-color gradient per level); else the default sequential palette.
    if category_cmaps is None:
        uns_cat = _uns_colors(adata, category, cat_levels)
        if uns_cat:
            category_cmaps = {c: _seq_cmap(uns_cat[c]) for c in cat_levels}
    cat_cmaps = _resolve_category_cmaps(cat_levels, category_cmaps)
    cat_solid = {c: plt.get_cmap(cat_cmaps[c])(0.7) if isinstance(cat_cmaps[c], str)
                 else cat_cmaps[c](0.7) for c in cat_levels}
    if group_colors is None:
        group_colors = _uns_colors(adata, groupby, row_levels)

    # ── pretty defaults (all overridable) ─────────────────────────────
    # tighter columns by default; legend wide enough to avoid clipping.
    if figsize is None:
        figsize = (max(4.5, 0.14 * len(genes) + 2.8),
                   max(3.0, 0.20 * len(y_order) + 1.5))
    if legend_width is None:
        longest = max([len(str(x)) for x in (cat_levels + [category])] or [8])
        legend_width = max(30.0, longest * 2.6)
    # rectangular colorbar; ticks=None → matplotlib AutoLocator picks round
    # numbers (the scanpy.pl.dotplot look) instead of raw vmin/center/vmax.
    cmap_lk = {"extend": "neither", "ticks": None}
    cmap_lk.update(cmap_legend_kws or {})
    color_lk = {"frameon": False}            # no legend-box outline
    color_lk.update(color_legend_kws or {})
    dot_lk = {"frameon": False}
    dot_lk.update(dot_legend_kws or {})

    # dot outline defaults (overridable via kwargs). Drop a default if the user
    # passes the singular alias so matplotlib doesn't error on both being set.
    dot_style = {"edgecolors": "black", "linewidths": 0.5}
    if any(k in kwargs for k in ("edgecolor", "ec", "edgecolors")):
        dot_style.pop("edgecolors", None)
    if any(k in kwargs for k in ("linewidth", "lw", "linewidths")):
        dot_style.pop("linewidths", None)

    fig = plt.figure(figsize=figsize)

    left_annotation = None
    if sidebar:
        annos = {}
        # groupby (cell type) text label, one per block, on the far left
        if show_group_label:
            annos[" "] = pch.anno_label(
                ann[groupby], merge=True, rotation=0, extend=True,
                colors={lv: "black" for lv in row_levels}, adjust_color=False,
                arrowprops={"visible": False})
        # groupby color bar
        gb_kw = dict(height=4, legend=False)
        if group_colors:
            gb_kw["colors"] = group_colors
        annos[groupby] = pch.anno_simple(ann[groupby], **gb_kw)
        # category color bar (colored sub-rows)
        annos[category] = pch.anno_simple(ann[category], colors=cat_solid,
                                          height=4, legend=True,
                                          legend_kws={"frameon": False})
        left_annotation = pch.HeatmapAnnotation(
            axis=0, orientation="left", label_side="top", legend=True, **annos)

    split_series = (pd.Series(ann[groupby].values, index=y_order)
                    if row_split else None)
    col_split = (pd.Series(gene_to_split) if gene_to_split is not None else None)

    plot_kwargs = dict(
        data=df, x="gene", y="y", hue=category, value="color",
        s=size_col, c="color", cmap=cat_cmaps, max_s=max_s, min_s=min_s,
        left_annotation=left_annotation,
        x_order=genes, y_order=y_order,
        row_cluster=False, col_cluster=False,
        show_rownames=show_rownames, show_colnames=True,
        grid=grid, legend_width=legend_width,
        cmap_legend_kws=cmap_lk, color_legend_kws=color_lk, dot_legend_kws=dot_lk,
        xticklabels_kws={"labelrotation": 90}, verbose=0,
        **dot_style,
    )
    if split_series is not None:
        plot_kwargs.update(row_split=split_series, row_split_order=row_levels,
                           row_split_gap=row_split_gap)
    if col_split is not None:
        split_order = [str(k) for k in var_group_dict.keys()]
        plot_kwargs.update(col_split=col_split, col_split_order=split_order,
                           col_split_gap=col_split_gap)
    if color_vmin is not None:
        plot_kwargs["vmin"] = color_vmin
    if color_vmax is not None:
        plot_kwargs["vmax"] = color_vmax
    plot_kwargs.update(kwargs)

    # Dot-size scale.  PyComplexHeatmap normalizes size to the data's own
    # min–max (so the smallest dot floats above 0).  Instead we anchor the low
    # end at 0 and, for pct, round the high end up to the next 10 % — the
    # scanpy.pl.dotplot convention — so the legend reads nicely and the largest
    # dot = the top value in this plot.  size_range=(low, high) overrides this
    # with a fixed reference (e.g. 0–100 for absolute, cross-plot fractions).
    if size_range is not None:
        if len(size_range) != 2 or size_range[0] >= size_range[1]:
            raise ValueError("size_range must be (low, high) with low < high.")
        srange = (float(size_range[0]), float(size_range[1]))
    else:
        smax_data = float(df[size_col].max())
        if smax_data <= 0:
            srange = None
        elif dot_size == "pct":
            srange = (0.0, min(100.0, float(np.ceil(smax_data / 10.0) * 10.0)))
        else:
            srange = (0.0, smax_data)

    if srange is None:
        cm = pch.DotClustermapPlotter(**plot_kwargs)
    else:
        lo, hi = srange

        # override smin/smax with the fixed reference so the dot area maps to an
        # absolute value and the size legend spans lo..hi.
        class _FixedSizeDotPlotter(pch.DotClustermapPlotter):
            def format_data(self, data, mask=None, z_score=None, standard_scale=None):
                data2d = super().format_data(
                    data, mask=mask, z_score=z_score, standard_scale=standard_scale)
                if isinstance(self.s, str):
                    raw = data.pivot_table(
                        index=self.y, columns=self.x, values=self.s,
                        aggfunc=self.aggfunc).fillna(self.s_na)
                    self.smin, self.smax = lo, hi
                    self.kwargs["s"] = raw.clip(lo, hi).map(lambda v: (v - lo) / (hi - lo))
                return data2d

            def collect_legends(self):
                super().collect_legends()
                # PyComplexHeatmap labels the dot-size legend with str(round(v, 2))
                # → "100.0".  Show pct as whole numbers (like scanpy.pl.dotplot);
                # for other size metrics just drop trailing zeros.
                key = f"{self.s} (dot)"
                entry = getattr(self, "legend_dict", {}).get(key)
                if entry is not None:
                    markers1, mid, ms = entry[0]

                    def _fmt(k):
                        try:
                            v = float(k)
                        except (TypeError, ValueError):
                            return k
                        return f"{v:.0f}" if dot_size == "pct" else f"{v:g}"

                    self.legend_dict[key] = ((
                        {_fmt(k): v for k, v in markers1.items()},
                        mid,
                        {_fmt(k): v for k, v in ms.items()},
                    ),) + tuple(entry[1:])
                    self.get_legend_list()

        cm = _FixedSizeDotPlotter(**plot_kwargs)

    if save:
        fig.savefig(save, bbox_inches="tight", format="pdf", dpi=300)
    if show:
        plt.show()
    return fig, cm
