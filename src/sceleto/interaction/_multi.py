"""Multi-dataset comparison — compare cell–cell communication across conditions.

Port of CellChat's comparison workflow (``mergeCellChat`` + ``compareInteractions``
+ ``rankNet`` comparison mode + ``netVisual_diffInteraction`` data layer). Given
two or more already-computed :class:`~sceleto.interaction.CellComm` objects (e.g.
disease vs normal), :class:`MultiCellComm` compares total communication, ranks
signaling pathways by differential information flow (with a Wilcoxon test), and
produces differential aggregated networks.

v1 assumes **all datasets share the same cell-group set** (same ``group_names``,
same order) — this makes the paired test valid and lets matrices be diffed
elementwise. (CellChat's ``liftCellChat`` for differing groups is not ported.)

Quickstart
----------
>>> mcc = MultiCellComm([cc_normal, cc_disease], names=["NL", "LS"])
>>> mcc.compare_interactions("count")         # total interactions per dataset
>>> mcc.rank_net(do_stat=True)                # per-pathway info flow + Wilcoxon
>>> mcc.diff_interaction("weight")            # KxK differential (LS - NL)
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


class MultiCellComm:
    """Compare communication across multiple :class:`CellComm` datasets.

    Parameters
    ----------
    objects
        The per-condition :class:`CellComm` objects, each already run through
        ``compute_communication`` (+ ``aggregate`` for count/weight comparisons,
        + ``compute_communication_pathway`` for pathway ranking).
    names
        Dataset names (e.g. ``["NL", "LS"]``). Defaults to ``Dataset1..N``.
    """

    def __init__(self, objects: Sequence, names: Optional[Sequence[str]] = None) -> None:
        self.objects = list(objects)
        if len(self.objects) < 2:
            raise ValueError("need at least 2 CellChat objects to compare")
        if names is None:
            names = [f"Dataset{i + 1}" for i in range(len(self.objects))]
        self.names = list(map(str, names))
        if len(self.names) != len(self.objects):
            raise ValueError("names must match the number of objects")
        # v1: require identical cell-group sets (same order) across datasets
        ref = list(self.objects[0].group_names)
        for nm, obj in zip(self.names, self.objects):
            if list(obj.group_names) != ref:
                raise ValueError(
                    f"dataset {nm!r} has different/reordered cell groups than "
                    f"{self.names[0]!r}; v1 requires identical group_names "
                    "(lifting to a joint group set is not supported)."
                )
        self.group_names = ref

    def _by_name(self, name):
        if name in self.names:
            return self.objects[self.names.index(name)]
        raise KeyError(f"unknown dataset {name!r}; have {self.names}")

    # -- total interactions per dataset (compareInteractions) -----------
    def compare_interactions(self, measure: str = "count") -> pd.DataFrame:
        """Total interactions (``count``) or strength (``weight``) per dataset.

        Requires :meth:`CellComm.aggregate` on each object.

        Returns
        -------
        pd.DataFrame
            Columns ``dataset``, ``value``.
        """
        if measure not in ("count", "weight"):
            raise ValueError("measure must be 'count' or 'weight'")
        rows = []
        for nm, obj in zip(self.names, self.objects):
            if measure not in obj.net:
                raise RuntimeError(f"call aggregate() on dataset {nm!r} first")
            v = float(obj.net[measure].sum())
            rows.append((nm, round(v, 3) if measure == "weight" else v))
        return pd.DataFrame(rows, columns=["dataset", "value"])

    # -- differential aggregated network --------------------------------
    def diff_interaction(
        self,
        measure: str = "count",
        *,
        comparison: tuple[int, int] = (0, 1),
        remove_isolate: bool = False,
        top: float = 1.0,
    ) -> pd.DataFrame:
        """Differential K×K matrix ``net[ds2] - net[ds1]`` (``netVisual_diffInteraction``).

        Positive entries = increased in the second dataset.

        Parameters
        ----------
        measure
            ``"count"`` or ``"weight"``.
        comparison
            ``(i, j)`` dataset indices; the result is ``ds_j - ds_i``.
        remove_isolate
            Drop groups that are all-zero in the differential.
        top
            Keep only the top fraction by magnitude (``1.0`` keeps all).

        Returns
        -------
        pd.DataFrame
            K×K differential matrix (rows=source, cols=target).
        """
        if measure not in ("count", "weight"):
            raise ValueError("measure must be 'count' or 'weight'")
        i, j = comparison
        a = self.objects[i].net.get(measure)
        b = self.objects[j].net.get(measure)
        if a is None or b is None:
            raise RuntimeError("call aggregate() on both datasets first")
        diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
        names = list(self.group_names)
        if top < 1.0:
            cutoff = np.quantile(np.abs(diff), 1.0 - top)
            diff = np.where(np.abs(diff) < cutoff, 0.0, diff)
        df = pd.DataFrame(diff, index=names, columns=names)
        if remove_isolate:
            keep = (df != 0).any(axis=1) | (df != 0).any(axis=0)
            df = df.loc[keep, keep]
        return df

    # -- rankNet comparison (per-pathway information flow) --------------
    def rank_net(
        self,
        *,
        measure: str = "weight",
        comparison: tuple[int, int] = (0, 1),
        do_stat: bool = False,
        paired_test: bool = True,
        thresh: float = 0.05,
        stacked: bool = False,
    ) -> pd.DataFrame:
        """Rank pathways by information flow across datasets (``rankNet`` comparison).

        For each dataset, information flow of a pathway = summed communication
        probability over all group pairs; for ``measure="weight"`` this is
        transformed as ``-1/log(x)`` (CellChat's "information flow"). Pathways
        absent in a dataset contribute 0.

        Parameters
        ----------
        measure
            ``"weight"`` (summed prob → info flow) or ``"count"`` (binarized —
            number of communicating group pairs per pathway).
        comparison
            ``(i, j)`` dataset indices to compare.
        do_stat
            If ``True`` (2 datasets only), a Wilcoxon test per pathway comparing
            the per-group-pair prob vectors of the two conditions.
        paired_test
            Paired signed-rank (default) vs unpaired rank-sum. Valid because
            datasets share the same K cell groups.
        thresh
            p-value threshold (only used when ranking the ``net`` slot; the
            default pathway-level ``netP`` has no p-values).
        stacked
            Add a per-pathway ``contribution.relative`` = fraction across datasets.

        Returns
        -------
        pd.DataFrame
            One row per (pathway, dataset): ``name``, ``contribution``,
            ``contribution.scaled``, ``group`` (dataset), ``contribution.relative.1``,
            and ``pvalues`` (if ``do_stat``).
        """
        if measure not in ("weight", "count"):
            raise ValueError("measure must be 'weight' or 'count'")
        idx = list(comparison)

        prob_list, raw_list, scaled_list, pw_list = [], [], [], []
        for di in idx:
            obj = self.objects[di]
            if not obj.netP or "prob" not in obj.netP:
                raise RuntimeError(
                    f"call compute_communication_pathway() on dataset "
                    f"{self.names[di]!r} first"
                )
            prob = np.asarray(obj.netP["prob"], dtype=float).copy()
            pathways = list(obj.netP["pathways"])
            if measure == "count":
                prob = (prob > 0).astype(float)
            raw = prob.sum(axis=(0, 1))  # per pathway
            if measure == "weight":
                with np.errstate(divide="ignore", invalid="ignore"):
                    scaled = -1.0 / np.log(raw)
                scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
                # degenerate x>=1 (log<=0) → clamp to 0 like the nan_to_num above;
                # CellChat reassigns synthetic ranks purely for plotting order.
                scaled[~np.isfinite(scaled)] = 0.0
                scaled[scaled < 0] = 0.0
            else:
                scaled = raw.copy()
            prob_list.append(prob)
            raw_list.append(dict(zip(pathways, raw)))
            scaled_list.append(dict(zip(pathways, scaled)))
            pw_list.append(pathways)

        # union of pathways (first-seen order)
        union = list(dict.fromkeys(pw for pws in pw_list for pw in pws))

        # per-dataset zero-filled contribution
        contrib = {di: np.array([raw_list[k].get(pw, 0.0) for pw in union])
                   for k, di in enumerate(idx)}
        scaled = {di: np.array([scaled_list[k].get(pw, 0.0) for pw in union])
                  for k, di in enumerate(idx)}

        # relative contribution ds2/ds1
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = contrib[idx[1]] / contrib[idx[0]]
        rel = np.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)

        # Wilcoxon per pathway
        pvals = None
        if do_stat and len(idx) == 2:
            pvals = self._wilcoxon_per_pathway(union, pw_list, prob_list, idx, paired_test)

        rows = []
        for k, di in enumerate(idx):
            nm = self.names[di]
            for p, pw in enumerate(union):
                row = {
                    "name": pw,
                    "contribution": abs(contrib[di][p]),
                    "contribution.scaled": abs(scaled[di][p]),
                    "group": nm,
                    "contribution.relative.1": float(f"{rel[p]:.1g}") if rel[p] else 0.0,
                }
                if pvals is not None:
                    row["pvalues"] = pvals[p]
                rows.append(row)
        df = pd.DataFrame(rows)

        # drop pathways whose total contribution across datasets is 0
        totals = df.groupby("name")["contribution"].transform("sum")
        df = df[totals > 0].reset_index(drop=True)

        if stacked:
            tot = df.groupby("name")["contribution"].transform("sum")
            df["contribution.relative"] = np.where(tot > 0, df["contribution"] / tot, 0.0)
        return df

    def _wilcoxon_per_pathway(self, union, pw_list, prob_list, idx, paired):
        """Per-pathway Wilcoxon test of the two conditions' per-group-pair prob."""
        from scipy.stats import wilcoxon, mannwhitneyu

        pw_index = [{pw: p for p, pw in enumerate(pws)} for pws in pw_list]
        out = np.zeros(len(union))
        for i, pw in enumerate(union):
            cols = []
            present = True
            for k in range(2):
                pi = pw_index[k].get(pw)
                if pi is None:
                    present = False
                    break
                cols.append(prob_list[k][:, :, pi].ravel())
            if not present:
                out[i] = 0.0
                continue
            a, b = cols[0], cols[1]
            # drop group-pairs that are zero in both conditions
            keep = (a != 0) | (b != 0)
            a, b = a[keep], b[keep]
            if len(a) <= 3:
                out[i] = 0.0
                continue
            try:
                if paired:
                    out[i] = wilcoxon(a, b).pvalue
                else:
                    out[i] = mannwhitneyu(a, b, use_continuity=True,
                                          alternative="two-sided").pvalue
            except Exception:
                out[i] = 0.0
        return np.nan_to_num(out, nan=0.0)

    def __repr__(self) -> str:  # pragma: no cover
        return f"MultiCellComm(datasets={self.names}, groups={len(self.group_names)})"
