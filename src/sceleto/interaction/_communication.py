"""Core cell–cell communication inference — the :class:`CellComm` pipeline.

Python port of CellChat's inference engine (``modeling.R`` / ``utilities.R``),
operating on :class:`~anndata.AnnData`. The :class:`CellComm` object is the
analogue of CellChat's S4 object: it holds the AnnData, the grouping, the
database in use, and — as the pipeline runs — the over-expressed features, the
significant L–R pairs, and the per-pair probability/​p-value arrays.

Pipeline
--------
1. :meth:`CellComm.identify_overexpressed` — one-vs-rest Wilcoxon DE (via scanpy)
   → over-expressed genes → significant L–R pairs (``LRsig``).
2. :meth:`CellComm.compute_communication` — Hill-function mass-action model with
   geometric-mean complexes, co-receptor/agonist/antagonist modulation, and a
   label-permutation significance test → ``prob`` / ``pval`` arrays.
3. :meth:`CellComm.compute_communication_pathway` — sum significant L–R prob
   within each signaling pathway.
4. :meth:`CellComm.aggregate` — cluster×cluster ``count`` / ``weight`` matrices.
5. :meth:`CellComm.subset_communication` — tidy long-form results.

The heavy statistics (Wilcoxon DE) are delegated to scanpy; the CellChat-specific
math (Hill model, permutation p-values) is implemented here.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from ._database import (
    CellCommDB, load_cellcommdb, ANNOTATION_ORDER, DIFFUSIBLE_ANNOTATIONS,
)
from ._genes import GeneMapper
from . import _expr as E


def _match_col(columns, key: str):
    """Find the column whose string form equals *key* (dtype-agnostic match)."""
    for c in columns:
        if str(c) == key:
            return c
    raise KeyError(key)


class CellComm:
    """Cell–cell communication analysis over an :class:`~anndata.AnnData`.

    Parameters
    ----------
    adata
        Cells × genes AnnData with **normalized, non-negative** expression in
        the chosen layer (log-normalized ``.X`` is typical).
    groupby
        ``adata.obs`` column with the cell-group labels. Its categorical order
        (or sorted uniques) defines the network's node order.
    db
        A loaded :class:`CellCommDB`. If ``None``, loads the species default.
    species
        Used only when ``db is None`` to pick the bundled database.
    layer
        Layer to read expression from; ``None`` uses ``adata.X``.
    map_symbols
        If ``True`` (default), normalize ``adata.var_names`` onto database
        canonical symbols (alias/case-aware) before analysis — this is what
        makes cross-species and symbol-update mismatches recoverable.
    verbose
        Print progress.

    Notes
    -----
    The AnnData is **not** copied; ``compute_communication`` writes results into
    ``self`` (and, optionally, ``adata.uns['cellcomm']`` via :meth:`to_uns`).
    """

    def __init__(
        self,
        adata: AnnData,
        groupby: str,
        *,
        db: Optional[CellCommDB] = None,
        species: str = "human",
        layer: Optional[str] = None,
        map_symbols: bool = True,
        coordinates: Optional[np.ndarray] = None,
        spatial_factors: Optional[dict] = None,
        verbose: bool = True,
    ) -> None:
        if groupby not in adata.obs:
            raise KeyError(f"groupby {groupby!r} not in adata.obs")
        # spatial inputs (optional): per-cell x,y coordinates + {ratio, tol}
        self.coordinates = None if coordinates is None else np.asarray(coordinates, dtype=float)
        if self.coordinates is not None:
            if self.coordinates.shape[0] != adata.n_obs:
                raise ValueError(
                    f"coordinates has {self.coordinates.shape[0]} rows but adata "
                    f"has {adata.n_obs} cells; they must align 1:1 (same order)."
                )
            if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2:
                raise ValueError("coordinates must be an (n_cells, 2) array of x,y")
        self.spatial_factors = spatial_factors
        self.is_spatial = coordinates is not None
        self.groupby = groupby
        self.layer = layer
        self.verbose = verbose
        self.db = db if db is not None else load_cellcommdb(species)

        # Drop unlabeled (NaN) cells up front so they can't be silently absorbed
        # into a phantom group nor distort the population-size denominator.
        col = adata.obs[groupby]
        if col.isna().any():
            n_drop = int(col.isna().sum())
            keep_mask = ~col.isna().to_numpy()
            adata = adata[keep_mask].copy()
            if self.coordinates is not None:
                self.coordinates = self.coordinates[keep_mask]
            col = adata.obs[groupby]
            if verbose:
                print(f"[interaction] dropped {n_drop} cell(s) with NaN {groupby!r} label")
        self.adata = adata

        # group labels → ordered Categorical (drop unused levels, keep order)
        if isinstance(col.dtype, pd.CategoricalDtype):
            present = set(col.dropna().unique())
            cats = [c for c in col.cat.categories if c in present]
            self.groups = pd.Categorical(col, categories=cats)
        else:
            # str() the labels before sorting so a mixed-type or numeric-labeled
            # column can't raise TypeError in sorted()
            labels = col.astype(str)
            self.groups = pd.Categorical(labels, categories=sorted(labels.unique()))
        self.group_names = list(self.groups.categories)

        # gene-symbol normalization onto DB canonical symbols
        self._var_names = list(map(str, adata.var_names))
        if map_symbols:
            gm = GeneMapper.from_db(self.db)
            rename = gm.rename_frame(self._var_names)
            if rename:
                self._var_names = [rename.get(g, g) for g in self._var_names]
                if verbose:
                    print(
                        f"[interaction] normalized {len(rename)} gene symbols "
                        "onto CellChatDB canonical symbols (alias/case)"
                    )
        # guard: if the *input* already had duplicate var_names (or any slipped
        # through), keep the FIRST occurrence's column index — never let a later
        # duplicate silently shadow an earlier real gene.
        self._gene_index: dict[str, int] = {}
        dups = 0
        for i, g in enumerate(self._var_names):
            if g in self._gene_index:
                dups += 1
                continue
            self._gene_index[g] = i
        if dups and verbose:
            print(
                f"[interaction] warning: {dups} duplicate gene symbol(s) in "
                "var_names; keeping first occurrence of each"
            )

        # pipeline state (populated as steps run)
        self.overexpressed_: Optional[dict[str, list[str]]] = None
        self.features_sig_: Optional[set[str]] = None
        self.LRsig: Optional[pd.DataFrame] = None
        self.net: dict[str, np.ndarray] = {}
        self.netP: dict[str, object] = {}
        self.patterns: dict[str, dict] = {}  # NMF communication patterns

    # -- typed state accessors + readiness guards -----------------------
    # net/netP stay plain dicts (round-tripped through adata.uns by to_uns);
    # these read-only views add discoverability + one canonical guard per slot.
    def _require_net(self) -> dict:
        """The communication result dict; raises if compute_communication() unrun."""
        if not self.net:
            raise RuntimeError("Call compute_communication() first.")
        return self.net

    def _require_aggregate(self) -> tuple[np.ndarray, np.ndarray]:
        """(weight, count) K×K matrices; raises if aggregate() unrun."""
        net = self._require_net()
        if "weight" not in net or "count" not in net:
            raise RuntimeError("Call aggregate() first.")
        return net["weight"], net["count"]

    def _require_netP(self) -> tuple[np.ndarray, list[str]]:
        """(prob (K,K,nPathway), pathways); raises if pathway step unrun."""
        if not self.netP or "prob" not in self.netP:
            raise RuntimeError(
                "cc.netP['prob'] is missing; call compute_communication_pathway() first."
            )
        prob = np.asarray(self.netP["prob"], dtype=float)
        if prob.ndim != 3:
            raise ValueError(
                f"cc.netP['prob'] must be 3-D (K,K,nPathway), got {prob.shape}"
            )
        return prob, [str(p) for p in self.netP.get("pathways", [])]

    @property
    def is_computed(self) -> bool:
        """Whether :meth:`compute_communication` has run."""
        return bool(self.net)

    @property
    def prob(self) -> np.ndarray:
        """Per-pair communication probability (K×K×nLR)."""
        return self._require_net()["prob"]

    @property
    def pval(self) -> np.ndarray:
        """Per-pair permutation p-values (K×K×nLR)."""
        return self._require_net()["pval"]

    @property
    def weight(self) -> np.ndarray:
        """Aggregated K×K interaction-strength matrix (needs aggregate())."""
        return self._require_aggregate()[0]

    @property
    def count(self) -> np.ndarray:
        """Aggregated K×K interaction-count matrix (needs aggregate())."""
        return self._require_aggregate()[1]

    @property
    def pathways(self) -> list[str]:
        """Signaling pathway names (needs compute_communication_pathway())."""
        return self._require_netP()[1]

    # -- expression access ----------------------------------------------
    def _get_expr(self) -> np.ndarray:
        """Dense cells × genes matrix from the chosen layer/``.X``."""
        X = self.adata.layers[self.layer] if self.layer is not None else self.adata.X
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=float)
        # A single non-finite entry silently corrupts the whole analysis: the
        # per-signaling max-normalization (`_signaling_matrix`) divides by
        # max(X), so one inf zeroes every probability and one NaN skips
        # normalization entirely. CellChat assumes finite log-normalized data,
        # so reject non-finite input up front rather than return a wrong result.
        if not getattr(self, "_expr_checked", False):
            if not np.isfinite(X).all():
                n_bad = int((~np.isfinite(X)).sum())
                raise ValueError(
                    f"expression matrix contains {n_bad} non-finite value(s) "
                    "(NaN/inf); CellChat expects finite log-normalized data. "
                    "Filter or impute these before constructing CellChat."
                )
            self._expr_checked = True
        return X

    def _signaling_matrix(self, genes: Sequence[str]) -> np.ndarray:
        """Normalized ``cells × signaling-genes`` matrix.

        Mirrors CellChat exactly: it restricts to the signaling-gene subset
        (``object@data.signaling``) *first*, then divides by the max **of that
        subset** (``data.use <- data/max(data)``). Normalizing by the global
        max over all genes — dominated by housekeeping genes like MALAT1/MT- —
        would shrink signaling values by a large constant and, through the
        nonlinear Hill term, distort every probability.

        The normalized subset is small (cells × ~800 signaling genes) and is
        read twice per ``compute_communication`` (observed + bootstrap), so it
        is cached — much cheaper on memory than caching the full dense matrix.
        """
        key = tuple(genes)
        cached = getattr(self, "_sigmat_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        X = self._get_expr()
        cols = [self._gene_index[g] for g in genes]
        sub = X[:, cols]
        mx = sub.max()
        if mx > 0:
            sub = sub / mx
        self._sigmat_cache = (key, sub)
        return sub

    def _signaling_avg(
        self, type: str, trim: float, genes: Sequence[str]
    ) -> pd.DataFrame:
        """``genes × groups`` average expression over the signaling gene set."""
        sub = self._signaling_matrix(genes)
        return E.group_mean(sub, genes, self.groups, type=type, trim=trim)

    # -- step 1: over-expressed genes & interactions --------------------
    def identify_overexpressed(
        self,
        *,
        thresh_p: float = 0.05,
        thresh_fc: float = 0.0,
        thresh_pc: float = 0.0,
        only_pos: bool = True,
        variable_both: bool = True,
    ) -> "CellComm":
        """DE (Wilcoxon, one-vs-rest via scanpy) → significant L–R pairs.

        Populates :attr:`overexpressed_`, :attr:`features_sig_`, and
        :attr:`LRsig` (the exact pair set :meth:`compute_communication` scores).

        Parameters
        ----------
        thresh_p
            Adjusted-p cutoff for a gene to count as over-expressed.
        thresh_fc
            Minimum log-fold-change.
        thresh_pc
            Minimum detection fraction in the group.
        only_pos
            Keep only up-regulated genes.
        variable_both
            If ``True``, keep an interaction only when **both** ligand and
            receptor are over-expressed; else either endpoint suffices
            (mirrors CellChat ``variable.both``).
        """
        from ._de import wilcoxauc

        # CellChat runs DE on the SIGNALING-gene subset only (object@data.signaling
        # = extractGene(DB) ∩ var_names), NOT the whole transcriptome. Restricting
        # here is essential for parity: DE over all ~17k genes selects an order of
        # magnitude too many "over-expressed" genes.
        universe = [g for g in self.db.signaling_genes() if g in self._gene_index]
        cols = [self._gene_index[g] for g in universe]
        X = self._get_expr()[:, cols]

        # presto-style one-vs-rest Wilcoxon + AUC (the same statistic CellChat
        # uses via presto::wilcoxauc; logFC = mean_in − mean_out in log space).
        de = wilcoxauc(X, self.groups, var_names=universe)

        # CellChat filter: raw p < thresh_p, |logFC| >= thresh_fc,
        # pct.max > thresh_pc*100; then (only_pos) keep logFC > 0.
        pct_max = np.maximum(de["pct_in"].to_numpy(), de["pct_out"].to_numpy())
        keep = (de["pval"].to_numpy() < thresh_p)
        keep &= np.abs(de["logFC"].to_numpy()) >= thresh_fc
        keep &= pct_max > thresh_pc * 100.0
        if only_pos:
            keep &= de["logFC"].to_numpy() > 0
        de_sig = de.loc[keep]

        oe: dict[str, list[str]] = {
            str(g): sorted(sub["features"].astype(str))
            for g, sub in de_sig.groupby("group")
        }
        sig: set[str] = set(de_sig["features"].astype(str))
        self.overexpressed_ = oe
        self.features_sig_ = sig
        self.de_ = de_sig

        # select L-R interactions whose endpoints are over-expressed
        self.LRsig = self._select_interactions(sig, variable_both=variable_both)
        if self.verbose:
            print(
                f"[interaction] {len(sig)} over-expressed genes → "
                f"{len(self.LRsig)} significant L-R pairs"
            )
        return self

    def _resolve_present(self, token: str) -> list[str]:
        """Subunits of a token that are present in the (mapped) var set."""
        genes = self.db.resolve_genes(token)
        return [g for g in genes if g in self._gene_index]

    def _select_interactions(
        self, features_sig: set[str], *, variable_both: bool
    ) -> pd.DataFrame:
        """Filter the interaction table to measurable + over-expressed pairs."""
        inter = self.db.interaction
        rows = []
        for name, row in inter.iterrows():
            lig, rec = str(row["ligand"]), str(row["receptor"])
            lig_g = self._resolve_present(lig)
            rec_g = self._resolve_present(rec)
            # all subunits must be measured, else the pair is undefined
            if not lig_g or not rec_g:
                continue
            if len(lig_g) != len(self.db.resolve_genes(lig)):
                continue
            if len(rec_g) != len(self.db.resolve_genes(rec)):
                continue
            lig_oe = any(g in features_sig for g in lig_g)
            rec_oe = any(g in features_sig for g in rec_g)
            if variable_both:
                ok = lig_oe and rec_oe
            else:
                ok = lig_oe or rec_oe
            if ok:
                rows.append(name)
        out = inter.loc[rows].copy()
        # order by annotation so diffusible pairs precede contact-dependent ones
        if "annotation" in out.columns:
            order = {a: i for i, a in enumerate(ANNOTATION_ORDER)}
            out = out.sort_values(
                by="annotation", key=lambda s: s.map(lambda a: order.get(a, 99)), kind="stable"
            )
        return out

    def _spatial_factors(
        self, pairs, K, *, distance_use, interaction_range, scale_distance,
        k_min, contact_dependent, contact_range,
    ):
        """Return ``(P_diff, P_contact, nLR1)`` spatial probability factors.

        For RNA data (no coordinates) both matrices are all-ones and
        ``nLR1 = nLR`` (no spatial effect). For spatial data, ``P_diff`` is the
        inverse-distance diffusion weight and ``P_contact = P_diff * adj_contact``
        for the trailing ``Cell-Cell Contact`` L-R block (index >= nLR1).
        """
        nLR = len(pairs)
        if not self.is_spatial:
            ones = np.ones((K, K))
            return ones, ones, nLR

        from ._spatial import compute_region_distance, spatial_probability_factor

        if self.spatial_factors is None:
            raise ValueError(
                "spatial data requires spatial_factors={'ratio':..., 'tol':...}"
            )
        ratio = float(self.spatial_factors["ratio"])
        tol = float(self.spatial_factors["tol"])
        res = compute_region_distance(
            self.coordinates, self.groups,
            interaction_range=interaction_range, ratio=ratio, tol=tol, k_min=k_min,
            contact_dependent=contact_dependent, contact_range=contact_range,
        )
        if distance_use:
            P_diff = spatial_probability_factor(res.d_spatial, scale_distance=scale_distance)
        else:
            # 0/1 proximity mask only (no distance weighting)
            P_diff = np.where(np.isnan(res.d_spatial), 0.0, 1.0)
            np.fill_diagonal(P_diff, 1.0)
        P_contact = P_diff * res.adj_contact

        # nLR1 = boundary between diffusible pairs and the Cell-Cell Contact
        # block (pairs are pre-sorted diffusible-first by _select_interactions).
        ann = pairs["annotation"].astype(str).to_numpy() if "annotation" in pairs else np.array([""] * nLR)
        diffusible = np.isin(ann, list(DIFFUSIBLE_ANNOTATIONS))
        if contact_dependent and diffusible.any() and (~diffusible).any():
            nLR1 = int(np.max(np.where(diffusible)[0])) + 1  # first contact index
        elif contact_dependent and not diffusible.any():
            nLR1 = 0  # all contact
        else:
            nLR1 = nLR  # all diffusion (or contact not applied)
        return P_diff, P_contact, nLR1

    # -- step 2: communication probability ------------------------------
    def compute_communication(
        self,
        *,
        type: str = "triMean",
        trim: float = 0.1,
        population_size: bool = False,
        nboot: int = 100,
        seed: int = 1,
        Kh: float = 0.5,
        n: float = 1.0,
        n_jobs: int = 1,
        distance_use: bool = True,
        interaction_range: float = 250.0,
        scale_distance: float = 0.01,
        k_min: int = 10,
        contact_dependent: bool = True,
        contact_range: Optional[float] = None,
    ) -> "CellComm":
        """Communication probability + permutation p-value per L–R pair.

        Requires :meth:`identify_overexpressed` first (or a manually set
        :attr:`LRsig`). Writes ``self.net = {'prob', 'pval'}``: each a
        ``K × K × nLR`` array (sender × receiver × interaction).

        Parameters
        ----------
        type, trim
            Group-summary statistic (``triMean`` default) and trim fraction.
        population_size
            Weight probabilities by group cell-fractions (CellChat
            ``population.size``).
        nboot
            Number of label permutations for the p-value.
        seed
            RNG seed for the permutations.
        Kh, n
            Hill function parameters (dissociation constant, coefficient).
        n_jobs
            Threads for the permutation bootstrap (the dominant cost). ``1``
            (default) runs serially; higher values parallelize the per-boot
            group averaging. Results are identical regardless of ``n_jobs``.
        distance_use, interaction_range, scale_distance, k_min, contact_dependent, contact_range
            **Spatial only** (when the object was built with ``coordinates``).
            Constrain communication by spatial proximity: ``interaction_range``
            (µm diffusion cutoff), ``scale_distance`` (distance scaling),
            ``contact_range`` (µm, required if ``contact_dependent`` for the
            ``Cell-Cell Contact`` L-R pairs). See :func:`compute_region_distance`.
        """
        if self.LRsig is None:
            raise RuntimeError(
                "No L-R pairs selected. Call identify_overexpressed() first "
                "or assign self.LRsig."
            )
        pairs = self.LRsig
        gene_universe = self.db.signaling_genes()
        genes = [g for g in gene_universe if g in self._gene_index]
        avg = self._signaling_avg(type, trim, genes)

        K = len(self.group_names)
        nLR = len(pairs)
        geneL = pairs["ligand"].astype(str).tolist()
        geneR = pairs["receptor"].astype(str).tolist()

        # spatial constraint P.spatial (diffusion) + P.spatial*adj.contact
        # (contact). For RNA data both are all-ones (no spatial effect). nLR1 is
        # the index boundary: diffusible pairs (< nLR1) use P_diff, the trailing
        # Cell-Cell Contact block (>= nLR1) uses P_contact.
        P_diff, P_contact, nLR1 = self._spatial_factors(
            pairs, K, distance_use=distance_use, interaction_range=interaction_range,
            scale_distance=scale_distance, k_min=k_min,
            contact_dependent=contact_dependent, contact_range=contact_range,
        )

        cx = self.db.complex
        cof = self.db.cofactor

        # ligand / receptor expression (K-length vectors per pair)
        dataLavg = E.lr_expr(geneL, avg, cx)          # nLR x K
        dataRavg = E.lr_expr(geneR, avg, cx)          # nLR x K
        coA = E.coreceptor_factor(pairs["co_A_receptor"].astype(str).tolist(), avg, cof)
        coI = E.coreceptor_factor(pairs["co_I_receptor"].astype(str).tolist(), avg, cof)
        dataRavg = dataRavg * coA / coI

        # population size term
        counts = np.array([np.sum(self.groups.codes == j) for j in range(K)], float)
        nC = len(self.groups)
        pop = counts / nC  # length K

        agonist = pairs["agonist"].astype(str).tolist()
        antagonist = pairs["antagonist"].astype(str).tolist()

        # permutation bootstraps of the group labels → averaged expr per boot.
        # Uses the SAME signaling-subset-normalized matrix as the observed avg
        # (BUG-FIX: previously divided by the global all-gene max). This is the
        # dominant cost (nboot × per-group triMean), and each permutation is
        # independent, so it parallelizes cleanly over n_jobs.
        rng = np.random.default_rng(seed)
        Xsub = self._signaling_matrix(genes)
        codes = self.groups.codes
        levels = self.group_names
        perms = [rng.permutation(nC) for _ in range(nboot)]

        def _boot_avg(perm):
            gboot = pd.Categorical.from_codes(codes[perm], categories=levels)
            return E.group_mean(Xsub, genes, gboot, type=type, trim=trim)

        if n_jobs == 1 or nboot < 8:
            boot_avgs = [_boot_avg(p) for p in perms]
        else:
            from joblib import Parallel, delayed

            boot_avgs = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_boot_avg)(p) for p in perms
            )

        prob = np.zeros((K, K, nLR))
        pval = np.zeros((K, K, nLR))

        for i in range(nLR):
            L = dataLavg[i][:, None]        # K x 1
            R = dataRavg[i][None, :]        # 1 x K
            dataLR = L @ R                  # K x K  (sender ligand × receiver receptor)
            P1 = dataLR**n / (Kh**n + dataLR**n)
            # spatial factor: diffusible pairs (i < nLR1) vs contact block
            Psp = P_diff if i < nLR1 else P_contact

            if (P1 * Psp).sum() == 0:
                prob[:, :, i] = 0.0
                pval[:, :, i] = 1.0
                continue

            P2 = self._agonist_matrix(agonist[i], avg, cof, Kh, n)
            P3 = self._antagonist_matrix(antagonist[i], avg, cof, Kh, n)
            if population_size:
                P4 = pop[:, None] @ pop[None, :]
            else:
                P4 = 1.0
            Pnull = P1 * P2 * P3 * P4 * Psp
            prob[:, :, i] = Pnull

            # permutation null distribution
            exceed = np.zeros((K, K))
            for b in range(nboot):
                bavg = boot_avgs[b]
                Lb = E.lr_expr([geneL[i]], bavg, cx)[0][:, None]
                Rb = E.lr_expr([geneR[i]], bavg, cx)[0][None, :]
                coAb = E.coreceptor_factor([pairs["co_A_receptor"].iloc[i]], bavg, cof)[0]
                coIb = E.coreceptor_factor([pairs["co_I_receptor"].iloc[i]], bavg, cof)[0]
                Rb = Rb * (coAb / coIb)[None, :]
                dLRb = Lb @ Rb
                P1b = dLRb**n / (Kh**n + dLRb**n)
                P2b = self._agonist_matrix(agonist[i], bavg, cof, Kh, n)
                P3b = self._antagonist_matrix(antagonist[i], bavg, cof, Kh, n)
                Pboot = P1b * P2b * P3b * P4 * Psp
                exceed += (Pboot - Pnull > 0)
            pval[:, :, i] = exceed / nboot

        pval[prob == 0] = 1.0
        self.net = {"prob": prob, "pval": pval}
        self._pair_names = list(pairs.index)
        if self.verbose:
            print(f"[interaction] communication inference done: {nLR} pairs, {K} groups")
        return self

    @staticmethod
    def _agonist_matrix(name, avg, cof, Kh, n) -> np.ndarray:
        v = E.agonist_factor(name, avg, cof, Kh=Kh, n=n)
        return v[:, None] @ v[None, :]

    @staticmethod
    def _antagonist_matrix(name, avg, cof, Kh, n) -> np.ndarray:
        v = E.antagonist_factor(name, avg, cof, Kh=Kh, n=n)
        return v[:, None] @ v[None, :]

    # -- step 3: pathway-level aggregation ------------------------------
    def compute_communication_pathway(self, *, thresh: float = 0.05) -> "CellComm":
        """Sum significant per-pair prob within each signaling pathway.

        Writes ``self.netP = {'prob' (K×K×nPathway), 'pathways' (list)}``.
        """
        self._require_net()
        prob = self.net["prob"].copy()
        pval = self.net["pval"]
        prob[pval > thresh] = 0.0
        pathways = self.LRsig["pathway_name"].astype(str).to_numpy()
        uniq = pd.unique(pathways)
        K = prob.shape[0]
        summed = np.zeros((K, K, len(uniq)))
        totals = np.empty(len(uniq))
        for p, pw in enumerate(uniq):
            idx = np.where(pathways == pw)[0]
            summed[:, :, p] = prob[:, :, idx].sum(axis=2)
            totals[p] = summed[:, :, p].sum()
        # drop pathways with no significant communication (CellChat keeps only
        # pathways.sig = pathways where the summed prob != 0), then order by
        # total communication strength (descending).
        keep = np.where(totals > 0)[0]
        order = keep[np.argsort(totals[keep])[::-1]]
        self.netP = {
            "prob": summed[:, :, order],
            "pathways": [str(uniq[o]) for o in order],
            "thresh": thresh,
        }
        if self.verbose:
            print(
                f"[interaction] {len(order)} signaling pathways "
                f"({len(uniq) - len(order)} with no significant communication dropped)"
            )
        return self

    # -- optional QC filter ---------------------------------------------
    def filter_communication(self, *, min_cells: int = 10) -> "CellComm":
        """Zero out communication involving cell groups with too few cells.

        Ports CellChat ``filterCommunication`` (min-cells branch): any cell group
        with ``<= min_cells`` cells is unreliable, so all communication where it
        is the sender OR receiver is set to 0. Should be called after
        :meth:`compute_communication` and before :meth:`aggregate` /
        :meth:`compute_communication_pathway`.

        Parameters
        ----------
        min_cells
            Cell groups with at most this many cells are excluded.
        """
        self._require_net()
        counts = np.array(
            [int(np.sum(self.groups.codes == j)) for j in range(len(self.group_names))]
        )
        excluded = np.where(counts <= min_cells)[0]
        if len(excluded):
            prob = self.net["prob"]
            before = int((prob > 0).sum())
            prob[excluded, :, :] = 0.0
            prob[:, excluded, :] = 0.0
            self.net["pval"][prob == 0] = 1.0
            after = int((prob > 0).sum())
            if self.verbose:
                names = [self.group_names[i] for i in excluded]
                pct = 100 * (before - after) / before if before else 0.0
                print(
                    f"[interaction] filtered groups with <= {min_cells} cells: "
                    f"{names} ({pct:.1f}% of interactions removed)"
                )
        elif self.verbose:
            print(f"[interaction] no cell group has <= {min_cells} cells; nothing filtered")
        return self

    # -- step 4: cluster×cluster aggregation ----------------------------
    def aggregate(self, *, thresh: float = 0.05) -> "CellComm":
        """Aggregate to K×K ``count`` and ``weight`` matrices.

        ``count`` = number of significant L–R pairs between each group pair;
        ``weight`` = summed communication probability.
        """
        self._require_net()
        prob, pval = self.net["prob"], self.net["pval"]
        sig = pval < thresh
        weight = np.where(sig, prob, 0.0).sum(axis=2)
        count = sig.sum(axis=2)
        self.net["count"] = count
        self.net["weight"] = weight
        return self

    # -- results extraction ---------------------------------------------
    def subset_communication(
        self,
        *,
        thresh: float = 0.05,
        sources: Optional[Sequence[str]] = None,
        targets: Optional[Sequence[str]] = None,
        signaling: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Tidy long-form significant communication table.

        Returns
        -------
        pd.DataFrame
            Columns: ``source``, ``target``, ``ligand``, ``receptor``,
            ``interaction_name``, ``pathway_name``, ``prob``, ``pval``.
        """
        self._require_net()
        prob, pval = self.net["prob"], self.net["pval"]
        names = self.group_names
        pairs = self.LRsig
        rows = []
        src_set = set(sources) if sources else None
        tgt_set = set(targets) if targets else None
        sig_set = set(signaling) if signaling else None
        for k in range(prob.shape[2]):
            pw = str(pairs["pathway_name"].iloc[k])
            if sig_set is not None and pw not in sig_set:
                continue
            iname = str(pairs.index[k])
            lig = str(pairs["ligand"].iloc[k])
            rec = str(pairs["receptor"].iloc[k])
            for a in range(len(names)):
                if src_set is not None and names[a] not in src_set:
                    continue
                for b in range(len(names)):
                    if tgt_set is not None and names[b] not in tgt_set:
                        continue
                    p, q = prob[a, b, k], pval[a, b, k]
                    if p > 0 and q < thresh:
                        rows.append((names[a], names[b], lig, rec, iname, pw, p, q))
        return pd.DataFrame(
            rows,
            columns=["source", "target", "ligand", "receptor",
                     "interaction_name", "pathway_name", "prob", "pval"],
        )

    # -- ranking & contribution -----------------------------------------
    def rank_net(self, *, measure: str = "weight", thresh: float = 0.05) -> pd.DataFrame:
        """Rank signaling pathways by total information flow (``rankNet``).

        For each pathway, "information flow" = the summed communication
        probability (``measure="weight"``) or the number of significant links
        (``measure="count"``) over all group pairs. Pathways are returned in
        descending order.

        Parameters
        ----------
        measure
            ``"weight"`` (summed probability, the CellChat "information flow") or
            ``"count"`` (number of significant group-pair links).
        thresh
            p-value threshold for significance.

        Returns
        -------
        pd.DataFrame
            Columns ``pathway_name``, ``flow`` (the ranking value), sorted
            descending.
        """
        self._require_net()
        prob = self.net["prob"].copy()
        prob[self.net["pval"] > thresh] = 0.0
        pathways = self.LRsig["pathway_name"].astype(str).to_numpy()
        rows = []
        for pw in pd.unique(pathways):
            idx = np.where(pathways == pw)[0]
            sl = prob[:, :, idx]
            if measure == "count":
                flow = float((sl > 0).sum())
            elif measure == "weight":
                flow = float(sl.sum())
            else:
                raise ValueError(f"measure must be 'weight' or 'count', got {measure!r}")
            if flow > 0:
                rows.append((str(pw), flow))
        df = pd.DataFrame(rows, columns=["pathway_name", "flow"])
        return df.sort_values("flow", ascending=False).reset_index(drop=True)

    def contribution(self, signaling: str, *, thresh: float = 0.05) -> pd.DataFrame:
        """Per-L–R-pair contribution to one signaling pathway (``netAnalysis_contribution``).

        Within a pathway, each ligand–receptor pair contributes a share of the
        total communication probability. Returns each pair's summed probability
        and its fractional contribution, descending.

        Parameters
        ----------
        signaling
            Pathway name (see :attr:`CellComm.netP` ``["pathways"]`` or
            ``db.pathways``).
        thresh
            p-value threshold for significance.

        Returns
        -------
        pd.DataFrame
            Columns ``interaction_name``, ``ligand``, ``receptor``, ``prob``,
            ``contribution`` (fraction of the pathway total), sorted descending.
        """
        self._require_net()
        pathways = self.LRsig["pathway_name"].astype(str).to_numpy()
        idx = np.where(pathways == str(signaling))[0]
        if len(idx) == 0:
            raise ValueError(
                f"pathway {signaling!r} not found among the significant L-R pairs"
            )
        prob = self.net["prob"].copy()
        prob[self.net["pval"] > thresh] = 0.0
        rows = []
        for k in idx:
            rows.append((
                str(self.LRsig.index[k]),
                str(self.LRsig["ligand"].iloc[k]),
                str(self.LRsig["receptor"].iloc[k]),
                float(prob[:, :, k].sum()),
            ))
        df = pd.DataFrame(rows, columns=["interaction_name", "ligand", "receptor", "prob"])
        total = df["prob"].sum()
        df["contribution"] = df["prob"] / total if total > 0 else 0.0
        return df.sort_values("prob", ascending=False).reset_index(drop=True)

    # -- discoverable method wrappers -----------------------------------
    # These delegate to the reusable free functions in _analysis/_manifold/
    # _patterns so the whole pipeline is reachable as cc.<method>() (fluent,
    # tab-completable). Each is a pure alias — `cc.X(...)` == `X(cc, ...)` — so
    # the free functions remain public and nothing else changes.
    def signaling_role_pathway(self, *, pattern: str = "outgoing") -> pd.DataFrame:
        """Per-pathway × cell-group signaling-role matrix (see
        :func:`~sceleto.interaction.signaling_role_pathway`)."""
        from ._analysis import signaling_role_pathway
        return signaling_role_pathway(self, pattern=pattern)

    def compute_net_similarity(
        self, *, slot: str = "netP", type: str = "functional",
        k=None, thresh=None,
    ) -> "CellComm":
        """Pairwise pathway-network similarity → ``netP['similarity'][type]``.

        Delegates to :func:`~sceleto.interaction.compute_net_similarity` (which
        returns the matrix); the method stores it and returns ``self`` so the
        manifold sub-pipeline chains: ``cc.compute_net_similarity().net_embedding()``.
        """
        from ._manifold import compute_net_similarity
        compute_net_similarity(self, slot=slot, type=type, k=k, thresh=thresh)
        return self

    def net_embedding(
        self, *, type: str = "functional", n_neighbors=None, min_dist: float = 0.3,
        metric: str = "correlation", seed: int = 42, pathway_remove=None,
        **umap_kwargs,
    ) -> "CellComm":
        """2-D UMAP of the pathway similarity → ``netP['embedding'][type]``.

        Delegates to :func:`~sceleto.interaction.net_embedding`; stores the
        embedding and returns ``self`` (chainable)."""
        from ._manifold import net_embedding
        net_embedding(
            self, type=type, n_neighbors=n_neighbors, min_dist=min_dist,
            metric=metric, seed=seed, pathway_remove=pathway_remove, **umap_kwargs,
        )
        return self

    def net_clustering(
        self, *, type: str = "functional", k=None, method: str = "kmeans",
        k_eigen=None, n_init: int = 10, seed: int = 0,
    ) -> pd.Series:
        """Cluster pathways in the similarity manifold (see
        :func:`~sceleto.interaction.net_clustering`)."""
        from ._manifold import net_clustering
        return net_clustering(
            self, type=type, k=k, method=method, k_eigen=k_eigen,
            n_init=n_init, seed=seed,
        )

    def identify_communication_patterns(
        self, pattern: str = "outgoing", k=None, *,
        seed: int = 0, max_iter: int = 500, store: bool = True, verbose: bool = True,
    ) -> dict:
        """NMF outgoing/incoming communication patterns (see
        :func:`~sceleto.interaction.identify_communication_patterns`)."""
        from ._patterns import identify_communication_patterns
        return identify_communication_patterns(
            self, pattern, k, seed=seed, max_iter=max_iter,
            store=store, verbose=verbose,
        )

    def select_k(
        self, pattern: str = "outgoing", k_range=range(2, 11), *,
        n_runs: int = 30, seed: int = 10, max_iter: int = 500, verbose: bool = True,
    ) -> pd.DataFrame:
        """NMF rank selection (cophenetic + silhouette; see
        :func:`~sceleto.interaction.select_k`)."""
        from ._patterns import select_k
        return select_k(
            self, pattern, k_range, n_runs=n_runs, seed=seed,
            max_iter=max_iter, verbose=verbose,
        )

    def to_uns(self, key: str = "cellcomm") -> None:
        """Store the result dict into ``adata.uns[key]``."""
        self.adata.uns[key] = {
            "groupby": self.groupby,
            "group_names": self.group_names,
            "net": self.net,
            "netP": self.netP,
            "LRsig": self.LRsig,
        }

    def __repr__(self) -> str:  # pragma: no cover
        nlr = 0 if self.LRsig is None else len(self.LRsig)
        return (
            f"CellComm(groups={len(self.group_names)}, "
            f"LRsig={nlr}, "
            f"computed={'yes' if self.net else 'no'})"
        )
