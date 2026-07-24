"""Gene-symbol normalization for cross-species / symbol-update robustness.

User AnnData objects frequently carry gene symbols that differ from the ones
CellChatDB was curated against — because HGNC/MGI rename genes over time
(*symbol updates*), or because a different alias was used upstream. If a
ligand or receptor symbol fails to match, that interaction is silently dropped
and communication is under-called.

:class:`GeneMapper` closes that gap. It builds an alias → canonical lookup from
the bundled ``geneinfo`` table (``Symbol`` + space-separated ``Synonym``) and
maps an arbitrary set of query symbols onto the database's canonical symbols,
reporting what matched directly, what matched via an alias, and what could not
be matched at all.

The mapper is deliberately DB-agnostic and reusable: give it any
``symbols``/``synonyms`` frame and it works. :func:`GeneMapper.from_db` wires it
to a loaded :class:`~sceleto.interaction._database.CellCommDB`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import pandas as pd


@dataclass
class MappingResult:
    """Outcome of mapping a query gene set onto canonical symbols.

    Attributes
    ----------
    mapping
        ``{query_symbol: canonical_symbol}`` for every query that resolved.
    matched_direct
        Queries that matched a canonical symbol as-is.
    matched_alias
        ``{query_symbol: canonical_symbol}`` resolved through a synonym.
    unmatched
        Queries that matched nothing.
    """

    mapping: dict[str, str]
    matched_direct: set[str]
    matched_alias: dict[str, str]
    unmatched: set[str]

    @property
    def n_matched(self) -> int:
        return len(self.mapping)

    def summary(self) -> str:
        return (
            f"{self.n_matched} matched "
            f"({len(self.matched_direct)} direct, {len(self.matched_alias)} via alias), "
            f"{len(self.unmatched)} unmatched"
        )


class GeneMapper:
    """Map query gene symbols onto a reference symbol set, alias-aware.

    Parameters
    ----------
    symbols
        Canonical reference symbols (the database's official symbols).
    synonyms
        Optional ``{canonical_symbol: [alias, ...]}``. Aliases are matched only
        when they are unambiguous — an alias pointing at more than one canonical
        symbol is dropped from the alias table (a conservative choice: better to
        leave it unmatched than to guess wrong).
    case_insensitive
        Also match on a case-folded key. Enables e.g. human ``TGFB1`` in the DB
        to catch a query written ``Tgfb1``, which matters when a mouse-cased
        object is scored against the human DB or vice versa. Direct
        (case-sensitive) hits always take priority over case-folded ones.
    """

    def __init__(
        self,
        symbols: Iterable[str],
        synonyms: Optional[dict[str, list[str]]] = None,
        *,
        case_insensitive: bool = True,
    ) -> None:
        self.case_insensitive = case_insensitive
        self._canonical: set[str] = {s for s in symbols if s}

        # case-folded canonical index (fold -> canonical); collisions dropped
        self._fold_to_canon: dict[str, str] = {}
        if case_insensitive:
            seen_fold: dict[str, int] = {}
            for s in self._canonical:
                f = s.casefold()
                seen_fold[f] = seen_fold.get(f, 0) + 1
            for s in self._canonical:
                f = s.casefold()
                if seen_fold[f] == 1:
                    self._fold_to_canon[f] = s

        # alias -> canonical, keeping only unambiguous aliases
        self._alias_to_canon: dict[str, str] = {}
        self._exact_alias: set[str] = set()
        if synonyms:
            alias_targets: dict[str, set[str]] = {}
            for canon, aliases in synonyms.items():
                if canon not in self._canonical:
                    continue
                for a in aliases:
                    if not a or a in self._canonical:
                        # skip blanks and aliases that are themselves canonical
                        continue
                    alias_targets.setdefault(a, set()).add(canon)
            self._alias_to_canon = {
                a: next(iter(t)) for a, t in alias_targets.items() if len(t) == 1
            }
            # exact (non-folded) alias keys, kept for match-tier classification
            self._exact_alias = set(self._alias_to_canon)
            if case_insensitive:
                # add case-folded alias keys where unambiguous (collect first,
                # then merge — never mutate the dict we're iterating)
                fold_targets: dict[str, set[str]] = {}
                for a, canon in self._alias_to_canon.items():
                    fold_targets.setdefault(a.casefold(), set()).add(canon)
                extra: dict[str, str] = {}
                for a, canon in self._alias_to_canon.items():
                    fa = a.casefold()
                    if fa not in self._alias_to_canon and len(fold_targets[fa]) == 1:
                        extra.setdefault(fa, canon)
                self._alias_to_canon.update(extra)

    # -- construction ----------------------------------------------------
    @classmethod
    def from_db(cls, db, *, case_insensitive: bool = True) -> "GeneMapper":
        """Build a :class:`GeneMapper` from a loaded :class:`CellCommDB`.

        Uses ``db.geneinfo`` (``Symbol`` + space-separated ``Synonym``).
        """
        gi = db.geneinfo
        symbols = gi["Symbol"].astype(str).tolist()
        synonyms: dict[str, list[str]] = {}
        if "Synonym" in gi.columns:
            for sym, syn in zip(gi["Symbol"].astype(str), gi["Synonym"].astype(str)):
                if syn:
                    synonyms[sym] = [t for t in syn.replace(",", " ").split() if t]
        return cls(symbols, synonyms, case_insensitive=case_insensitive)

    # -- single lookup ---------------------------------------------------
    def resolve(self, symbol: str) -> Optional[str]:
        """Resolve one query symbol to a canonical symbol, or ``None``.

        Resolution order: exact canonical → exact alias → case-folded canonical
        → case-folded alias.
        """
        if symbol in self._canonical:
            return symbol
        if symbol in self._alias_to_canon:
            return self._alias_to_canon[symbol]
        if self.case_insensitive:
            f = symbol.casefold()
            if f in self._fold_to_canon:
                return self._fold_to_canon[f]
            if f in self._alias_to_canon:
                return self._alias_to_canon[f]
        return None

    # -- bulk mapping ----------------------------------------------------
    def map(self, query: Iterable[str]) -> MappingResult:
        """Map a set of query symbols onto canonical symbols.

        Parameters
        ----------
        query
            Query gene symbols (e.g. ``adata.var_names``).

        Returns
        -------
        MappingResult
            Full breakdown of direct / alias / unmatched resolutions.
        """
        mapping: dict[str, str] = {}
        direct: set[str] = set()
        alias: dict[str, str] = {}
        unmatched: set[str] = set()
        for q in query:
            q = str(q)
            if q in self._canonical:
                mapping[q] = q
                direct.add(q)
                continue
            r = self.resolve(q)
            if r is None:
                unmatched.add(q)
            else:
                mapping[q] = r
                alias[q] = r
        return MappingResult(
            mapping=mapping,
            matched_direct=direct,
            matched_alias=alias,
            unmatched=unmatched,
        )

    def rename_frame(self, adata_var_names: Iterable[str]) -> dict[str, str]:
        """Return a collision-safe rename dict → canonical symbols.

        Only entries that actually change are returned, suitable for
        ``adata.var.rename(index=...)``. Direct matches are omitted (no change).

        Renames that would **collide** are dropped so the result never
        introduces duplicate var_names: an alias is not renamed to a canonical
        that is already present among the query names, nor to a canonical that
        two different aliases both target. This prevents silently overwriting a
        real gene's expression column (e.g. a panel carrying both ``TGFB1`` and
        a legacy alias of it).
        """
        names = list(map(str, adata_var_names))
        present = set(names)
        res = self.map(names)
        proposed = {q: c for q, c in res.mapping.items() if q != c}

        # target already present as its own var_name → would duplicate it
        safe = {q: c for q, c in proposed.items() if c not in present}

        # Group candidate queries by target, then resolve collisions BY PRIORITY
        # TIER rather than dropping all of them: a case-fold match (higher
        # priority) should win over an alias match (lower) on the same target,
        # mirroring resolve()'s own precedence. Only drop when the winning tier
        # is itself ambiguous (2+ queries tie at the best tier).
        by_target: dict[str, list[str]] = {}
        for q, c in safe.items():
            by_target.setdefault(c, []).append(q)

        out: dict[str, str] = {}
        for c, qs in by_target.items():
            if len(qs) == 1:
                out[qs[0]] = c
                continue
            tiers = {q: self._match_tier(q) for q in qs}
            best = min(tiers.values())
            winners = [q for q in qs if tiers[q] == best]
            if len(winners) == 1:
                out[winners[0]] = c
            # else: still ambiguous at the best tier → drop (safe)
        return out

    def _match_tier(self, symbol: str) -> int:
        """Priority tier a query matched at: 0=exact, 1=alias, 2=fold, 3=fold-alias.

        Lower is higher priority, mirroring :meth:`resolve`'s precedence.
        """
        if symbol in self._canonical:
            return 0
        if symbol in self._exact_alias:
            return 1
        if self.case_insensitive:
            f = symbol.casefold()
            if f in self._fold_to_canon:
                return 2
            if f in self._alias_to_canon:
                return 3
        return 4

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GeneMapper(n_canonical={len(self._canonical)}, "
            f"n_alias={len(self._alias_to_canon)}, "
            f"case_insensitive={self.case_insensitive})"
        )
