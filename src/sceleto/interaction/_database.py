"""CellChatDB ligand–receptor database — loading and gene resolution.

The database is a port of CellChatDB (Jin et al., *Nat. Commun.* 2021; CellChat
v2). Each species ships four tables under :mod:`sceleto.interaction.db`:

``interaction``
    One row per ligand–receptor pair: ``pathway_name``, ``ligand``, ``receptor``
    (each a single gene symbol *or* a complex name), the four cofactor columns
    (``agonist``, ``antagonist``, ``co_A_receptor``, ``co_I_receptor``), and the
    ``annotation`` class (``Secreted Signaling`` / ``ECM-Receptor`` /
    ``Cell-Cell Contact`` / ``Non-protein Signaling``).
``complex``
    Multi-subunit ligands/receptors → up to five ``subunit_*`` gene symbols.
``cofactor``
    Named cofactor sets referenced by the interaction cofactor columns.
``geneinfo``
    Official gene symbols; used to decide whether a token is a single gene or a
    complex name.

Usage
-----
>>> db = load_cellcommdb("human")
>>> db.interaction.head()
>>> db.subset(annotation="Secreted Signaling")           # filter by class
>>> db.resolve_genes("TGFbR1_R2")                          # complex → subunits
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Optional, Sequence

import pandas as pd

SPECIES = ("human", "mouse", "zebrafish")

ANNOTATIONS = (
    "Secreted Signaling",
    "ECM-Receptor",
    "Cell-Cell Contact",
    "Non-protein Signaling",
)

#: Diffusible annotation classes (mediate long-range signaling) — the rest
#: (``Cell-Cell Contact``) are contact-dependent. Used by the spatial model.
DIFFUSIBLE_ANNOTATIONS = ("Secreted Signaling", "ECM-Receptor", "Non-protein Signaling")

#: Order in which L-R pairs are sorted before inference: all diffusible classes
#: first, contact-dependent last (so the ``nLR1`` split is a single boundary).
ANNOTATION_ORDER = (
    "Secreted Signaling",
    "ECM-Receptor",
    "Non-protein Signaling",
    "Cell-Cell Contact",
)

_SUBUNIT_COLS = ["subunit_1", "subunit_2", "subunit_3", "subunit_4", "subunit_5"]


def _read(name: str) -> pd.DataFrame:
    """Read a bundled parquet table from ``sceleto.interaction.db``."""
    ref = resources.files("sceleto.interaction.db").joinpath(f"{name}.parquet")
    with resources.as_file(ref) as path:
        return pd.read_parquet(path)


@dataclass
class CellCommDB:
    """A loaded CellChatDB for one species.

    Attributes
    ----------
    species
        One of ``"human"``, ``"mouse"``, ``"zebrafish"``.
    interaction
        Ligand–receptor interaction table (indexed by ``name``).
    complex
        Complex → subunit table (indexed by complex name).
    cofactor
        Cofactor sets (indexed by cofactor name).
    geneinfo
        Single-column frame of official gene symbols.
    """

    species: str
    interaction: pd.DataFrame
    complex: pd.DataFrame
    cofactor: pd.DataFrame
    geneinfo: pd.DataFrame

    # -- gene resolution -------------------------------------------------
    def is_complex(self, name: str) -> bool:
        """True if *name* is a complex (present in the complex table)."""
        return name in self._complex_index

    def resolve_genes(self, name: str) -> list[str]:
        """Resolve a ligand/receptor token to its constituent gene symbols.

        A single gene resolves to ``[name]``; a complex resolves to its
        non-empty subunits (order preserved). Unknown tokens resolve to
        ``[name]`` (treated as a single gene).

        Parameters
        ----------
        name
            A gene symbol or a complex name from the ``ligand``/``receptor``
            columns of :attr:`interaction`.
        """
        if name in self._complex_index:
            row = self.complex.loc[name]
            subs = [str(row[c]).strip() for c in _SUBUNIT_COLS if c in row.index]
            return [s for s in subs if s]
        return [name]

    def resolve_cofactor(self, name: str) -> list[str]:
        """Resolve a cofactor-set name to its gene symbols (``[]`` if blank)."""
        if not name or name not in self._cofactor_index:
            return []
        row = self.cofactor.loc[name]
        genes = [str(v).strip() for v in row.values]
        return [g for g in genes if g]

    # -- filtering -------------------------------------------------------
    def subset(
        self,
        *,
        annotation: Optional[str | Sequence[str]] = None,
        pathway: Optional[str | Sequence[str]] = None,
    ) -> "CellCommDB":
        """Return a new :class:`CellCommDB` with the interaction table filtered.

        Parameters
        ----------
        annotation
            Keep only interactions with these annotation class(es). See
            :data:`ANNOTATIONS`.
        pathway
            Keep only interactions in these signaling pathway(s).
        """
        df = self.interaction
        if annotation is not None:
            wanted = {annotation} if isinstance(annotation, str) else set(annotation)
            bad = wanted - set(ANNOTATIONS)
            if bad:
                raise ValueError(
                    f"Unknown annotation(s) {sorted(bad)}. Valid: {list(ANNOTATIONS)}"
                )
            df = df[df["annotation"].isin(wanted)]
        if pathway is not None:
            wanted = {pathway} if isinstance(pathway, str) else set(pathway)
            df = df[df["pathway_name"].isin(wanted)]
        return CellCommDB(
            species=self.species,
            interaction=df.copy(),
            complex=self.complex,
            cofactor=self.cofactor,
            geneinfo=self.geneinfo,
        )

    # -- introspection ---------------------------------------------------
    @property
    def pathways(self) -> list[str]:
        """Sorted unique signaling pathway names in the interaction table."""
        return sorted(self.interaction["pathway_name"].unique())

    def signaling_genes(self) -> list[str]:
        """All gene symbols referenced by any ligand/receptor (complexes expanded).

        This is the gene universe to intersect with ``adata.var_names`` before
        inference (CellChat's ``subsetData`` step).
        """
        genes: set[str] = set()
        for col in ("ligand", "receptor"):
            for tok in self.interaction[col].unique():
                genes.update(self.resolve_genes(str(tok)))
        # cofactor genes
        for col in ("agonist", "antagonist", "co_A_receptor", "co_I_receptor"):
            for tok in self.interaction[col].unique():
                genes.update(self.resolve_cofactor(str(tok)))
        genes.discard("")
        return sorted(genes)

    def __post_init__(self) -> None:
        # index lookups for O(1) membership / row access
        if self.interaction.index.name != "name" and "name" in self.interaction.columns:
            self.interaction = self.interaction.set_index("name", drop=False)
        self._complex_index = set(self.complex.index)
        self._cofactor_index = set(self.cofactor.index)

    def __repr__(self) -> str:  # pragma: no cover - display helper
        return (
            f"CellCommDB(species={self.species!r}, "
            f"n_interactions={len(self.interaction)}, "
            f"n_pathways={len(self.pathways)}, "
            f"n_complex={len(self.complex)})"
        )


def load_cellcommdb(species: str = "human") -> CellCommDB:
    """Load the bundled CellChatDB for a species.

    Parameters
    ----------
    species
        One of ``"human"``, ``"mouse"``, ``"zebrafish"``.

    Returns
    -------
    CellCommDB
        Loaded database with interaction / complex / cofactor / geneinfo tables.
    """
    if species not in SPECIES:
        raise ValueError(f"Unknown species {species!r}. Valid: {list(SPECIES)}")
    inter = _read(f"cellchatdb_{species}_interaction")
    cx = _read(f"cellchatdb_{species}_complex").set_index("name")
    cf = _read(f"cellchatdb_{species}_cofactor").set_index("name")
    gi = _read(f"cellchatdb_{species}_geneinfo")
    return CellCommDB(
        species=species,
        interaction=inter,
        complex=cx,
        cofactor=cf,
        geneinfo=gi,
    )


def load_ppi(species: str = "human") -> pd.DataFrame:
    """Load the bundled protein–protein interaction edge list.

    Ships the CellChat PPI network for the (not-yet-ported) expression-smoothing
    / diffusion step (``smoothData`` / ``projectData`` in R). Provided so the
    data is available; **the smoothing step itself is not implemented yet**, so
    the current pipeline does not consume this. Only ``"human"`` and ``"mouse"``
    are available.

    Returns
    -------
    pd.DataFrame
        Columns ``gene_i``, ``gene_j``, ``weight`` (undirected edges).
    """
    if species not in ("human", "mouse"):
        raise ValueError(f"PPI only available for 'human'/'mouse', got {species!r}")
    return _read(f"ppi_{species}")


def list_cellcommdb() -> list[str]:
    """Return the species for which a CellChatDB is bundled."""
    return list(SPECIES)
