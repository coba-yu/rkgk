"""Nodes and edges of the knowledge graph.

A paper-origin edge carries the evidence, quotes from the paper, that justifies it.
A general-knowledge edge carries a rationale instead, because no single paper backs it.
`KnowledgeGraph` gathers the nodes and edges of every paper into the one graph the index stores.
"""

from typing import Annotated, Self

from pydantic import Field, field_validator, model_validator

from rkgk.domain.models.base import Entity, Slug
from rkgk.domain.models.vocabulary import ConceptRelationType, ConceptType, Origin, PaperConceptRelation


class Evidence(Entity):
    page: int = Field(ge=1)
    quote: str
    chunk_id: str | None = None

    @field_validator("quote")
    @classmethod
    def _reject_blank_quote(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("quote must contain non-whitespace characters")
        return value


class UnresolvedEvidence(Entity):
    """A quote that appears on its page but in no single chunk of that page, identified by where it was claimed."""

    paper_id: int = Field(ge=1)
    page: int = Field(ge=1)
    quote: str


class EvidenceResolutionError(Exception):
    """Raised when evidence of a paper cannot be tied to a chunk; carries every unresolved quote at once."""

    def __init__(self, unresolved: tuple[UnresolvedEvidence, ...]) -> None:
        super().__init__(
            "; ".join(f"paper {item.paper_id} page {item.page}: quote {item.quote!r}" for item in unresolved)
        )
        self.unresolved = unresolved


class Concept(Entity):
    """A node of the graph: one canonical concept shared by every paper that refers to it.

    `paper_count` is how many papers the concept was merged from; against the number of papers in the graph it
    gives the document frequency that keeps generic concepts out of traversal.
    """

    id: Slug
    canonical_name: str = Field(min_length=1)
    type: ConceptType
    aliases: tuple[str, ...] = ()
    description: str = ""
    paper_count: int = Field(ge=1)


class PaperConceptEdge(Entity):
    """An edge from a paper to a concept, with the quotes that justify it."""

    paper_id: int = Field(ge=1)
    concept_id: Slug
    relation: PaperConceptRelation
    evidence: tuple[Evidence, ...] = Field(min_length=1)


class ConceptEdge(Entity):
    """An edge between two concepts, either backed by a paper or added from general knowledge.

    A paper-origin edge carries evidence and a general-knowledge edge carries a rationale, never both, so a
    search answer can tell a quoted relation from an unverified one by looking at the edge alone.
    """

    source_id: Slug
    target_id: Slug
    relation: ConceptRelationType
    origin: Origin
    paper_id: int | None = Field(default=None, ge=1)
    evidence: tuple[Evidence, ...] = ()
    rationale: str | None = None

    @model_validator(mode="after")
    def _check_origin_consistency(self) -> Self:
        """Keep provenance honest: a paper-origin edge must point at the text it came from, and nothing else may.

        Evidence on a general-knowledge edge would claim a paper backs a statement the paper never made, and a
        rationale on a paper-origin edge would let an unverified sentence pass as a quote.
        """
        if self.source_id == self.target_id:
            raise ValueError("source_id and target_id must differ")
        if self.origin.spec.requires_evidence:
            if self.paper_id is None:
                raise ValueError(f"paper_id is required when origin is {self.origin.value}")
            if not self.evidence:
                raise ValueError(f"evidence must not be empty when origin is {self.origin.value}")
            if self.rationale is not None:
                raise ValueError(f"rationale must be None when origin is {self.origin.value}")
        else:
            if self.paper_id is not None:
                raise ValueError(f"paper_id must be None when origin is {self.origin.value}")
            if self.evidence:
                raise ValueError(f"evidence must be empty when origin is {self.origin.value}")
            if self.rationale is None:
                raise ValueError(f"rationale is required when origin is {self.origin.value}")
            if not self.rationale.strip():
                raise ValueError(
                    f"rationale must contain non-whitespace characters when origin is {self.origin.value}"
                )
        return self


class KnowledgeGraph(Entity):
    """The whole graph as data: the papers, the concepts, and the edges between them.

    Papers are bare ids because their title and summary live in the paper and extraction artifacts; the graph only
    has to say which papers exist so every edge can be checked against them.
    Traversal runs on the networkx view built from this object, and this object is what the index stores, so the
    stored graph is validated on the way in like every other artifact.
    """

    paper_ids: tuple[Annotated[int, Field(ge=1)], ...]
    concepts: tuple[Concept, ...]
    paper_concepts: tuple[PaperConceptEdge, ...]
    concept_relations: tuple[ConceptEdge, ...]

    @model_validator(mode="after")
    def _check_references(self) -> Self:
        """Every edge must join declared nodes, and no node or edge may be declared twice."""
        papers = set(self.paper_ids)
        if len(papers) != len(self.paper_ids):
            raise ValueError("paper_ids must not repeat")

        declared: set[str] = set()
        for concept in self.concepts:
            if concept.id in declared:
                raise ValueError(f"concepts declares {concept.id!r} more than once")
            declared.add(concept.id)
            if concept.paper_count > len(papers):
                raise ValueError(
                    f"concept {concept.id!r} counts {concept.paper_count} papers but the graph has {len(papers)}"
                )

        seen_paper_edges: set[tuple[int, str, PaperConceptRelation]] = set()
        for edge in self.paper_concepts:
            if edge.paper_id not in papers:
                raise ValueError(f"paper_concepts refers to the undeclared paper {edge.paper_id}")
            if edge.concept_id not in declared:
                raise ValueError(f"paper_concepts refers to the undeclared concept {edge.concept_id!r}")
            key = (edge.paper_id, edge.concept_id, edge.relation)
            if key in seen_paper_edges:
                raise ValueError(
                    f"paper_concepts repeats paper {edge.paper_id} -> {edge.concept_id!r} "
                    f"with relation {edge.relation.value!r}"
                )
            seen_paper_edges.add(key)

        seen_concept_edges: set[tuple[str, str, ConceptRelationType, Origin, int | None]] = set()
        for concept_edge in self.concept_relations:
            for slug in (concept_edge.source_id, concept_edge.target_id):
                if slug not in declared:
                    raise ValueError(f"concept_relations refers to the undeclared concept {slug!r}")
            if concept_edge.paper_id is not None and concept_edge.paper_id not in papers:
                raise ValueError(f"concept_relations refers to the undeclared paper {concept_edge.paper_id}")
            edge_key = (
                concept_edge.source_id,
                concept_edge.target_id,
                concept_edge.relation,
                concept_edge.origin,
                concept_edge.paper_id,
            )
            if edge_key in seen_concept_edges:
                raise ValueError(
                    f"concept_relations repeats {concept_edge.source_id!r} -> {concept_edge.target_id!r} "
                    f"with relation {concept_edge.relation.value!r} from {concept_edge.origin.value}"
                    + (f" of paper {concept_edge.paper_id}" if concept_edge.paper_id is not None else "")
                )
            seen_concept_edges.add(edge_key)
        return self

    def document_frequency(self, concept_id: str) -> float:
        """Share of the papers in the graph that hold the concept.

        Traversal compares it to a threshold to keep generic concepts from linking unrelated papers.
        """
        concept = next(item for item in self.concepts if item.id == concept_id)
        return concept.paper_count / len(self.paper_ids)
