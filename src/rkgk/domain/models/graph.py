"""Nodes and edges of the knowledge graph.

A paper-origin edge carries the evidence, quotes from the paper, that justifies it.
A general-knowledge edge carries no evidence because no single paper backs it.
"""

from typing import Self

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


class Concept(Entity):
    """A node of the graph: one canonical concept shared by every paper that refers to it."""

    id: Slug
    canonical_name: str = Field(min_length=1)
    type: ConceptType
    aliases: tuple[str, ...] = ()
    description: str = ""


class PaperConceptEdge(Entity):
    """An edge from a paper to a concept, with the quotes that justify it."""

    paper_id: int = Field(ge=1)
    concept_id: Slug
    relation: PaperConceptRelation
    evidence: tuple[Evidence, ...] = Field(min_length=1)


class ConceptEdge(Entity):
    """An edge between two concepts, either backed by a paper or added from general knowledge."""

    source_id: Slug
    target_id: Slug
    relation: ConceptRelationType
    origin: Origin
    paper_id: int | None = Field(default=None, ge=1)
    evidence: tuple[Evidence, ...] = ()

    @model_validator(mode="after")
    def _check_origin_consistency(self) -> Self:
        """Keep provenance honest: a paper-origin edge must point at the text it came from, and nothing else may.

        Evidence on a general-knowledge edge would claim a paper backs a statement the paper never made.
        """
        if self.source_id == self.target_id:
            raise ValueError("source_id and target_id must differ")
        if self.origin.spec.requires_evidence:
            if self.paper_id is None:
                raise ValueError(f"paper_id is required when origin is {self.origin.value}")
            if not self.evidence:
                raise ValueError(f"evidence must not be empty when origin is {self.origin.value}")
        else:
            if self.paper_id is not None:
                raise ValueError(f"paper_id must be None when origin is {self.origin.value}")
            if self.evidence:
                raise ValueError(f"evidence must be empty when origin is {self.origin.value}")
        return self
