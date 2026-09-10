"""Entities stored in the knowledge graph and the index.

Every model is frozen and forbids unknown fields so that a schema change surfaces as a validation error
instead of silently dropping data that a later pipeline stage expects.
"""

from datetime import datetime
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rkgk.domain.vocabulary import ConceptRelationType, ConceptType, Origin, PaperConceptRelation

SLUG_PATTERN = r"^[a-z0-9]+(-[a-z0-9]+)*$"

Slug = Annotated[str, Field(pattern=SLUG_PATTERN)]


class Entity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PreprocessInfo(Entity):
    tool: str
    version: str
    processed_at: datetime


class PaperMeta(Entity):
    id: int = Field(ge=1)
    title: str = Field(min_length=1)
    authors: list[str]
    year: int
    venue: str
    page_count: int = Field(ge=1)
    doi: str | None = None
    arxiv_id: str | None = None
    preprocess: PreprocessInfo

    @property
    def dir_name(self) -> str:
        return f"{self.id:04d}"


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
    aliases: list[str] = []
    description: str = ""


class PaperConcept(Entity):
    paper_id: int = Field(ge=1)
    concept_id: Slug
    relation: PaperConceptRelation
    evidence: list[Evidence] = Field(min_length=1)


class ConceptRelation(Entity):
    source_id: Slug
    target_id: Slug
    relation: ConceptRelationType
    origin: Origin
    paper_id: int | None = None
    evidence: list[Evidence] = []

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


class Chunk(Entity):
    id: str
    paper_id: int = Field(ge=1)
    seq: int = Field(ge=0)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str

    @field_validator("text")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain non-whitespace characters")
        return value

    @model_validator(mode="after")
    def _check_pages_and_id(self) -> Self:
        """The id is derived from paper_id and seq so a chunk read from disk is traceable without a lookup."""
        if self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        expected_id = self.build_id(self.paper_id, self.seq)
        if self.id != expected_id:
            raise ValueError(f"id must be {expected_id!r}")
        return self

    @staticmethod
    def build_id(paper_id: int, seq: int) -> str:
        return f"{paper_id}:{seq}"

    @classmethod
    def create(cls, paper_id: int, seq: int, page_start: int, page_end: int, text: str) -> Self:
        return cls(
            id=cls.build_id(paper_id, seq),
            paper_id=paper_id,
            seq=seq,
            page_start=page_start,
            page_end=page_end,
            text=text,
        )
