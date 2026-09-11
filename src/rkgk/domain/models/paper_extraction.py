"""Extraction of the knowledge graph of one paper.

An agent reads the pages of a single paper and reports the concepts it found, so the ids here are local to that
paper (`c1`, `c2`, ...) and mean nothing outside it; a later normalization step assigns the global slugs.
"""

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from rkgk.domain.models.base import Entity
from rkgk.domain.models.vocabulary import ConceptRelationType, ConceptType, PaperConceptRelation

EXTRACTION_SCHEMA_VERSION = 1

LOCAL_CONCEPT_ID_PATTERN = r"^c[1-9][0-9]*$"

LocalConceptId = Annotated[str, Field(pattern=LOCAL_CONCEPT_ID_PATTERN)]


class PageEvidence(Entity):
    """A quote from one page; chunk ids do not exist yet at extraction time, so evidence points at a page."""

    page: int = Field(ge=1)
    quote: str

    @field_validator("quote")
    @classmethod
    def _reject_blank_quote(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("quote must contain non-whitespace characters")
        return value


class ExtractedConcept(Entity):
    local_id: LocalConceptId
    name: str = Field(min_length=1)
    type: ConceptType
    aliases: tuple[str, ...] = ()
    description: str = ""


class ExtractedPaperConceptEdge(Entity):
    concept_id: LocalConceptId
    relation: PaperConceptRelation
    evidence: tuple[PageEvidence, ...] = Field(min_length=1)


class ExtractedConceptEdge(Entity):
    """An edge between two concepts of this paper; its origin is `paper`, so evidence is always required."""

    source_id: LocalConceptId
    target_id: LocalConceptId
    relation: ConceptRelationType
    evidence: tuple[PageEvidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_self_relation(self) -> Self:
        if self.source_id == self.target_id:
            raise ValueError(f"source_id and target_id must differ, both are {self.source_id!r}")
        return self


class PaperExtraction(Entity):
    # Literal pins the version so an artifact written against another schema fails validation instead of being
    # read as if it were the current one.
    schema_version: Literal[1]
    paper_id: int = Field(ge=1)
    summary_ja: str
    concepts: tuple[ExtractedConcept, ...] = Field(min_length=1)
    paper_concepts: tuple[ExtractedPaperConceptEdge, ...]
    concept_relations: tuple[ExtractedConceptEdge, ...] = ()

    @field_validator("summary_ja")
    @classmethod
    def _reject_blank_summary(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("summary_ja must contain non-whitespace characters")
        return value

    @model_validator(mode="after")
    def _check_local_ids(self) -> Self:
        """Keep the local id space consistent: declared once, referenced only after declaration, never duplicated."""
        declared: set[str] = set()
        for concept in self.concepts:
            if concept.local_id in declared:
                raise ValueError(f"concepts declares {concept.local_id!r} more than once")
            declared.add(concept.local_id)

        seen_paper_edges: set[tuple[str, PaperConceptRelation]] = set()
        for edge in self.paper_concepts:
            if edge.concept_id not in declared:
                raise ValueError(f"paper_concepts refers to the undeclared concept {edge.concept_id!r}")
            if (edge.concept_id, edge.relation) in seen_paper_edges:
                raise ValueError(f"paper_concepts repeats {edge.concept_id!r} with relation {edge.relation.value!r}")
            seen_paper_edges.add((edge.concept_id, edge.relation))

        seen_concept_edges: set[tuple[str, str, ConceptRelationType]] = set()
        for concept_edge in self.concept_relations:
            for concept_id in (concept_edge.source_id, concept_edge.target_id):
                if concept_id not in declared:
                    raise ValueError(f"concept_relations refers to the undeclared concept {concept_id!r}")
            key = (concept_edge.source_id, concept_edge.target_id, concept_edge.relation)
            if key in seen_concept_edges:
                raise ValueError(
                    f"concept_relations repeats {concept_edge.source_id!r} -> {concept_edge.target_id!r} "
                    f"with relation {concept_edge.relation.value!r}"
                )
            seen_concept_edges.add(key)
        return self


class PaperExtractionIssue(Entity):
    """One reason an extraction is rejected, addressed by the path of the offending field."""

    path: str
    message: str


class PaperExtractionValidationError(Exception):
    """Raised when an extraction does not fit the schema or does not match the paper it claims to describe."""

    def __init__(self, issues: tuple[PaperExtractionIssue, ...]) -> None:
        super().__init__("; ".join(f"{issue.path}: {issue.message}" for issue in issues))
        self.issues = issues


def build_paper_extraction_schema() -> dict[str, object]:
    """Render the schema an agent must follow; it is generated so the prompt can never drift from the model."""
    return PaperExtraction.model_json_schema()


class PaperExtractionRun(Entity):
    """What one extraction run produced, with the number of attempts it took to pass validation."""

    extraction: PaperExtraction
    attempts: int = Field(ge=1)
