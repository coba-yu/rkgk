"""Extraction of one paper as an agent writes it.

An agent reads the pages of a single paper and reports the concepts it found, so the ids here are local to that
paper (`c1`, `c2`, ...) and mean nothing outside it; a later normalization step assigns the global slugs.
Every edge carries quotes from the paper, and `check_extraction_against_paper` decides whether those quotes are
really in the text, which is the only defence against an agent inventing evidence.
"""

import re
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from rkgk.domain.base import Entity
from rkgk.domain.paper import Paper
from rkgk.domain.vocabulary import ConceptRelationType, ConceptType, PaperConceptRelation

EXTRACTION_SCHEMA_VERSION = 1

LOCAL_CONCEPT_ID_PATTERN = r"^c[1-9][0-9]*$"

LocalConceptId = Annotated[str, Field(pattern=LOCAL_CONCEPT_ID_PATTERN)]

_WHITESPACE_RUN = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse every run of whitespace into one space and strip the ends.

    A quote and the page it came from differ in line breaks and indentation once the agent has copied it,
    so both sides are normalized before they are compared.
    """
    return _WHITESPACE_RUN.sub(" ", text).strip()


class ExtractedEvidence(Entity):
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
    evidence: tuple[ExtractedEvidence, ...] = Field(min_length=1)


class ExtractedConceptEdge(Entity):
    """An edge between two concepts of this paper; its origin is `paper`, so evidence is always required."""

    source_id: LocalConceptId
    target_id: LocalConceptId
    relation: ConceptRelationType
    evidence: tuple[ExtractedEvidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_self_relation(self) -> Self:
        if self.source_id == self.target_id:
            raise ValueError(f"source_id and target_id must differ, both are {self.source_id!r}")
        return self


class ExtractionResult(Entity):
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


class ExtractionIssue(Entity):
    """One reason an extraction is rejected, addressed by the path of the offending field."""

    path: str
    message: str


class ExtractionValidationError(Exception):
    """Raised when an extraction does not fit the schema or does not match the paper it claims to describe."""

    def __init__(self, issues: tuple[ExtractionIssue, ...]) -> None:
        super().__init__("; ".join(f"{issue.path}: {issue.message}" for issue in issues))
        self.issues = issues


def _check_evidence(
    evidence: tuple[ExtractedEvidence, ...], prefix: str, page_texts: dict[int, str], page_count: int
) -> list[ExtractionIssue]:
    issues: list[ExtractionIssue] = []
    for index, item in enumerate(evidence):
        path = f"{prefix}.evidence[{index}]"
        if item.page not in page_texts:
            issues.append(ExtractionIssue(path=f"{path}.page", message=f"page {item.page} is outside 1..{page_count}"))
            continue
        if normalize_whitespace(item.quote) not in page_texts[item.page]:
            issues.append(
                ExtractionIssue(path=f"{path}.quote", message=f"is not found in the text of page {item.page}")
            )
    return issues


def check_extraction_against_paper(result: ExtractionResult, paper: Paper) -> tuple[ExtractionIssue, ...]:
    """Report every place where the extraction disagrees with the paper text.

    All issues are collected instead of raising on the first one, because an agent fixing its output needs the
    whole list to converge in one more attempt.
    """
    issues: list[ExtractionIssue] = []
    if result.paper_id != paper.meta.id:
        issues.append(
            ExtractionIssue(
                path="paper_id", message=f"is {result.paper_id}, but the paper being checked is {paper.meta.id}"
            )
        )
    page_texts = {page.number: normalize_whitespace(page.text) for page in paper.pages}
    for index, edge in enumerate(result.paper_concepts):
        issues += _check_evidence(edge.evidence, f"paper_concepts[{index}]", page_texts, paper.meta.page_count)
    for index, concept_edge in enumerate(result.concept_relations):
        issues += _check_evidence(
            concept_edge.evidence, f"concept_relations[{index}]", page_texts, paper.meta.page_count
        )
    return tuple(issues)


def build_extraction_schema() -> dict[str, object]:
    """Render the schema an agent must follow; it is generated so the prompt can never drift from the model."""
    return ExtractionResult.model_json_schema()
