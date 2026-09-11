"""Extraction of the knowledge graph of one paper.

An agent reads the pages of a single paper and reports the concepts it found, so the ids here are local to that
paper (`c1`, `c2`, ...) and mean nothing outside it; a later normalization step assigns the global slugs.
Every edge carries quotes from the paper, and `check_extraction_against_paper` decides whether those quotes are
really in the text, which is the only defence against an agent inventing evidence.
The prompt is built here too, so the rules the agent is told and the rules that are enforced cannot drift apart.
"""

import json
import re
from typing import Annotated, Literal, Protocol, Self

from pydantic import Field, field_validator, model_validator

from rkgk.domain.models.base import Entity
from rkgk.domain.models.paper import Paper
from rkgk.domain.models.vocabulary import ConceptRelationType, ConceptType, PaperConceptRelation, describe_vocabulary

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


def check_extraction_against_paper(result: PaperExtraction, paper: Paper) -> tuple[ExtractionIssue, ...]:
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
    return PaperExtraction.model_json_schema()


_RULES = (
    "Report only concepts that appear in the paper text; never add a concept from general knowledge.",
    "Write `name` as the English canonical name of the concept, and put abbreviations and spelling variants "
    "in `aliases`.",
    "Use `proposes` for a concept the paper introduces as its own contribution and `uses` for one it applies "
    "in its method or experiments.",
    "Use `addresses` for a problem the paper tries to solve and `mentions` when the paper only refers to the "
    "concept; the line between `mentions` and `uses` is whether the paper actually works with the concept.",
    "Copy every `quote` verbatim from the paper text; never summarize or paraphrase it.",
    "Set every `page` to the page the quote is printed on.",
    "Add a concept-to-concept relation only when the paper text backs it, and leave `concept_relations` empty "
    "otherwise.",
    "Write `summary_ja` in Japanese, between 200 and 400 characters, covering the problem, the method, and "
    "the results.",
)


def _identifier_rules(paper_id: int) -> tuple[str, ...]:
    return (
        "Give every concept a local id `c1`, `c2`, ... in declaration order.",
        "Refer to those local ids only; `paper_concepts` and `concept_relations` may use no other id.",
        "Never invent a global id or a slug, because normalization assigns them after this step.",
        f"Set `paper_id` to {paper_id}.",
        f"Set `schema_version` to {EXTRACTION_SCHEMA_VERSION}.",
    )


def build_extraction_prompt(
    paper: Paper, previous: object | None = None, issues: tuple[ExtractionIssue, ...] = ()
) -> str:
    """Write the instructions and the paper text an agent needs to extract this one paper.

    A retry gets the rejected JSON and the issues appended, so the agent corrects its own answer instead of
    starting over and losing the parts that were already right.
    """
    lines = [
        "# Task",
        "",
        "You extract the knowledge graph of one research paper.",
        "Read the paper below and report the concepts it contains, how the paper relates to them, and how they "
        "relate to each other.",
        "Answer with a single JSON object that follows the JSON Schema you were given.",
        "",
        "## Rules",
        "",
        *(f"- {rule}" for rule in _RULES),
        "",
        "## Vocabulary",
        "",
        "Use these node types and relation names, spelled exactly as shown.",
        "",
        describe_vocabulary().rstrip("\n"),
        "",
        "## Identifiers",
        "",
        *(f"- {rule}" for rule in _identifier_rules(paper.meta.id)),
        "",
        "## Paper",
        "",
        f"Title: {paper.meta.title}",
        f"Year: {paper.meta.year}",
        f"Venue: {paper.meta.venue}",
    ]
    for page in paper.pages:
        lines += ["", f"## Page {page.number}", "", page.text.rstrip("\n")]
    if previous is not None:
        lines += [
            "",
            "## Previous attempt",
            "",
            "This JSON was rejected.",
            "",
            "```json",
            json.dumps(previous, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Issues",
            "",
            *(f"- {issue.path}: {issue.message}" for issue in issues),
            "",
            "Return a complete corrected JSON object that fixes every issue above.",
        ]
    lines += ["", "Return only the JSON object."]
    return "\n".join(lines) + "\n"


class ExtractionOutcome(Entity):
    """What one extraction run produced, with the number of attempts it took to pass validation."""

    result: PaperExtraction
    attempts: int = Field(ge=1)


class Extractor(Protocol):
    """An agent that answers a prompt with JSON that follows the given schema."""

    def extract(self, prompt: str, schema: dict[str, object]) -> object: ...


class ExtractorError(Exception):
    """Raised when the agent could not be run or answered with something other than the requested JSON."""
