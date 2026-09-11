"""Normalization of the concepts of every extracted paper into one shared vocabulary.

Extraction reports concepts per paper with ids that mean nothing outside that paper, so the same method appears
once per paper under a different local id; this step merges those local concepts into global ones identified by
a slug and records every spelling it was merged from in `aliases`.
The agent also proposes concept-to-concept relations from general knowledge, which carry a rationale instead of
evidence because no single paper backs them.
`check_normalization_against_extractions` decides whether the merge really covers the extractions it claims to,
which is the only defence against an agent dropping or inventing a concept.
The prompt is built here too, so the rules the agent is told and the rules that are enforced cannot drift apart.
"""

import json
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from rkgk.domain.models.base import Entity, Slug
from rkgk.domain.models.paper_extraction import LocalConceptId, PaperExtraction
from rkgk.domain.models.vocabulary import ConceptRelationType, ConceptType, describe_vocabulary

CONCEPT_NORMALIZATION_SCHEMA_VERSION = 1


class LocalConceptRef(Entity):
    """A concept as extraction left it: a paper and an id that is local to that paper."""

    paper_id: int = Field(ge=1)
    local_id: LocalConceptId


class NormalizedConcept(Entity):
    """One global concept, with the local concepts it was merged from kept as its provenance."""

    id: Slug
    canonical_name: str = Field(min_length=1)
    type: ConceptType
    aliases: tuple[str, ...] = ()
    description: str = ""
    merged_from: tuple[LocalConceptRef, ...] = Field(min_length=1)


class GeneralKnowledgeEdge(Entity):
    """An edge between two normalized concepts that no paper states.

    `rationale` takes the place of evidence: one sentence saying why the relation holds, since quoting a paper
    would claim the paper backs a statement it never made.
    """

    source_id: Slug
    target_id: Slug
    relation: ConceptRelationType
    rationale: str

    @field_validator("rationale")
    @classmethod
    def _reject_blank_rationale(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("rationale must contain non-whitespace characters")
        return value

    @model_validator(mode="after")
    def _reject_self_relation(self) -> Self:
        if self.source_id == self.target_id:
            raise ValueError(f"source_id and target_id must differ, both are {self.source_id!r}")
        return self


class ConceptNormalization(Entity):
    # Literal pins the version so an artifact written against another schema fails validation instead of being
    # read as if it were the current one.
    schema_version: Literal[1]
    concepts: tuple[NormalizedConcept, ...] = Field(min_length=1)
    concept_relations: tuple[GeneralKnowledgeEdge, ...] = ()

    @model_validator(mode="after")
    def _check_slugs_and_refs(self) -> Self:
        """Keep the global id space consistent: one slug per concept, one concept per local ref, edges on slugs."""
        declared: set[str] = set()
        for concept in self.concepts:
            if concept.id in declared:
                raise ValueError(f"concepts declares {concept.id!r} more than once")
            declared.add(concept.id)

        owner_of: dict[tuple[int, str], str] = {}
        for concept in self.concepts:
            for ref in concept.merged_from:
                key = (ref.paper_id, ref.local_id)
                owner = owner_of.get(key)
                if owner is not None:
                    raise ValueError(
                        f"merged_from claims paper {ref.paper_id} {ref.local_id!r} "
                        f"for both {owner!r} and {concept.id!r}"
                    )
                owner_of[key] = concept.id

        seen_edges: set[tuple[str, str, ConceptRelationType]] = set()
        for edge in self.concept_relations:
            for slug in (edge.source_id, edge.target_id):
                if slug not in declared:
                    raise ValueError(f"concept_relations refers to the undeclared concept {slug!r}")
            edge_key = (edge.source_id, edge.target_id, edge.relation)
            if edge_key in seen_edges:
                raise ValueError(
                    f"concept_relations repeats {edge.source_id!r} -> {edge.target_id!r} "
                    f"with relation {edge.relation.value!r}"
                )
            seen_edges.add(edge_key)
        return self


class ConceptNormalizationIssue(Entity):
    """One reason a normalization is rejected, addressed by the path of the offending field."""

    path: str
    message: str


class ConceptNormalizationValidationError(Exception):
    """Raised when a normalization does not fit the schema or does not match the extractions it merges."""

    def __init__(self, issues: tuple[ConceptNormalizationIssue, ...]) -> None:
        super().__init__("; ".join(f"{issue.path}: {issue.message}" for issue in issues))
        self.issues = issues


class MissingPaperExtractionsError(Exception):
    """Raised when papers of the index have no extraction, so normalizing them all is not possible yet."""

    def __init__(self, paper_ids: tuple[int, ...]) -> None:
        listed = ", ".join(str(paper_id) for paper_id in paper_ids)
        super().__init__(f"these papers have no extraction yet: {listed}")
        self.paper_ids = paper_ids


def check_normalization_against_extractions(
    normalization: ConceptNormalization, extractions: tuple[PaperExtraction, ...]
) -> tuple[ConceptNormalizationIssue, ...]:
    """Report every place where the normalization disagrees with the extractions it merges.

    All issues are collected instead of raising on the first one, because an agent fixing its output needs the
    whole list to converge in one more attempt.
    """
    types_by_ref = {
        (extraction.paper_id, concept.local_id): concept.type
        for extraction in extractions
        for concept in extraction.concepts
    }
    known_papers = {extraction.paper_id for extraction in extractions}
    issues: list[ConceptNormalizationIssue] = []
    covered: set[tuple[int, str]] = set()
    for index, concept in enumerate(normalization.concepts):
        source_types: list[ConceptType] = []
        for position, ref in enumerate(concept.merged_from):
            key = (ref.paper_id, ref.local_id)
            source_type = types_by_ref.get(key)
            if source_type is None:
                problem = (
                    f"paper {ref.paper_id} has no concept {ref.local_id!r}"
                    if ref.paper_id in known_papers
                    else f"there is no extraction for paper {ref.paper_id}"
                )
                issues.append(
                    ConceptNormalizationIssue(path=f"concepts[{index}].merged_from[{position}]", message=problem)
                )
                continue
            covered.add(key)
            source_types.append(source_type)
        if source_types and concept.type not in source_types:
            issues.append(
                ConceptNormalizationIssue(
                    path=f"concepts[{index}].type",
                    message=(
                        f"is {concept.type.value!r}, which none of the merged concepts has; they are "
                        f"{', '.join(sorted({source.value for source in source_types}))}"
                    ),
                )
            )
    for extraction in extractions:
        for concept in extraction.concepts:
            if (extraction.paper_id, concept.local_id) not in covered:
                issues.append(
                    ConceptNormalizationIssue(
                        path="concepts",
                        message=f"paper {extraction.paper_id} {concept.local_id!r} is in no merged_from",
                    )
                )
    return tuple(issues)


def build_concept_normalization_schema() -> dict[str, object]:
    """Render the schema an agent must follow; it is generated so the prompt can never drift from the model."""
    return ConceptNormalization.model_json_schema()


_RULES = (
    "Merge two concepts only when they mean the same thing, and keep concepts whose meaning you cannot tell "
    "apart as separate normalized concepts.",
    "Write `canonical_name` as the English canonical name of the concept, and derive `id` from it as its "
    "lowercase words joined by `-`, matching `^[a-z0-9]+(-[a-z0-9]+)*$`.",
    "Collect every spelling, abbreviation and Japanese name of the merged concepts in `aliases`.",
    "List every extracted concept in exactly one `merged_from`, as the paper id and the local id it was "
    "extracted under.",
    "Keep `type` the type the merged concepts were extracted with.",
    "Never add a concept that no paper extracted.",
    "Add concept relations from general knowledge between normalized concepts only, and name no other id.",
    "Prefer `is_a`, `part_of` and `used_for` for those relations, and use `related_to` only when none of the "
    "other three fits.",
    "Give every relation a `rationale` of one sentence saying why it holds, because a general-knowledge "
    "relation carries no evidence.",
    f"Set `schema_version` to {CONCEPT_NORMALIZATION_SCHEMA_VERSION}.",
)


def _describe_concepts(extraction: PaperExtraction) -> list[str]:
    lines = []
    for concept in extraction.concepts:
        parts = [concept.local_id, concept.name, concept.type.value]
        if concept.aliases:
            parts.append(f"aliases: {', '.join(concept.aliases)}")
        if concept.description:
            parts.append(concept.description)
        lines.append("- " + " | ".join(parts))
    return lines


def build_concept_normalization_prompt(
    extractions: tuple[PaperExtraction, ...],
    previous: object | None = None,
    issues: tuple[ConceptNormalizationIssue, ...] = (),
) -> str:
    """Write the instructions and the extracted concepts an agent needs to normalize them in one pass.

    A retry gets the rejected JSON and the issues appended, so the agent corrects its own answer instead of
    starting over and losing the parts that were already right.
    """
    lines = [
        "# Task",
        "",
        "You merge the concepts extracted from several research papers into one shared vocabulary.",
        "Read the concepts below, report each of them once as a normalized concept, and say how those "
        "normalized concepts relate to each other.",
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
        "## Papers",
        "",
        "Each paper was extracted on its own, so `c1`, `c2`, ... are local to the paper they are listed under.",
        "A concept line reads `- local id | name | type | aliases: ... | description`, and it ends early when "
        "the paper reported no aliases or no description.",
    ]
    for extraction in extractions:
        lines += ["", f"## Paper {extraction.paper_id}", "", *_describe_concepts(extraction)]
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


class ConceptNormalizationRun(Entity):
    """What one normalization run produced, with the papers it covered and the attempts it took."""

    normalization: ConceptNormalization
    attempts: int = Field(ge=1)
    paper_ids: tuple[int, ...]
