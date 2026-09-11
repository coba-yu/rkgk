"""Normalization of the concepts of every extracted paper into one shared vocabulary.

Extraction reports concepts per paper with ids that mean nothing outside that paper, so the same method appears
once per paper under a different local id; this step merges those local concepts into global ones identified by
a slug and records every spelling it was merged from in `aliases`.
The agent also proposes concept-to-concept relations from general knowledge, which carry a rationale instead of
evidence because no single paper backs them.
"""

from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from rkgk.domain.models.base import Entity, Slug
from rkgk.domain.models.paper_extraction import LocalConceptId
from rkgk.domain.models.vocabulary import ConceptRelationType, ConceptType

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


def build_concept_normalization_schema() -> dict[str, object]:
    """Render the schema an agent must follow; it is generated so the prompt can never drift from the model."""
    return ConceptNormalization.model_json_schema()


class ConceptNormalizationRun(Entity):
    """What one normalization run produced, with the papers it covered and the attempts it took."""

    normalization: ConceptNormalization
    attempts: int = Field(ge=1)
    paper_ids: tuple[int, ...]
