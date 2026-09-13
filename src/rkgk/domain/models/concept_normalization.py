"""Normalization of the concepts of every extracted paper into one shared vocabulary.

Extraction reports concepts per paper with ids that mean nothing outside that paper, so the same method appears
once per paper under a different local id; this step merges those local concepts into global ones identified by
a slug and records every spelling it was merged from in `aliases`.
The agent also proposes concept-to-concept relations from general knowledge, which carry a rationale instead of
evidence because no single paper backs them; the relations the papers themselves state are handed to that stage
as `PaperStatedRelation` so it does not propose them a second time.
The two jobs are asked for one at a time: `ConceptMerge` is what the agent answers about the merge and
`GeneralKnowledgeRelationProposal` what it answers about the relations, while `ConceptNormalization` is the artifact
assembled from both and the only one of the three that is stored.
"""

from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from rkgk.domain.models.base import Entity, Slug
from rkgk.domain.models.paper_extraction import LocalConceptId
from rkgk.domain.models.vocabulary import ConceptRelationType, ConceptType

CONCEPT_NORMALIZATION_SCHEMA_VERSION = 1

# The two agent calls normalization is made of, named so a rejection can say which one it came from.
ConceptNormalizationStage = Literal["merge", "relations"]


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


class PaperStatedRelation(Entity):
    """A relation a paper states in its own text, written in the slug the merge gave each of its two ends.

    Handed to the relation stage as what is already known, so general knowledge does not propose again a pair
    and relation the graph carries from a paper.
    """

    source_id: Slug
    target_id: Slug
    relation: ConceptRelationType
    paper_ids: tuple[int, ...] = Field(min_length=1)


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


def _collect_declared_slugs(concepts: tuple[NormalizedConcept, ...]) -> set[str]:
    """Return the slug of every concept, rejecting a slug or a local reference that two concepts claim.

    Shared by the merge the agent answers and the artifact assembled from it, so both refuse the same id space.
    """
    declared: set[str] = set()
    for concept in concepts:
        if concept.id in declared:
            raise ValueError(f"concepts declares {concept.id!r} more than once")
        declared.add(concept.id)

    owner_of: dict[tuple[int, str], str] = {}
    for concept in concepts:
        for ref in concept.merged_from:
            key = (ref.paper_id, ref.local_id)
            owner = owner_of.get(key)
            if owner is not None:
                raise ValueError(
                    f"merged_from claims paper {ref.paper_id} {ref.local_id!r} for both {owner!r} and {concept.id!r}"
                )
            owner_of[key] = concept.id
    return declared


def _reject_repeated_edges(edges: tuple[GeneralKnowledgeEdge, ...]) -> None:
    """Reject a pair that carries the same relation twice, which would weight one edge of the graph twice."""
    seen_edges: set[tuple[str, str, ConceptRelationType]] = set()
    for edge in edges:
        edge_key = (edge.source_id, edge.target_id, edge.relation)
        if edge_key in seen_edges:
            raise ValueError(
                f"concept_relations repeats {edge.source_id!r} -> {edge.target_id!r} "
                f"with relation {edge.relation.value!r}"
            )
        seen_edges.add(edge_key)


class ConceptMerge(Entity):
    """What the agent answers when it is asked only to merge the extracted concepts.

    No `schema_version`, because this is one stage's answer on its way to `ConceptNormalization` and is never
    written to the data directory.
    """

    concepts: tuple[NormalizedConcept, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_slugs_and_refs(self) -> Self:
        _collect_declared_slugs(self.concepts)
        return self


class GeneralKnowledgeRelationProposal(Entity):
    """What the agent answers when it is asked only for the relations between the merged concepts.

    Empty is a valid answer: a vocabulary whose concepts share nothing beyond the papers is not an error.
    Whether the slugs are ones the merge declared cannot be seen from here, so a service checks that instead.
    """

    concept_relations: tuple[GeneralKnowledgeEdge, ...] = ()

    @model_validator(mode="after")
    def _check_edges(self) -> Self:
        _reject_repeated_edges(self.concept_relations)
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
        declared = _collect_declared_slugs(self.concepts)
        for edge in self.concept_relations:
            for slug in (edge.source_id, edge.target_id):
                if slug not in declared:
                    raise ValueError(f"concept_relations refers to the undeclared concept {slug!r}")
        _reject_repeated_edges(self.concept_relations)
        return self


class ConceptNormalizationIssue(Entity):
    """One reason a normalization is rejected, addressed by the path of the offending field."""

    path: str
    message: str


class ConceptNormalizationValidationError(Exception):
    """Raised when an answer does not fit the schema or does not match what the stage before it produced.

    `stage` says which of the two agent calls was rejected, so a caller can report where normalization stopped
    instead of leaving the reader to guess it from the paths of the issues.
    """

    def __init__(self, stage: ConceptNormalizationStage, issues: tuple[ConceptNormalizationIssue, ...]) -> None:
        super().__init__("; ".join(f"{issue.path}: {issue.message}" for issue in issues))
        self.stage = stage
        self.issues = issues


class MissingPaperExtractionsError(Exception):
    """Raised when papers of the index have no extraction, so normalizing them all is not possible yet."""

    def __init__(self, paper_ids: tuple[int, ...]) -> None:
        listed = ", ".join(str(paper_id) for paper_id in paper_ids)
        super().__init__(f"these papers have no extraction yet: {listed}")
        self.paper_ids = paper_ids


def build_concept_normalization_schema() -> dict[str, object]:
    """Render the schema of the stored artifact; it is generated so nothing can drift from the model."""
    return ConceptNormalization.model_json_schema()


def build_concept_merge_schema() -> dict[str, object]:
    """Render the schema the merge stage must follow; it is generated so the prompt cannot drift from the model."""
    return ConceptMerge.model_json_schema()


def build_general_knowledge_relation_proposal_schema() -> dict[str, object]:
    """Render the schema the relation stage must follow; it is generated so the prompt cannot drift."""
    return GeneralKnowledgeRelationProposal.model_json_schema()


class ConceptNormalizationRun(Entity):
    """What one normalization run produced, with the papers it covered and the attempts each stage took."""

    normalization: ConceptNormalization
    merge_attempts: int = Field(ge=1)
    relation_attempts: int = Field(ge=1)
    paper_ids: tuple[int, ...]
