import json
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from rkgk.domain.models.base import SLUG_PATTERN
from rkgk.domain.models.concept_normalization import (
    CONCEPT_NORMALIZATION_SCHEMA_VERSION,
    ConceptMerge,
    ConceptNormalization,
    ConceptNormalizationIssue,
    ConceptNormalizationRun,
    ConceptNormalizationValidationError,
    GeneralKnowledgeEdge,
    GeneralKnowledgeProposal,
    LocalConceptRef,
    MissingPaperExtractionsError,
    NormalizedConcept,
    build_concept_merge_schema,
    build_concept_normalization_schema,
    build_general_knowledge_schema,
)
from rkgk.domain.models.vocabulary import ConceptRelationType, ConceptType

RAG = NormalizedConcept(
    id="retrieval-augmented-generation",
    canonical_name="Retrieval-Augmented Generation",
    type=ConceptType.METHOD,
    aliases=("RAG", "検索拡張生成"),
    merged_from=(LocalConceptRef(paper_id=1, local_id="c1"), LocalConceptRef(paper_id=2, local_id="c1")),
)
CHUNKING = NormalizedConcept(
    id="page-aligned-chunking",
    canonical_name="Page-Aligned Chunking",
    type=ConceptType.METHOD,
    merged_from=(LocalConceptRef(paper_id=1, local_id="c2"),),
)
KNOWLEDGE_GRAPH = NormalizedConcept(
    id="knowledge-graph",
    canonical_name="Knowledge Graph",
    type=ConceptType.METHOD,
    aliases=("KG",),
    description="A graph of concepts and their relations.",
    merged_from=(LocalConceptRef(paper_id=2, local_id="c2"),),
)

PART_OF = GeneralKnowledgeEdge(
    source_id="page-aligned-chunking",
    target_id="retrieval-augmented-generation",
    relation=ConceptRelationType.PART_OF,
    rationale="Chunking is the indexing step of a retrieval-augmented generation pipeline.",
)


def build_normalization(**overrides: object) -> ConceptNormalization:
    payload: dict[str, object] = {
        "schema_version": 1,
        "concepts": (RAG, CHUNKING, KNOWLEDGE_GRAPH),
        "concept_relations": (PART_OF,),
    }
    payload.update(overrides)
    return ConceptNormalization.model_validate(payload)


def test_a_normalization_declares_the_current_schema_version() -> None:
    assert build_normalization().schema_version == CONCEPT_NORMALIZATION_SCHEMA_VERSION


def test_another_schema_version_is_rejected() -> None:
    with pytest.raises(ValidationError):
        build_normalization(schema_version=2)


def test_the_same_slug_twice_is_rejected_with_that_slug() -> None:
    with pytest.raises(ValidationError, match="'knowledge-graph' more than once"):
        build_normalization(concepts=(RAG, KNOWLEDGE_GRAPH, KNOWLEDGE_GRAPH), concept_relations=())


@pytest.mark.parametrize("slug", ["Knowledge-Graph", "knowledge graph", "knowledge_graph", "-knowledge", ""])
def test_a_slug_outside_the_pattern_is_rejected(slug: str) -> None:
    with pytest.raises(ValidationError):
        NormalizedConcept(
            id=slug,
            canonical_name="Knowledge Graph",
            type=ConceptType.METHOD,
            merged_from=(LocalConceptRef(paper_id=2, local_id="c2"),),
        )


def test_one_extracted_concept_claimed_by_two_normalized_concepts_is_rejected_with_that_reference() -> None:
    stolen = CHUNKING.model_copy(update={"merged_from": (LocalConceptRef(paper_id=1, local_id="c1"),)})
    with pytest.raises(
        ValidationError,
        match="claims paper 1 'c1' for both 'retrieval-augmented-generation' and 'page-aligned-chunking'",
    ):
        build_normalization(concepts=(RAG, stolen), concept_relations=())


def test_a_relation_to_a_concept_that_was_not_declared_is_rejected_with_that_slug() -> None:
    edge = PART_OF.model_copy(update={"target_id": "dense-retrieval"})
    with pytest.raises(ValidationError, match="undeclared concept 'dense-retrieval'"):
        build_normalization(concept_relations=(edge,))


def test_a_relation_from_a_concept_to_itself_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        GeneralKnowledgeEdge(
            source_id="knowledge-graph",
            target_id="knowledge-graph",
            relation=ConceptRelationType.IS_A,
            rationale="A graph is a graph.",
        )


def test_the_same_pair_and_relation_twice_is_rejected() -> None:
    with pytest.raises(ValidationError, match="repeats 'page-aligned-chunking' -> 'retrieval-augmented-generation'"):
        build_normalization(concept_relations=(PART_OF, PART_OF))


def test_the_same_pair_with_another_relation_is_accepted() -> None:
    other = PART_OF.model_copy(update={"relation": ConceptRelationType.USED_FOR})
    assert len(build_normalization(concept_relations=(PART_OF, other)).concept_relations) == 2


@pytest.mark.parametrize("rationale", ["", "   \n "])
def test_a_relation_without_a_rationale_is_rejected(rationale: str) -> None:
    with pytest.raises(ValidationError):
        GeneralKnowledgeEdge(
            source_id="page-aligned-chunking",
            target_id="retrieval-augmented-generation",
            relation=ConceptRelationType.PART_OF,
            rationale=rationale,
        )


def test_a_concept_that_was_merged_from_nothing_is_rejected() -> None:
    with pytest.raises(ValidationError):
        NormalizedConcept(
            id="dense-retrieval", canonical_name="Dense Retrieval", type=ConceptType.METHOD, merged_from=()
        )


def test_a_normalization_without_concepts_is_rejected() -> None:
    with pytest.raises(ValidationError):
        build_normalization(concepts=(), concept_relations=())


def test_a_merge_carries_the_concepts_alone() -> None:
    merge = ConceptMerge(concepts=(RAG, CHUNKING, KNOWLEDGE_GRAPH))
    assert [concept.id for concept in merge.concepts] == [
        "retrieval-augmented-generation",
        "page-aligned-chunking",
        "knowledge-graph",
    ]


def test_a_merge_that_declares_a_schema_version_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ConceptMerge.model_validate({"schema_version": 1, "concepts": [RAG.model_dump()]})


def test_a_merge_that_declares_the_same_slug_twice_is_rejected_with_that_slug() -> None:
    with pytest.raises(ValidationError, match="'knowledge-graph' more than once"):
        ConceptMerge(concepts=(RAG, KNOWLEDGE_GRAPH, KNOWLEDGE_GRAPH))


def test_a_merge_that_gives_one_extracted_concept_to_two_concepts_is_rejected_with_that_reference() -> None:
    stolen = CHUNKING.model_copy(update={"merged_from": (LocalConceptRef(paper_id=1, local_id="c1"),)})
    with pytest.raises(
        ValidationError,
        match="claims paper 1 'c1' for both 'retrieval-augmented-generation' and 'page-aligned-chunking'",
    ):
        ConceptMerge(concepts=(RAG, stolen))


def test_a_merge_without_concepts_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ConceptMerge(concepts=())


def test_a_proposal_without_relations_is_accepted() -> None:
    assert GeneralKnowledgeProposal().concept_relations == ()


def test_a_proposal_that_repeats_a_pair_and_relation_is_rejected() -> None:
    with pytest.raises(ValidationError, match="repeats 'page-aligned-chunking' -> 'retrieval-augmented-generation'"):
        GeneralKnowledgeProposal(concept_relations=(PART_OF, PART_OF))


def test_a_proposal_on_a_slug_no_concept_declares_is_left_to_the_service() -> None:
    edge = PART_OF.model_copy(update={"target_id": "dense-retrieval"})
    assert GeneralKnowledgeProposal(concept_relations=(edge,)).concept_relations == (edge,)


def test_the_schema_describes_the_provenance_and_the_slug_pattern() -> None:
    schema = build_concept_normalization_schema()
    definitions = schema["$defs"]
    assert isinstance(definitions, dict)
    assert "merged_from" in definitions["NormalizedConcept"]["properties"]
    assert SLUG_PATTERN in json.dumps(schema)


def test_the_merge_schema_asks_for_the_concepts_and_for_no_version() -> None:
    schema = build_concept_merge_schema()
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert properties.keys() == {"concepts"}
    assert SLUG_PATTERN in json.dumps(schema)


def test_the_general_knowledge_schema_asks_for_the_relations_with_their_rationale() -> None:
    schema = build_general_knowledge_schema()
    definitions = schema["$defs"]
    properties = schema["properties"]
    assert isinstance(definitions, dict)
    assert isinstance(properties, dict)
    assert properties.keys() == {"concept_relations"}
    assert "rationale" in definitions["GeneralKnowledgeEdge"]["properties"]


@pytest.mark.parametrize(
    "build_schema", [build_concept_normalization_schema, build_concept_merge_schema, build_general_knowledge_schema]
)
def test_the_schema_is_json_serializable(build_schema: Callable[[], dict[str, object]]) -> None:
    assert json.loads(json.dumps(build_schema())) == build_schema()


def test_a_rejection_keeps_the_stage_it_came_from_and_its_issues() -> None:
    issues = (ConceptNormalizationIssue(path="concept_relations[0].source_id", message="is not a concept"),)
    error = ConceptNormalizationValidationError("relations", issues)
    assert error.stage == "relations"
    assert error.issues == issues
    assert "concept_relations[0].source_id: is not a concept" in str(error)


def test_a_run_counts_the_attempts_of_each_stage_apart() -> None:
    run = ConceptNormalizationRun(
        normalization=build_normalization(), merge_attempts=1, relation_attempts=3, paper_ids=(1, 2)
    )
    assert run.paper_ids == (1, 2)
    assert run.merge_attempts == 1
    assert run.relation_attempts == 3


@pytest.mark.parametrize("attempts", [{"merge_attempts": 0}, {"relation_attempts": 0}])
def test_a_run_of_a_stage_that_never_ran_is_rejected(attempts: dict[str, int]) -> None:
    counts = {"merge_attempts": 1, "relation_attempts": 1} | attempts
    with pytest.raises(ValidationError):
        ConceptNormalizationRun(normalization=build_normalization(), paper_ids=(1, 2), **counts)


def test_the_missing_extractions_error_lists_every_paper() -> None:
    error = MissingPaperExtractionsError((2, 3))
    assert error.paper_ids == (2, 3)
    assert "2, 3" in str(error)
