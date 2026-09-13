from typing import Any

import pytest

from rkgk.domain.agents import StructuredOutputAgent, StructuredOutputAgentError
from rkgk.domain.models.concept_normalization import (
    ConceptNormalizationValidationError,
    MissingPaperExtractionsError,
)
from rkgk.domain.models.paper_extraction import (
    ExtractedConcept,
    ExtractedConceptEdge,
    PageEvidence,
    PaperExtraction,
)
from rkgk.domain.models.vocabulary import ConceptRelationType, ConceptType
from rkgk.infrastructure.fake_embedder import FakeEmbedder
from rkgk.usecase.normalize_concepts import NormalizeConceptsUseCase
from tests.usecase.fakes import (
    FakeConceptNormalizationRepository,
    FakePaperExtractionRepository,
    FakePaperRepository,
    build_paper,
)

PAPERS = {
    1: build_paper(1, "We study a retrieval-augmented generation pipeline.\n"),
    2: build_paper(2, "We use a knowledge graph as a retrieval backbone.\n"),
}


def build_extraction(paper_id: int, *names: str) -> PaperExtraction:
    return PaperExtraction(
        schema_version=1,
        paper_id=paper_id,
        summary_ja="この論文は検索の手法を提案する。",
        concepts=tuple(
            ExtractedConcept(local_id=f"c{index}", name=name, type=ConceptType.METHOD)
            for index, name in enumerate(names, start=1)
        ),
        paper_concepts=(),
    )


# Both papers extract the same two ideas, so every concept has a partner and the grouping has something to do.
EXTRACTIONS = (
    build_extraction(1, "Retrieval-Augmented Generation", "Knowledge Graph"),
    build_extraction(2, "RAG", "KG"),
)

GROUPING: dict[str, Any] = {
    "groups": [
        [{"paper_id": 1, "local_id": "c1"}, {"paper_id": 2, "local_id": "c1"}],
        [{"paper_id": 1, "local_id": "c2"}, {"paper_id": 2, "local_id": "c2"}],
    ]
}

EMPTY_GROUPING: dict[str, Any] = {"groups": []}

UNKNOWN_GROUPING: dict[str, Any] = {"groups": [[{"paper_id": 1, "local_id": "c1"}, {"paper_id": 2, "local_id": "c9"}]]}

RAG_MERGE: dict[str, Any] = {
    "concepts": [
        {
            "id": "retrieval-augmented-generation",
            "canonical_name": "Retrieval-Augmented Generation",
            "type": "method",
            "aliases": ["RAG"],
            "merged_from": [{"paper_id": 1, "local_id": "c1"}, {"paper_id": 2, "local_id": "c1"}],
        }
    ]
}

GRAPH_MERGE: dict[str, Any] = {
    "concepts": [
        {
            "id": "knowledge-graph",
            "canonical_name": "Knowledge Graph",
            "type": "method",
            "aliases": ["KG"],
            "merged_from": [{"paper_id": 1, "local_id": "c2"}, {"paper_id": 2, "local_id": "c2"}],
        }
    ]
}

# Both groups pick `knowledge-graph`, which no single call can see and stage 3 has to repair.
COLLIDING_RAG_MERGE: dict[str, Any] = {
    "concepts": [
        {
            "id": "knowledge-graph",
            "canonical_name": "Knowledge Graph",
            "type": "method",
            "merged_from": [{"paper_id": 1, "local_id": "c1"}, {"paper_id": 2, "local_id": "c1"}],
        }
    ]
}

JOINED_MERGE: dict[str, Any] = {
    "concepts": [
        {
            "id": "knowledge-graph",
            "canonical_name": "Knowledge Graph",
            "type": "method",
            "merged_from": [
                {"paper_id": 1, "local_id": "c1"},
                {"paper_id": 2, "local_id": "c1"},
                {"paper_id": 1, "local_id": "c2"},
                {"paper_id": 2, "local_id": "c2"},
            ],
        }
    ]
}

INCOMPLETE_RAG_MERGE: dict[str, Any] = {
    "concepts": [
        {
            "id": "retrieval-augmented-generation",
            "canonical_name": "Retrieval-Augmented Generation",
            "type": "method",
            "merged_from": [{"paper_id": 1, "local_id": "c1"}],
        }
    ]
}

VALID_RELATIONS: dict[str, Any] = {
    "concept_relations": [
        {
            "source_id": "knowledge-graph",
            "target_id": "retrieval-augmented-generation",
            "relation": "used_for",
            "rationale": "A knowledge graph supplies the structure the pipeline walks.",
        }
    ]
}

UNKNOWN_RELATIONS: dict[str, Any] = {
    "concept_relations": [
        {
            "source_id": "dense-retrieval",
            "target_id": "retrieval-augmented-generation",
            "relation": "used_for",
            "rationale": "Dense retrieval finds the passages the pipeline generates from.",
        }
    ]
}

NO_RELATIONS: dict[str, Any] = {"concept_relations": []}

# Paper 1 states with its own local ids the relation `VALID_RELATIONS` proposes between the merged slugs.
STATING_EXTRACTIONS = (
    EXTRACTIONS[0].model_copy(
        update={
            "concept_relations": (
                ExtractedConceptEdge(
                    source_id="c2",
                    target_id="c1",
                    relation=ConceptRelationType.USED_FOR,
                    evidence=(PageEvidence(page=1, quote="We use a knowledge graph as a retrieval backbone."),),
                ),
            )
        }
    ),
    EXTRACTIONS[1],
)

GENERAL_RELATIONS: dict[str, Any] = {
    "concept_relations": [
        {
            "source_id": "retrieval-augmented-generation",
            "target_id": "knowledge-graph",
            "relation": "related_to",
            "rationale": "Both organise the passages an answer is grounded in.",
        }
    ]
}


class FakeAgent:
    """Answers with one canned payload per call, so a test decides what each stage and each attempt gets.

    A payload may instead be keyed by a line the prompt has to hold, because the merge stage asks about several
    groups at once and the order the threads reach the agent in is not the test's to decide.
    """

    def __init__(self, *payloads: object, keyed: dict[str, list[object]] | None = None) -> None:
        self._payloads = list(payloads)
        self._keyed = {key: list(values) for key, values in (keyed or {}).items()}
        self.prompts: list[str] = []

    def answer(self, prompt: str, schema: dict[str, object]) -> object:
        self.prompts.append(prompt)
        for key, values in self._keyed.items():
            if key in prompt and values:
                return values.pop(0)
        return self._payloads.pop(0)


class FailingAgent:
    def answer(self, prompt: str, schema: dict[str, object]) -> object:
        raise StructuredOutputAgentError("claude was not found, so no normalization can run")


RAG_GROUP_LINE = "- 1:c1 | Retrieval-Augmented Generation"
GRAPH_GROUP_LINE = "- 1:c2 | Knowledge Graph"


def build_use_case(
    agent: StructuredOutputAgent,
    saved: FakeConceptNormalizationRepository,
    extractions: tuple[PaperExtraction, ...] = EXTRACTIONS,
    max_attempts: int = 3,
    # One worker at a time by default, so the prompts of the merge stage are recorded in group order.
    concurrency: int = 1,
) -> NormalizeConceptsUseCase:
    return NormalizeConceptsUseCase(
        FakePaperRepository(PAPERS),
        FakePaperExtractionRepository(*extractions),
        agent,
        FakeEmbedder(),
        saved,
        max_attempts=max_attempts,
        concurrency=concurrency,
    )


def test_a_grouping_a_merge_per_group_and_the_relations_are_saved_after_one_attempt_each() -> None:
    saved = FakeConceptNormalizationRepository()
    agent = FakeAgent(GROUPING, RAG_MERGE, GRAPH_MERGE, VALID_RELATIONS)
    result = build_use_case(agent, saved).execute()
    assert result.grouping_attempts == 1
    assert result.groups == 2
    assert result.merge_calls == 2
    assert result.merge_rounds == 1
    assert result.relation_attempts == 1
    assert result.paper_ids == (1, 2)
    assert saved.saved == [result.normalization]


def test_the_saved_normalization_joins_the_concepts_of_every_group_with_the_relations() -> None:
    result = build_use_case(
        FakeAgent(GROUPING, RAG_MERGE, GRAPH_MERGE, VALID_RELATIONS), FakeConceptNormalizationRepository()
    ).execute()
    normalization = result.normalization
    assert normalization.schema_version == 1
    assert [concept.id for concept in normalization.concepts] == [
        "retrieval-augmented-generation",
        "knowledge-graph",
    ]
    assert [edge.source_id for edge in normalization.concept_relations] == ["knowledge-graph"]


def test_the_grouping_prompt_holds_every_concept_and_the_candidate_pairs() -> None:
    agent = FakeAgent(GROUPING, RAG_MERGE, GRAPH_MERGE, VALID_RELATIONS)
    build_use_case(agent, FakeConceptNormalizationRepository()).execute()
    assert '<paper id="1">' in agent.prompts[0]
    assert "- c1 | Retrieval-Augmented Generation | method" in agent.prompts[0]
    assert "# Candidate pairs" in agent.prompts[0]


def test_each_merge_prompt_holds_only_the_concepts_of_its_own_group() -> None:
    agent = FakeAgent(GROUPING, RAG_MERGE, GRAPH_MERGE, VALID_RELATIONS)
    build_use_case(agent, FakeConceptNormalizationRepository()).execute()
    assert "- 1:c1 | Retrieval-Augmented Generation | method" in agent.prompts[1]
    assert "1:c2" not in agent.prompts[1]
    assert "- 1:c2 | Knowledge Graph | method" in agent.prompts[2]
    assert "1:c1" not in agent.prompts[2]


def test_the_relation_prompt_lists_the_slugs_the_merge_settled_on_and_no_paper() -> None:
    agent = FakeAgent(GROUPING, RAG_MERGE, GRAPH_MERGE, VALID_RELATIONS)
    build_use_case(agent, FakeConceptNormalizationRepository()).execute()
    assert "- retrieval-augmented-generation | Retrieval-Augmented Generation | method" in agent.prompts[3]
    assert "- knowledge-graph | Knowledge Graph | method" in agent.prompts[3]
    assert '<paper id="1">' not in agent.prompts[3]


def test_the_relation_prompt_lists_the_relations_the_papers_state_in_the_slugs_of_the_merge() -> None:
    agent = FakeAgent(GROUPING, RAG_MERGE, GRAPH_MERGE, GENERAL_RELATIONS)
    build_use_case(agent, FakeConceptNormalizationRepository(), extractions=STATING_EXTRACTIONS).execute()
    assert "# Paper-stated relations" in agent.prompts[3]
    assert "- knowledge-graph | used_for | retrieval-augmented-generation" in agent.prompts[3]


def test_a_concept_no_group_holds_is_normalized_without_asking_the_agent() -> None:
    saved = FakeConceptNormalizationRepository()
    agent = FakeAgent(EMPTY_GROUPING, NO_RELATIONS)
    result = build_use_case(agent, saved).execute()
    assert result.groups == 0
    assert result.merge_calls == 0
    assert result.merge_rounds == 1
    assert len(agent.prompts) == 2
    assert [concept.id for concept in result.normalization.concepts] == [
        "retrieval-augmented-generation",
        "knowledge-graph",
        "rag",
        "kg",
    ]
    assert saved.saved == [result.normalization]


def test_a_solo_concept_keeps_the_name_the_paper_gave_it_as_its_canonical_name() -> None:
    result = build_use_case(FakeAgent(EMPTY_GROUPING, NO_RELATIONS), FakeConceptNormalizationRepository()).execute()
    solo = result.normalization.concepts[2]
    assert solo.canonical_name == "RAG"
    assert [ref.paper_id for ref in solo.merged_from] == [2]


def test_a_grouping_that_names_a_concept_no_paper_extracted_is_retried_with_the_issue() -> None:
    saved = FakeConceptNormalizationRepository()
    agent = FakeAgent(UNKNOWN_GROUPING, GROUPING, RAG_MERGE, GRAPH_MERGE, VALID_RELATIONS)
    result = build_use_case(agent, saved).execute()
    assert result.grouping_attempts == 2
    assert "Previous attempt" not in agent.prompts[0]
    assert "- groups[0][1]: paper 2 has no concept 'c9'" in agent.prompts[1]
    assert saved.saved == [result.normalization]


def test_a_grouping_that_never_passes_is_raised_as_the_grouping_stage_and_nothing_is_merged() -> None:
    saved = FakeConceptNormalizationRepository()
    agent = FakeAgent(UNKNOWN_GROUPING, UNKNOWN_GROUPING)
    with pytest.raises(ConceptNormalizationValidationError, match="has no concept 'c9'") as caught:
        build_use_case(agent, saved, max_attempts=2).execute()
    assert caught.value.stage == "grouping"
    assert len(agent.prompts) == 2
    assert saved.saved == []


def test_a_group_merge_that_drops_a_concept_of_its_group_is_retried_on_its_own() -> None:
    saved = FakeConceptNormalizationRepository()
    agent = FakeAgent(GROUPING, INCOMPLETE_RAG_MERGE, RAG_MERGE, GRAPH_MERGE, VALID_RELATIONS)
    result = build_use_case(agent, saved).execute()
    assert result.merge_calls == 3
    assert result.merge_rounds == 1
    assert "- concepts: paper 2 'c1' is in no merged_from" in agent.prompts[2]
    assert saved.saved == [result.normalization]


def test_a_group_that_never_passes_is_raised_as_the_merge_stage_naming_the_group() -> None:
    saved = FakeConceptNormalizationRepository()
    agent = FakeAgent(GROUPING, INCOMPLETE_RAG_MERGE, INCOMPLETE_RAG_MERGE, GRAPH_MERGE)
    with pytest.raises(ConceptNormalizationValidationError, match="is in no merged_from") as caught:
        build_use_case(agent, saved, max_attempts=2).execute()
    assert caught.value.stage == "merge"
    assert [issue.path for issue in caught.value.issues] == ["groups[0].concepts"]
    assert saved.saved == []


def test_two_groups_that_picked_the_same_slug_are_joined_and_merged_again() -> None:
    saved = FakeConceptNormalizationRepository()
    agent = FakeAgent(
        GROUPING,
        COLLIDING_RAG_MERGE,
        GRAPH_MERGE,
        JOINED_MERGE,
        NO_RELATIONS,
    )
    result = build_use_case(agent, saved).execute()
    assert result.merge_rounds == 2
    assert result.merge_calls == 3
    assert [concept.id for concept in result.normalization.concepts] == ["knowledge-graph"]
    # The third merge prompt holds both groups at once, which is what joining them means.
    assert "1:c1" in agent.prompts[3]
    assert "2:c2" in agent.prompts[3]
    assert saved.saved == [result.normalization]


def test_a_collision_that_is_not_repaired_within_the_rounds_allowed_is_raised_as_the_merge_stage() -> None:
    saved = FakeConceptNormalizationRepository()
    agent = FakeAgent(GROUPING, COLLIDING_RAG_MERGE, GRAPH_MERGE)
    with pytest.raises(ConceptNormalizationValidationError, match="'knowledge-graph'") as caught:
        build_use_case(agent, saved, max_attempts=1).execute()
    assert caught.value.stage == "merge"
    assert [issue.path for issue in caught.value.issues] == ["concepts"]
    assert "paper 1 'c1', paper 2 'c1'; paper 1 'c2', paper 2 'c2'" in caught.value.issues[0].message
    assert saved.saved == []


def test_several_groups_asked_about_at_once_each_get_the_answer_to_their_own_group() -> None:
    saved = FakeConceptNormalizationRepository()
    agent = FakeAgent(
        GROUPING,
        VALID_RELATIONS,
        keyed={RAG_GROUP_LINE: [RAG_MERGE], GRAPH_GROUP_LINE: [GRAPH_MERGE]},
    )
    result = build_use_case(agent, saved, concurrency=4).execute()
    assert result.merge_calls == 2
    assert [concept.id for concept in result.normalization.concepts] == [
        "retrieval-augmented-generation",
        "knowledge-graph",
    ]
    assert saved.saved == [result.normalization]


def test_a_relation_on_a_slug_the_merge_never_declared_is_retried_without_merging_again() -> None:
    saved = FakeConceptNormalizationRepository()
    agent = FakeAgent(GROUPING, RAG_MERGE, GRAPH_MERGE, UNKNOWN_RELATIONS, VALID_RELATIONS)
    result = build_use_case(agent, saved).execute()
    assert result.merge_calls == 2
    assert result.relation_attempts == 2
    assert len(agent.prompts) == 5
    assert saved.saved == [result.normalization]


def test_an_answer_that_is_not_a_grouping_at_all_is_retried_with_the_schema_issue() -> None:
    agent = FakeAgent("not an object", GROUPING, RAG_MERGE, GRAPH_MERGE, VALID_RELATIONS)
    result = build_use_case(agent, FakeConceptNormalizationRepository()).execute()
    assert result.grouping_attempts == 2
    assert "## Issues" in agent.prompts[1]


def test_an_answer_that_is_not_a_merge_at_all_is_retried_with_the_schema_issue() -> None:
    agent = FakeAgent(GROUPING, "not an object", RAG_MERGE, GRAPH_MERGE, VALID_RELATIONS)
    result = build_use_case(agent, FakeConceptNormalizationRepository()).execute()
    assert result.merge_calls == 3
    assert "## Issues" in agent.prompts[2]


def test_relations_that_never_pass_leave_the_accepted_merge_unsaved() -> None:
    saved = FakeConceptNormalizationRepository()
    agent = FakeAgent(GROUPING, RAG_MERGE, GRAPH_MERGE, UNKNOWN_RELATIONS, UNKNOWN_RELATIONS)
    with pytest.raises(ConceptNormalizationValidationError, match="not one of the normalized concepts") as caught:
        build_use_case(agent, saved, max_attempts=2).execute()
    assert caught.value.stage == "relations"
    assert [issue.path for issue in caught.value.issues] == ["concept_relations[0].source_id"]
    assert saved.saved == []


def test_every_paper_without_an_extraction_is_reported_before_the_agent_is_asked() -> None:
    agent = FakeAgent(GROUPING, RAG_MERGE, GRAPH_MERGE, VALID_RELATIONS)
    with pytest.raises(MissingPaperExtractionsError, match="2") as caught:
        build_use_case(agent, FakeConceptNormalizationRepository(), extractions=(EXTRACTIONS[0],)).execute()
    assert caught.value.paper_ids == (2,)
    assert agent.prompts == []


def test_an_index_without_papers_is_reported_before_any_extraction_is_read() -> None:
    agent = FakeAgent(GROUPING, RAG_MERGE, GRAPH_MERGE, VALID_RELATIONS)
    use_case = NormalizeConceptsUseCase(
        FakePaperRepository({}),
        FakePaperExtractionRepository(),
        agent,
        FakeEmbedder(),
        FakeConceptNormalizationRepository(),
    )
    with pytest.raises(ValueError, match="no papers in the index"):
        use_case.execute()
    assert agent.prompts == []


def test_the_agent_error_is_propagated() -> None:
    with pytest.raises(StructuredOutputAgentError, match="was not found"):
        build_use_case(FailingAgent(), FakeConceptNormalizationRepository()).execute()


def build_broken_use_case(**overrides: int) -> NormalizeConceptsUseCase:
    return NormalizeConceptsUseCase(
        FakePaperRepository(PAPERS),
        FakePaperExtractionRepository(*EXTRACTIONS),
        FakeAgent(GROUPING),
        FakeEmbedder(),
        FakeConceptNormalizationRepository(),
        **overrides,
    )


@pytest.mark.parametrize(
    ("setting", "value"),
    [("max_attempts", 0), ("max_attempts", -1), ("neighbors", 0), ("concurrency", 0)],
)
def test_a_setting_below_one_is_rejected_before_anything_runs(setting: str, value: int) -> None:
    with pytest.raises(ValueError, match=f"{setting} must be at least 1"):
        build_broken_use_case(**{setting: value})
