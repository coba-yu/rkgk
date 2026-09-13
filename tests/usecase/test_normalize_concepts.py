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


def build_extraction(paper_id: int, name: str) -> PaperExtraction:
    return PaperExtraction(
        schema_version=1,
        paper_id=paper_id,
        summary_ja="この論文は検索の手法を提案する。",
        concepts=(ExtractedConcept(local_id="c1", name=name, type=ConceptType.METHOD),),
        paper_concepts=(),
    )


EXTRACTIONS = (
    build_extraction(1, "Retrieval-Augmented Generation"),
    build_extraction(2, "Knowledge Graph"),
)

VALID_MERGE = {
    "concepts": [
        {
            "id": "retrieval-augmented-generation",
            "canonical_name": "Retrieval-Augmented Generation",
            "type": "method",
            "merged_from": [{"paper_id": 1, "local_id": "c1"}],
        },
        {
            "id": "knowledge-graph",
            "canonical_name": "Knowledge Graph",
            "type": "method",
            "merged_from": [{"paper_id": 2, "local_id": "c1"}],
        },
    ]
}

INCOMPLETE_MERGE = {"concepts": VALID_MERGE["concepts"][:1]}

VALID_RELATIONS = {
    "concept_relations": [
        {
            "source_id": "knowledge-graph",
            "target_id": "retrieval-augmented-generation",
            "relation": "used_for",
            "rationale": "A knowledge graph supplies the structure the pipeline walks.",
        }
    ]
}

# Paper 1 states with its own local ids the relation `VALID_RELATIONS` proposes between the merged slugs.
STATING_EXTRACTIONS = (
    PaperExtraction(
        schema_version=1,
        paper_id=1,
        summary_ja="この論文は知識グラフを使う検索の手法を提案する。",
        concepts=(
            ExtractedConcept(local_id="c1", name="Retrieval-Augmented Generation", type=ConceptType.METHOD),
            ExtractedConcept(local_id="c2", name="Knowledge Graph", type=ConceptType.METHOD),
        ),
        paper_concepts=(),
        concept_relations=(
            ExtractedConceptEdge(
                source_id="c2",
                target_id="c1",
                relation=ConceptRelationType.USED_FOR,
                evidence=(PageEvidence(page=1, quote="We use a knowledge graph as a retrieval backbone."),),
            ),
        ),
    ),
    build_extraction(2, "Knowledge Graph"),
)

STATING_MERGE = {
    "concepts": [
        {
            "id": "retrieval-augmented-generation",
            "canonical_name": "Retrieval-Augmented Generation",
            "type": "method",
            "merged_from": [{"paper_id": 1, "local_id": "c1"}],
        },
        {
            "id": "knowledge-graph",
            "canonical_name": "Knowledge Graph",
            "type": "method",
            "merged_from": [{"paper_id": 1, "local_id": "c2"}, {"paper_id": 2, "local_id": "c1"}],
        },
    ]
}

GENERAL_RELATIONS = {
    "concept_relations": [
        {
            "source_id": "retrieval-augmented-generation",
            "target_id": "knowledge-graph",
            "relation": "related_to",
            "rationale": "Both organise the passages an answer is grounded in.",
        }
    ]
}

UNKNOWN_RELATIONS = {
    "concept_relations": [
        {
            "source_id": "dense-retrieval",
            "target_id": "retrieval-augmented-generation",
            "relation": "used_for",
            "rationale": "Dense retrieval finds the passages the pipeline generates from.",
        }
    ]
}


class FakeAgent:
    """Answers with one canned payload per call, so a test decides what each stage and each attempt gets."""

    def __init__(self, *payloads: object) -> None:
        self._payloads = list(payloads)
        self.prompts: list[str] = []

    def answer(self, prompt: str, schema: dict[str, object]) -> object:
        self.prompts.append(prompt)
        return self._payloads.pop(0)


class FailingAgent:
    def answer(self, prompt: str, schema: dict[str, object]) -> object:
        raise StructuredOutputAgentError("claude was not found, so no normalization can run")


def build_use_case(
    agent: StructuredOutputAgent,
    saved: FakeConceptNormalizationRepository,
    extractions: tuple[PaperExtraction, ...] = EXTRACTIONS,
    max_attempts: int = 3,
) -> NormalizeConceptsUseCase:
    return NormalizeConceptsUseCase(
        FakePaperRepository(PAPERS),
        FakePaperExtractionRepository(*extractions),
        agent,
        saved,
        max_attempts=max_attempts,
    )


def test_an_answer_to_each_stage_that_passes_validation_is_saved_after_one_attempt_each() -> None:
    saved = FakeConceptNormalizationRepository()
    result = build_use_case(FakeAgent(VALID_MERGE, VALID_RELATIONS), saved).execute()
    assert result.merge_attempts == 1
    assert result.relation_attempts == 1
    assert result.paper_ids == (1, 2)
    assert saved.saved == [result.normalization]


def test_the_saved_normalization_joins_the_concepts_of_one_stage_with_the_relations_of_the_other() -> None:
    result = build_use_case(FakeAgent(VALID_MERGE, VALID_RELATIONS), FakeConceptNormalizationRepository()).execute()
    normalization = result.normalization
    assert normalization.schema_version == 1
    assert [concept.id for concept in normalization.concepts] == [
        "retrieval-augmented-generation",
        "knowledge-graph",
    ]
    assert [edge.source_id for edge in normalization.concept_relations] == ["knowledge-graph"]


def test_the_merge_prompt_holds_the_concepts_of_every_extracted_paper() -> None:
    agent = FakeAgent(VALID_MERGE, VALID_RELATIONS)
    build_use_case(agent, FakeConceptNormalizationRepository()).execute()
    assert '<paper id="1">' in agent.prompts[0]
    assert "- c1 | Knowledge Graph | method" in agent.prompts[0]


def test_the_relation_prompt_lists_the_slugs_the_merge_settled_on_and_no_paper() -> None:
    agent = FakeAgent(VALID_MERGE, VALID_RELATIONS)
    build_use_case(agent, FakeConceptNormalizationRepository()).execute()
    assert "- retrieval-augmented-generation | Retrieval-Augmented Generation | method" in agent.prompts[1]
    assert "- knowledge-graph | Knowledge Graph | method" in agent.prompts[1]
    assert '<paper id="1">' not in agent.prompts[1]


def test_the_relation_prompt_lists_the_relations_the_papers_state_in_the_slugs_of_the_merge() -> None:
    agent = FakeAgent(STATING_MERGE, GENERAL_RELATIONS)
    build_use_case(agent, FakeConceptNormalizationRepository(), extractions=STATING_EXTRACTIONS).execute()
    assert "# Paper-stated relations" in agent.prompts[1]
    assert "- knowledge-graph | used_for | retrieval-augmented-generation" in agent.prompts[1]


def test_the_relation_prompt_says_none_when_no_paper_states_a_relation() -> None:
    agent = FakeAgent(VALID_MERGE, VALID_RELATIONS)
    build_use_case(agent, FakeConceptNormalizationRepository()).execute()
    assert agent.prompts[1].endswith("なし\n")


def test_a_relation_the_papers_already_state_is_retried_until_general_knowledge_adds_something() -> None:
    agent = FakeAgent(STATING_MERGE, VALID_RELATIONS, GENERAL_RELATIONS)
    saved = FakeConceptNormalizationRepository()
    result = build_use_case(agent, saved, extractions=STATING_EXTRACTIONS).execute()
    assert result.merge_attempts == 1
    assert result.relation_attempts == 2
    assert [edge.relation.value for edge in result.normalization.concept_relations] == ["related_to"]
    assert saved.saved == [result.normalization]


def test_the_retry_after_a_repeated_relation_names_the_paper_that_states_it() -> None:
    agent = FakeAgent(STATING_MERGE, VALID_RELATIONS, GENERAL_RELATIONS)
    build_use_case(agent, FakeConceptNormalizationRepository(), extractions=STATING_EXTRACTIONS).execute()
    assert (
        "- concept_relations[0]: repeats what paper 1 states: knowledge-graph used_for "
        "retrieval-augmented-generation" in agent.prompts[2]
    )


def test_a_merge_that_drops_a_concept_is_retried_and_only_its_own_attempts_are_counted() -> None:
    saved = FakeConceptNormalizationRepository()
    result = build_use_case(FakeAgent(INCOMPLETE_MERGE, VALID_MERGE, VALID_RELATIONS), saved).execute()
    assert result.merge_attempts == 2
    assert result.relation_attempts == 1
    assert saved.saved == [result.normalization]


def test_the_merge_retry_shows_the_agent_its_rejected_answer_and_the_issues() -> None:
    agent = FakeAgent(INCOMPLETE_MERGE, VALID_MERGE, VALID_RELATIONS)
    build_use_case(agent, FakeConceptNormalizationRepository()).execute()
    assert "Previous attempt" not in agent.prompts[0]
    assert '"canonical_name": "Retrieval-Augmented Generation"' in agent.prompts[1]
    assert "- concepts: paper 2 'c1' is in no merged_from" in agent.prompts[1]


def test_a_relation_on_a_slug_the_merge_never_declared_is_retried_without_asking_for_the_merge_again() -> None:
    agent = FakeAgent(VALID_MERGE, UNKNOWN_RELATIONS, VALID_RELATIONS)
    saved = FakeConceptNormalizationRepository()
    result = build_use_case(agent, saved).execute()
    assert result.merge_attempts == 1
    assert result.relation_attempts == 2
    assert len(agent.prompts) == 3
    assert saved.saved == [result.normalization]


def test_the_relation_retry_shows_the_agent_its_rejected_relations_and_the_issues() -> None:
    agent = FakeAgent(VALID_MERGE, UNKNOWN_RELATIONS, VALID_RELATIONS)
    build_use_case(agent, FakeConceptNormalizationRepository()).execute()
    assert "Previous attempt" not in agent.prompts[1]
    assert '"source_id": "dense-retrieval"' in agent.prompts[2]
    assert (
        "- concept_relations[0].source_id: is 'dense-retrieval', which is not one of the normalized concepts"
        in agent.prompts[2]
    )


def test_an_answer_that_is_not_a_merge_at_all_is_retried_with_the_schema_issue() -> None:
    agent = FakeAgent("not an object", VALID_MERGE, VALID_RELATIONS)
    result = build_use_case(agent, FakeConceptNormalizationRepository()).execute()
    assert result.merge_attempts == 2
    assert "## Issues" in agent.prompts[1]


def test_an_answer_that_is_not_a_proposal_at_all_is_retried_with_the_schema_issue() -> None:
    agent = FakeAgent(VALID_MERGE, "not an object", VALID_RELATIONS)
    result = build_use_case(agent, FakeConceptNormalizationRepository()).execute()
    assert result.relation_attempts == 2
    assert "## Issues" in agent.prompts[2]


def test_a_merge_that_never_passes_is_raised_as_the_merge_stage_and_the_relations_are_never_asked_for() -> None:
    saved = FakeConceptNormalizationRepository()
    agent = FakeAgent(INCOMPLETE_MERGE, INCOMPLETE_MERGE)
    with pytest.raises(ConceptNormalizationValidationError, match="is in no merged_from") as caught:
        build_use_case(agent, saved, max_attempts=2).execute()
    assert caught.value.stage == "merge"
    assert len(agent.prompts) == 2
    assert saved.saved == []


def test_relations_that_never_pass_leave_the_accepted_merge_unsaved() -> None:
    saved = FakeConceptNormalizationRepository()
    agent = FakeAgent(VALID_MERGE, UNKNOWN_RELATIONS, UNKNOWN_RELATIONS)
    with pytest.raises(ConceptNormalizationValidationError, match="not one of the normalized concepts") as caught:
        build_use_case(agent, saved, max_attempts=2).execute()
    assert caught.value.stage == "relations"
    assert [issue.path for issue in caught.value.issues] == ["concept_relations[0].source_id"]
    assert saved.saved == []


def test_each_stage_asks_the_agent_only_as_often_as_the_attempts_allow() -> None:
    agent = FakeAgent(VALID_MERGE, UNKNOWN_RELATIONS, UNKNOWN_RELATIONS, VALID_RELATIONS)
    with pytest.raises(ConceptNormalizationValidationError):
        build_use_case(agent, FakeConceptNormalizationRepository(), max_attempts=2).execute()
    assert len(agent.prompts) == 3


def test_every_paper_without_an_extraction_is_reported_before_the_agent_is_asked() -> None:
    agent = FakeAgent(VALID_MERGE, VALID_RELATIONS)
    with pytest.raises(MissingPaperExtractionsError, match="2") as caught:
        build_use_case(agent, FakeConceptNormalizationRepository(), extractions=(EXTRACTIONS[0],)).execute()
    assert caught.value.paper_ids == (2,)
    assert agent.prompts == []


def test_an_index_without_papers_is_reported_before_any_extraction_is_read() -> None:
    agent = FakeAgent(VALID_MERGE, VALID_RELATIONS)
    use_case = NormalizeConceptsUseCase(
        FakePaperRepository({}),
        FakePaperExtractionRepository(),
        agent,
        FakeConceptNormalizationRepository(),
    )
    with pytest.raises(ValueError, match="no papers in the index"):
        use_case.execute()
    assert agent.prompts == []


def test_the_agent_error_is_propagated() -> None:
    with pytest.raises(StructuredOutputAgentError, match="was not found"):
        build_use_case(FailingAgent(), FakeConceptNormalizationRepository()).execute()


@pytest.mark.parametrize("max_attempts", [0, -1])
def test_fewer_than_one_attempt_is_rejected_before_anything_runs(max_attempts: int) -> None:
    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        build_use_case(FakeAgent(VALID_MERGE), FakeConceptNormalizationRepository(), max_attempts=max_attempts)
