import pytest

from rkgk.domain.agents import StructuredOutputAgent, StructuredOutputAgentError
from rkgk.domain.models.concept_normalization import (
    ConceptNormalizationValidationError,
    MissingPaperExtractionsError,
)
from rkgk.domain.models.paper_extraction import ExtractedConcept, PaperExtraction
from rkgk.domain.models.vocabulary import ConceptType
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

VALID_PAYLOAD = {
    "schema_version": 1,
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
    ],
    "concept_relations": [
        {
            "source_id": "knowledge-graph",
            "target_id": "retrieval-augmented-generation",
            "relation": "used_for",
            "rationale": "A knowledge graph supplies the structure the pipeline walks.",
        }
    ],
}

INCOMPLETE_PAYLOAD = {"schema_version": 1, "concepts": VALID_PAYLOAD["concepts"][:1]}


class FakeAgent:
    """Answers with one canned payload per call, so a test decides how many attempts it takes to converge."""

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


def test_an_answer_that_passes_validation_is_saved_after_one_attempt() -> None:
    saved = FakeConceptNormalizationRepository()
    outcome = build_use_case(FakeAgent(VALID_PAYLOAD), saved).execute()
    assert outcome.attempts == 1
    assert outcome.paper_ids == (1, 2)
    assert saved.saved == [outcome.normalization]


def test_the_prompt_holds_the_concepts_of_every_extracted_paper() -> None:
    agent = FakeAgent(VALID_PAYLOAD)
    build_use_case(agent, FakeConceptNormalizationRepository()).execute()
    assert "## Paper 1" in agent.prompts[0]
    assert "- c1 | Knowledge Graph | method" in agent.prompts[0]


def test_an_answer_that_drops_a_concept_is_retried_and_the_attempts_are_counted() -> None:
    saved = FakeConceptNormalizationRepository()
    outcome = build_use_case(FakeAgent(INCOMPLETE_PAYLOAD, VALID_PAYLOAD), saved).execute()
    assert outcome.attempts == 2
    assert saved.saved == [outcome.normalization]


def test_the_retry_shows_the_agent_its_rejected_answer_and_the_issues() -> None:
    agent = FakeAgent(INCOMPLETE_PAYLOAD, VALID_PAYLOAD)
    build_use_case(agent, FakeConceptNormalizationRepository()).execute()
    assert "Previous attempt" not in agent.prompts[0]
    assert '"canonical_name": "Retrieval-Augmented Generation"' in agent.prompts[1]
    assert "- concepts: paper 2 'c1' is in no merged_from" in agent.prompts[1]


def test_an_answer_that_is_not_a_normalization_at_all_is_retried_with_the_schema_issue() -> None:
    agent = FakeAgent("not an object", VALID_PAYLOAD)
    outcome = build_use_case(agent, FakeConceptNormalizationRepository()).execute()
    assert outcome.attempts == 2
    assert "## Issues" in agent.prompts[1]


def test_the_last_rejection_is_raised_and_nothing_is_saved() -> None:
    saved = FakeConceptNormalizationRepository()
    agent = FakeAgent(INCOMPLETE_PAYLOAD, INCOMPLETE_PAYLOAD)
    with pytest.raises(ConceptNormalizationValidationError, match="is in no merged_from"):
        build_use_case(agent, saved, max_attempts=2).execute()
    assert saved.saved == []


def test_the_agent_is_asked_only_as_often_as_the_attempts_allow() -> None:
    agent = FakeAgent(INCOMPLETE_PAYLOAD, INCOMPLETE_PAYLOAD, VALID_PAYLOAD)
    with pytest.raises(ConceptNormalizationValidationError):
        build_use_case(agent, FakeConceptNormalizationRepository(), max_attempts=2).execute()
    assert len(agent.prompts) == 2


def test_every_paper_without_an_extraction_is_reported_before_the_agent_is_asked() -> None:
    agent = FakeAgent(VALID_PAYLOAD)
    with pytest.raises(MissingPaperExtractionsError, match="2") as caught:
        build_use_case(agent, FakeConceptNormalizationRepository(), extractions=(EXTRACTIONS[0],)).execute()
    assert caught.value.paper_ids == (2,)
    assert agent.prompts == []


def test_an_index_without_papers_is_reported_before_any_extraction_is_read() -> None:
    agent = FakeAgent(VALID_PAYLOAD)
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
        build_use_case(FakeAgent(VALID_PAYLOAD), FakeConceptNormalizationRepository(), max_attempts=max_attempts)
