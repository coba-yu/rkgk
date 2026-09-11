import pytest

from rkgk.domain.models.paper_extraction import ExtractionValidationError, Extractor, ExtractorError
from rkgk.domain.repositories.paper import PaperNotFoundError
from rkgk.usecase.extract_paper import ExtractPaperUseCase
from tests.usecase.fakes import FakeExtractionRepository, FakePaperRepository, build_paper

PAPER = build_paper(1, "We study a retrieval-augmented generation pipeline.\n", "The pipeline embeds chunks.\n")

VALID_PAYLOAD = {
    "schema_version": 1,
    "paper_id": 1,
    "summary_ja": "この論文は検索拡張生成のパイプラインを提案する。",
    "concepts": [{"local_id": "c1", "name": "Retrieval-Augmented Generation", "type": "method"}],
    "paper_concepts": [
        {
            "concept_id": "c1",
            "relation": "proposes",
            "evidence": [{"page": 1, "quote": "retrieval-augmented generation pipeline"}],
        }
    ],
}

INVALID_PAYLOAD = {
    **VALID_PAYLOAD,
    "paper_concepts": [
        {
            "concept_id": "c1",
            "relation": "proposes",
            "evidence": [{"page": 1, "quote": "a sentence the paper never wrote"}],
        }
    ],
}


class FakeExtractor:
    """Answers with one canned payload per call, so a test decides how many attempts it takes to converge."""

    def __init__(self, *payloads: object) -> None:
        self._payloads = list(payloads)
        self.prompts: list[str] = []

    def extract(self, prompt: str, schema: dict[str, object]) -> object:
        self.prompts.append(prompt)
        return self._payloads.pop(0)


class FailingExtractor:
    def extract(self, prompt: str, schema: dict[str, object]) -> object:
        raise ExtractorError("claude was not found, so no extraction can run")


def build_use_case(
    extractor: Extractor, saved: FakeExtractionRepository, max_attempts: int = 3
) -> ExtractPaperUseCase:
    return ExtractPaperUseCase(FakePaperRepository({1: PAPER}), extractor, saved, max_attempts=max_attempts)


def test_an_answer_that_passes_validation_is_saved_after_one_attempt() -> None:
    saved = FakeExtractionRepository()
    outcome = build_use_case(FakeExtractor(VALID_PAYLOAD), saved).execute(1)
    assert outcome.attempts == 1
    assert saved.saved == [outcome.extraction]


def test_a_rejected_answer_is_retried_and_the_attempts_are_counted() -> None:
    saved = FakeExtractionRepository()
    extractor = FakeExtractor(INVALID_PAYLOAD, VALID_PAYLOAD)
    outcome = build_use_case(extractor, saved).execute(1)
    assert outcome.attempts == 2
    assert saved.saved == [outcome.extraction]


def test_the_retry_shows_the_agent_its_rejected_answer_and_the_issues() -> None:
    extractor = FakeExtractor(INVALID_PAYLOAD, VALID_PAYLOAD)
    build_use_case(extractor, FakeExtractionRepository()).execute(1)
    assert "Previous attempt" not in extractor.prompts[0]
    assert "a sentence the paper never wrote" in extractor.prompts[1]
    assert "paper_concepts[0].evidence[0].quote" in extractor.prompts[1]


def test_the_last_rejection_is_raised_and_nothing_is_saved() -> None:
    saved = FakeExtractionRepository()
    extractor = FakeExtractor(INVALID_PAYLOAD, INVALID_PAYLOAD)
    with pytest.raises(ExtractionValidationError, match="is not found in the text of page 1"):
        build_use_case(extractor, saved, max_attempts=2).execute(1)
    assert saved.saved == []


def test_the_agent_is_asked_only_as_often_as_the_attempts_allow() -> None:
    extractor = FakeExtractor(INVALID_PAYLOAD, INVALID_PAYLOAD, VALID_PAYLOAD)
    with pytest.raises(ExtractionValidationError):
        build_use_case(extractor, FakeExtractionRepository(), max_attempts=2).execute(1)
    assert len(extractor.prompts) == 2


def test_an_answer_that_is_not_an_extraction_at_all_is_retried() -> None:
    saved = FakeExtractionRepository()
    outcome = build_use_case(FakeExtractor("not an object", VALID_PAYLOAD), saved).execute(1)
    assert outcome.attempts == 2


def test_the_extractor_error_is_propagated() -> None:
    with pytest.raises(ExtractorError, match="was not found"):
        build_use_case(FailingExtractor(), FakeExtractionRepository()).execute(1)


def test_an_unknown_paper_is_reported_before_the_agent_is_asked() -> None:
    extractor = FakeExtractor(VALID_PAYLOAD)
    use_case = ExtractPaperUseCase(FakePaperRepository({}), extractor, FakeExtractionRepository())
    with pytest.raises(PaperNotFoundError, match="paper 1"):
        use_case.execute(1)
    assert extractor.prompts == []
