import copy
from typing import Any

import pytest

from rkgk.domain.models.paper_extraction import ExtractionValidationError
from rkgk.domain.repositories.paper import PaperNotFoundError
from rkgk.usecase.validate_extraction import ValidateExtractionUseCase
from tests.usecase.fakes import FakePaperRepository, build_paper

PAPER = build_paper(1, "We study a retrieval-augmented generation pipeline.\n", "The pipeline embeds chunks.\n")

PAYLOAD: dict[str, Any] = {
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


def build_payload(**overrides: Any) -> dict[str, Any]:
    payload = copy.deepcopy(PAYLOAD)
    payload.update(overrides)
    return payload


def build_use_case() -> ValidateExtractionUseCase:
    return ValidateExtractionUseCase(FakePaperRepository({1: PAPER}))


def test_a_payload_backed_by_the_paper_is_returned_as_a_result() -> None:
    result = build_use_case().execute(1, build_payload())
    assert result.paper_id == 1
    assert result.concepts[0].name == "Retrieval-Augmented Generation"


def test_a_payload_that_is_not_an_object_is_rejected() -> None:
    with pytest.raises(ExtractionValidationError) as caught:
        build_use_case().execute(1, ["not an object"])
    assert caught.value.issues != ()


def test_a_missing_field_is_reported_by_its_name() -> None:
    payload = build_payload()
    del payload["summary_ja"]
    with pytest.raises(ExtractionValidationError) as caught:
        build_use_case().execute(1, payload)
    assert [issue.path for issue in caught.value.issues] == ["summary_ja"]


def test_a_field_inside_a_list_is_reported_with_its_index_path() -> None:
    payload = build_payload()
    payload["paper_concepts"][0]["evidence"][0]["quote"] = "   "
    with pytest.raises(ExtractionValidationError) as caught:
        build_use_case().execute(1, payload)
    assert "paper_concepts[0].evidence[0].quote" in [issue.path for issue in caught.value.issues]


def test_an_unknown_relation_is_reported_with_a_message() -> None:
    payload = build_payload()
    payload["paper_concepts"][0]["relation"] = "cites"
    with pytest.raises(ExtractionValidationError) as caught:
        build_use_case().execute(1, payload)
    assert caught.value.issues[0].path == "paper_concepts[0].relation"
    assert caught.value.issues[0].message


def test_a_quote_that_is_not_in_the_paper_is_reported() -> None:
    payload = build_payload()
    payload["paper_concepts"][0]["evidence"][0]["quote"] = "a sentence the paper never wrote"
    with pytest.raises(ExtractionValidationError, match="is not found in the text of page 1"):
        build_use_case().execute(1, payload)


def test_a_payload_for_another_paper_is_reported() -> None:
    with pytest.raises(ExtractionValidationError) as caught:
        build_use_case().execute(1, build_payload(paper_id=2))
    assert "paper_id" in [issue.path for issue in caught.value.issues]


def test_the_repository_error_for_an_unknown_paper_is_propagated() -> None:
    with pytest.raises(PaperNotFoundError, match="paper 3"):
        build_use_case().execute(3, build_payload(paper_id=3))


def test_the_error_message_lists_every_issue() -> None:
    payload = build_payload()
    payload["paper_concepts"][0]["evidence"][0]["page"] = 5
    with pytest.raises(ExtractionValidationError) as caught:
        build_use_case().execute(1, payload)
    assert "paper_concepts[0].evidence[0].page" in str(caught.value)
