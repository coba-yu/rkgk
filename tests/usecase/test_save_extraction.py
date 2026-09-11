import pytest

from rkgk.domain.models.paper_extraction import PaperExtractionValidationError
from rkgk.domain.repositories.paper import PaperNotFoundError
from rkgk.usecase.save_extraction import SaveExtractionUseCase
from tests.usecase.fakes import FakePaperExtractionRepository, FakePaperRepository
from tests.usecase.test_validate_paper_extraction import PAPER, build_payload


def build_use_case(extraction_repository: FakePaperExtractionRepository) -> SaveExtractionUseCase:
    return SaveExtractionUseCase(FakePaperRepository({1: PAPER}), extraction_repository)


def test_a_valid_payload_is_stored_and_returned() -> None:
    extraction_repository = FakePaperExtractionRepository()
    result = build_use_case(extraction_repository).execute(1, build_payload())
    assert extraction_repository.saved == [result]


def test_an_invalid_payload_is_not_stored() -> None:
    extraction_repository = FakePaperExtractionRepository()
    payload = build_payload()
    payload["paper_concepts"][0]["evidence"][0]["quote"] = "a sentence the paper never wrote"
    with pytest.raises(PaperExtractionValidationError):
        build_use_case(extraction_repository).execute(1, payload)
    assert extraction_repository.saved == []


def test_a_payload_for_a_paper_that_does_not_exist_is_not_stored() -> None:
    extraction_repository = FakePaperExtractionRepository()
    use_case = SaveExtractionUseCase(FakePaperRepository({}), extraction_repository)
    with pytest.raises(PaperNotFoundError, match="paper 1"):
        use_case.execute(1, build_payload())
    assert extraction_repository.saved == []


def test_the_stored_result_can_be_read_back_from_the_repository() -> None:
    extraction_repository = FakePaperExtractionRepository()
    build_use_case(extraction_repository).execute(1, build_payload())
    assert extraction_repository.find(1).paper_id == 1
