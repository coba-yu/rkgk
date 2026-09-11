"""Use case that checks an extraction payload against the schema and against the paper it describes."""

from pydantic import ValidationError

from rkgk.domain.models.base import format_error_path
from rkgk.domain.models.paper_extraction import (
    PaperExtraction,
    PaperExtractionIssue,
    PaperExtractionValidationError,
    check_extraction_against_paper,
)
from rkgk.domain.repositories.paper import PaperRepository


def _build_issues(error: ValidationError) -> tuple[PaperExtractionIssue, ...]:
    return tuple(
        PaperExtractionIssue(path=format_error_path(item["loc"]), message=item["msg"]) for item in error.errors()
    )


class ValidatePaperExtractionUseCase:
    def __init__(self, paper_repository: PaperRepository) -> None:
        self._paper_repository = paper_repository

    def execute(self, paper_id: int, payload: object) -> PaperExtraction:
        try:
            result = PaperExtraction.model_validate(payload)
        except ValidationError as error:
            raise PaperExtractionValidationError(_build_issues(error)) from error
        issues = check_extraction_against_paper(result, self._paper_repository.find(paper_id))
        if issues:
            raise PaperExtractionValidationError(issues)
        return result
