"""Use case that checks an extraction payload against the schema and against the paper it describes."""

from pydantic import ValidationError

from rkgk.domain.extraction import (
    ExtractionIssue,
    ExtractionResult,
    ExtractionValidationError,
    check_extraction_against_paper,
)
from rkgk.domain.repositories import PaperRepository


def _format_path(location: tuple[int | str, ...]) -> str:
    """Render a pydantic error location the way the agent reads its own JSON: `paper_concepts[2].evidence[0].quote`."""
    parts: list[str] = []
    for item in location:
        if isinstance(item, int):
            parts.append(f"[{item}]")
        else:
            parts.append(f".{item}" if parts else item)
    return "".join(parts) or "<root>"


def _build_issues(error: ValidationError) -> tuple[ExtractionIssue, ...]:
    return tuple(ExtractionIssue(path=_format_path(item["loc"]), message=item["msg"]) for item in error.errors())


class ValidateExtractionUseCase:
    def __init__(self, paper_repository: PaperRepository) -> None:
        self._paper_repository = paper_repository

    def execute(self, paper_id: int, payload: object) -> ExtractionResult:
        try:
            result = ExtractionResult.model_validate(payload)
        except ValidationError as error:
            raise ExtractionValidationError(_build_issues(error)) from error
        issues = check_extraction_against_paper(result, self._paper_repository.find(paper_id))
        if issues:
            raise ExtractionValidationError(issues)
        return result
