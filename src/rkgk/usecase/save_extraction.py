"""Use case that stores an extraction payload once it has been validated."""

from rkgk.domain.models.extraction import ExtractionResult
from rkgk.domain.repositories.extraction import ExtractionRepository
from rkgk.domain.repositories.paper import PaperRepository
from rkgk.usecase.validate_extraction import ValidateExtractionUseCase


class SaveExtractionUseCase:
    def __init__(self, paper_repository: PaperRepository, extraction_repository: ExtractionRepository) -> None:
        self._validate = ValidateExtractionUseCase(paper_repository)
        self._extraction_repository = extraction_repository

    def execute(self, paper_id: int, payload: object) -> ExtractionResult:
        """Validation runs first so that a rejected payload never reaches the disk."""
        result = self._validate.execute(paper_id, payload)
        self._extraction_repository.save(result)
        return result
