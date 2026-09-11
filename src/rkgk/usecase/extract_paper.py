"""Use case that extracts one paper with an agent and stores the result."""

from rkgk.domain.agents import StructuredOutputAgent
from rkgk.domain.models.paper_extraction import (
    PaperExtractionIssue,
    PaperExtractionRun,
    PaperExtractionValidationError,
    build_paper_extraction_prompt,
    build_paper_extraction_schema,
)
from rkgk.domain.repositories.paper import PaperRepository
from rkgk.domain.repositories.paper_extraction import PaperExtractionRepository
from rkgk.usecase.validate_paper_extraction import ValidatePaperExtractionUseCase


class ExtractPaperUseCase:
    def __init__(
        self,
        paper_repository: PaperRepository,
        agent: StructuredOutputAgent,
        extraction_repository: PaperExtractionRepository,
        max_attempts: int = 3,
    ) -> None:
        self._paper_repository = paper_repository
        self._agent = agent
        self._extraction_repository = extraction_repository
        self._validate = ValidatePaperExtractionUseCase(paper_repository)
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        self._max_attempts = max_attempts

    def execute(self, paper_id: int) -> PaperExtractionRun:
        """Ask the agent until its answer survives validation, then store it."""
        paper = self._paper_repository.find(paper_id)
        schema = build_paper_extraction_schema()
        attempts = 0
        previous: object | None = None
        issues: tuple[PaperExtractionIssue, ...] = ()
        while True:
            attempts += 1
            payload = self._agent.answer(build_paper_extraction_prompt(paper, previous, issues), schema)
            try:
                result = self._validate.execute(paper_id, payload)
            except PaperExtractionValidationError as error:
                # The agent sees its own answer and what was wrong with it, so a retry corrects rather than reruns.
                if attempts >= self._max_attempts:
                    raise
                previous, issues = payload, error.issues
                continue
            self._extraction_repository.save(result)
            return PaperExtractionRun(extraction=result, attempts=attempts)
