"""Use case that merges the concepts of every extracted paper with an agent and stores the result."""

from pydantic import ValidationError

from rkgk.domain.agents import StructuredOutputAgent
from rkgk.domain.models.concept_normalization import (
    ConceptNormalization,
    ConceptNormalizationIssue,
    ConceptNormalizationRun,
    ConceptNormalizationValidationError,
    MissingPaperExtractionsError,
    build_concept_normalization_schema,
)
from rkgk.domain.models.paper_extraction import PaperExtraction
from rkgk.domain.prompts.concept_normalization import build_concept_normalization_prompt
from rkgk.domain.repositories.concept_normalization import ConceptNormalizationRepository
from rkgk.domain.repositories.paper import PaperRepository
from rkgk.domain.repositories.paper_extraction import PaperExtractionNotFoundError, PaperExtractionRepository
from rkgk.domain.services.concept_normalization import check_normalization_against_extractions
from rkgk.domain.services.validation import format_error_path


def _build_issues(error: ValidationError) -> tuple[ConceptNormalizationIssue, ...]:
    return tuple(
        ConceptNormalizationIssue(path=format_error_path(item["loc"]), message=item["msg"]) for item in error.errors()
    )


class NormalizeConceptsUseCase:
    def __init__(
        self,
        paper_repository: PaperRepository,
        extraction_repository: PaperExtractionRepository,
        agent: StructuredOutputAgent,
        normalization_repository: ConceptNormalizationRepository,
        max_attempts: int = 3,
    ) -> None:
        self._paper_repository = paper_repository
        self._extraction_repository = extraction_repository
        self._agent = agent
        self._normalization_repository = normalization_repository
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        self._max_attempts = max_attempts

    def execute(self) -> ConceptNormalizationRun:
        """Normalize every extracted paper at once, asking the agent until its answer survives validation."""
        paper_ids = tuple(entry.id for entry in self._paper_repository.find_index())
        if not paper_ids:
            raise ValueError("no papers in the index")
        extractions = self._collect_extractions(paper_ids)
        schema = build_concept_normalization_schema()
        attempts = 0
        previous: object | None = None
        issues: tuple[ConceptNormalizationIssue, ...] = ()
        while True:
            attempts += 1
            payload = self._agent.answer(build_concept_normalization_prompt(extractions, previous, issues), schema)
            try:
                normalization = self._validate(payload, extractions)
            except ConceptNormalizationValidationError as error:
                # The agent sees its own answer and what was wrong with it, so a retry corrects rather than reruns.
                if attempts >= self._max_attempts:
                    raise
                previous, issues = payload, error.issues
                continue
            self._normalization_repository.save(normalization)
            return ConceptNormalizationRun(normalization=normalization, attempts=attempts, paper_ids=paper_ids)

    def _collect_extractions(self, paper_ids: tuple[int, ...]) -> tuple[PaperExtraction, ...]:
        """Read every extraction, reporting all papers that still need one instead of only the first."""
        extractions: list[PaperExtraction] = []
        missing: list[int] = []
        for paper_id in paper_ids:
            try:
                extractions.append(self._extraction_repository.find(paper_id))
            except PaperExtractionNotFoundError:
                missing.append(paper_id)
        if missing:
            raise MissingPaperExtractionsError(tuple(missing))
        return tuple(extractions)

    def _validate(self, payload: object, extractions: tuple[PaperExtraction, ...]) -> ConceptNormalization:
        try:
            normalization = ConceptNormalization.model_validate(payload)
        except ValidationError as error:
            raise ConceptNormalizationValidationError(_build_issues(error)) from error
        issues = check_normalization_against_extractions(normalization, extractions)
        if issues:
            raise ConceptNormalizationValidationError(issues)
        return normalization
