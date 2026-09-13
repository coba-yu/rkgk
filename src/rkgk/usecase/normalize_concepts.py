"""Use case that merges the concepts of every extracted paper with an agent and stores the result.

The agent is asked twice: once for the merge, and once for the relations between the concepts that merge
settled on. Splitting the two keeps each answer small enough for an agent to hold the whole input in view, and
lets a rejection name the stage that has to be repeated.
"""

from pydantic import ValidationError

from rkgk.domain.agents import StructuredOutputAgent
from rkgk.domain.models.concept_normalization import (
    ConceptMerge,
    ConceptNormalization,
    ConceptNormalizationIssue,
    ConceptNormalizationRun,
    ConceptNormalizationValidationError,
    GeneralKnowledgeRelationProposal,
    MissingPaperExtractionsError,
    NormalizedConcept,
    build_concept_merge_schema,
    build_general_knowledge_relation_proposal_schema,
)
from rkgk.domain.models.paper_extraction import PaperExtraction
from rkgk.domain.prompts.concept_merge.builder import build_concept_merge_prompt
from rkgk.domain.prompts.general_knowledge_relations.builder import build_general_knowledge_relations_prompt
from rkgk.domain.repositories.concept_normalization import ConceptNormalizationRepository
from rkgk.domain.repositories.paper import PaperRepository
from rkgk.domain.repositories.paper_extraction import PaperExtractionNotFoundError, PaperExtractionRepository
from rkgk.domain.services.concept_normalization import (
    check_merge_against_extractions,
    check_relations_against_concepts,
)
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
        """Merge every extracted paper, relate the merged concepts, and store the two answers as one artifact.

        Nothing is saved until both stages have been accepted, so a normalization never reaches the data
        directory with the concepts of one run and no relations at all.
        """
        paper_ids = tuple(entry.id for entry in self._paper_repository.find_index())
        if not paper_ids:
            raise ValueError("no papers in the index")
        extractions = self._collect_extractions(paper_ids)
        merge, merge_attempts = self._merge_concepts(extractions)
        proposal, relation_attempts = self._propose_relations(merge.concepts)
        normalization = ConceptNormalization(
            schema_version=1, concepts=merge.concepts, concept_relations=proposal.concept_relations
        )
        self._normalization_repository.save(normalization)
        return ConceptNormalizationRun(
            normalization=normalization,
            merge_attempts=merge_attempts,
            relation_attempts=relation_attempts,
            paper_ids=paper_ids,
        )

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

    def _merge_concepts(self, extractions: tuple[PaperExtraction, ...]) -> tuple[ConceptMerge, int]:
        """Ask for the merged vocabulary until it matches the extractions, and say how many attempts it took."""
        schema = build_concept_merge_schema()
        attempts = 0
        previous: object | None = None
        issues: tuple[ConceptNormalizationIssue, ...] = ()
        while True:
            attempts += 1
            payload = self._agent.answer(build_concept_merge_prompt(extractions, previous, issues), schema)
            try:
                merge = self._validate_merge(payload, extractions)
            except ConceptNormalizationValidationError as error:
                # The agent sees its own answer and what was wrong with it, so a retry corrects rather than reruns.
                if attempts >= self._max_attempts:
                    raise
                previous, issues = payload, error.issues
                continue
            return merge, attempts

    def _propose_relations(
        self, concepts: tuple[NormalizedConcept, ...]
    ) -> tuple[GeneralKnowledgeRelationProposal, int]:
        """Ask for the relations between the merged concepts, counting this stage's attempts on their own.

        The merge is not asked for again when a relation is wrong, because the concepts it settled on are what
        the rejected relations were judged against.
        """
        schema = build_general_knowledge_relation_proposal_schema()
        attempts = 0
        previous: object | None = None
        issues: tuple[ConceptNormalizationIssue, ...] = ()
        while True:
            attempts += 1
            payload = self._agent.answer(build_general_knowledge_relations_prompt(concepts, previous, issues), schema)
            try:
                proposal = self._validate_proposal(payload, concepts)
            except ConceptNormalizationValidationError as error:
                if attempts >= self._max_attempts:
                    raise
                previous, issues = payload, error.issues
                continue
            return proposal, attempts

    def _validate_merge(self, payload: object, extractions: tuple[PaperExtraction, ...]) -> ConceptMerge:
        try:
            merge = ConceptMerge.model_validate(payload)
        except ValidationError as error:
            raise ConceptNormalizationValidationError("merge", _build_issues(error)) from error
        issues = check_merge_against_extractions(merge, extractions)
        if issues:
            raise ConceptNormalizationValidationError("merge", issues)
        return merge

    def _validate_proposal(
        self, payload: object, concepts: tuple[NormalizedConcept, ...]
    ) -> GeneralKnowledgeRelationProposal:
        try:
            proposal = GeneralKnowledgeRelationProposal.model_validate(payload)
        except ValidationError as error:
            raise ConceptNormalizationValidationError("relations", _build_issues(error)) from error
        issues = check_relations_against_concepts(proposal, concepts)
        if issues:
            raise ConceptNormalizationValidationError("relations", issues)
        return proposal
