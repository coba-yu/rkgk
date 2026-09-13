"""Use case that merges the concepts of every extracted paper with an agent and stores the result.

The merge is reached in four steps rather than in one answer, because a corpus of a few dozen papers holds more
concepts than an agent can hold in view at once and one call over all of them never finishes:
embeddings pair the concepts that sit close together, one call bundles the pairs into groups of candidates, one
call per group merges that group alone, and Python joins the groups back together.
The relations between the merged concepts are then asked for in one more call, so a rejection can name the
stage that has to be repeated instead of restarting the whole run.
"""

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor

from pydantic import ValidationError

from rkgk.domain.agents import StructuredOutputAgent
from rkgk.domain.embedders import Embedder
from rkgk.domain.models.concept_normalization import (
    CandidatePair,
    ConceptGrouping,
    ConceptMerge,
    ConceptNormalization,
    ConceptNormalizationIssue,
    ConceptNormalizationRun,
    ConceptNormalizationValidationError,
    GeneralKnowledgeRelationProposal,
    GroupedConcept,
    MissingPaperExtractionsError,
    NormalizedConcept,
    PaperStatedRelation,
    build_concept_grouping_schema,
    build_concept_merge_schema,
    build_general_knowledge_relation_proposal_schema,
)
from rkgk.domain.models.paper_extraction import PaperExtraction
from rkgk.domain.prompts.concept_grouping.builder import build_concept_grouping_prompt
from rkgk.domain.prompts.concept_merge.builder import build_concept_merge_prompt
from rkgk.domain.prompts.general_knowledge_relations.builder import build_general_knowledge_relations_prompt
from rkgk.domain.repositories.concept_normalization import ConceptNormalizationRepository
from rkgk.domain.repositories.paper import PaperRepository
from rkgk.domain.repositories.paper_extraction import PaperExtractionNotFoundError, PaperExtractionRepository
from rkgk.domain.services.concept_candidates import DEFAULT_NEIGHBORS, collect_candidate_pairs
from rkgk.domain.services.concept_merging import (
    ConceptGroup,
    build_collision_issues,
    build_merge_from_groups,
    collect_groups,
    collect_solo_concepts,
    combine_colliding_groups,
    derive_solo_concept,
    find_slug_collisions,
)
from rkgk.domain.services.concept_normalization import (
    check_group_merge,
    check_grouping_against_extractions,
    check_merge_against_extractions,
    check_relations_against_concepts,
    collect_paper_stated_relations,
)
from rkgk.domain.services.validation import format_error_path

DEFAULT_CONCURRENCY = 4


def _build_issues(error: ValidationError) -> tuple[ConceptNormalizationIssue, ...]:
    return tuple(
        ConceptNormalizationIssue(path=format_error_path(item["loc"]), message=item["msg"]) for item in error.errors()
    )


def _prefix_issues(index: int, issues: tuple[ConceptNormalizationIssue, ...]) -> tuple[ConceptNormalizationIssue, ...]:
    """Say which group an issue came from, since every group answers under the same paths of its own merge."""
    return tuple(
        ConceptNormalizationIssue(path=f"groups[{index}].{issue.path}", message=issue.message) for issue in issues
    )


class NormalizeConceptsUseCase:
    def __init__(
        self,
        paper_repository: PaperRepository,
        extraction_repository: PaperExtractionRepository,
        agent: StructuredOutputAgent,
        embedder: Embedder,
        normalization_repository: ConceptNormalizationRepository,
        max_attempts: int = 3,
        neighbors: int = DEFAULT_NEIGHBORS,
        concurrency: int = DEFAULT_CONCURRENCY,
    ) -> None:
        self._paper_repository = paper_repository
        self._extraction_repository = extraction_repository
        self._agent = agent
        self._embedder = embedder
        self._normalization_repository = normalization_repository
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        if neighbors < 1:
            raise ValueError(f"neighbors must be at least 1, got {neighbors}")
        if concurrency < 1:
            raise ValueError(f"concurrency must be at least 1, got {concurrency}")
        self._max_attempts = max_attempts
        self._neighbors = neighbors
        self._concurrency = concurrency

    def execute(self) -> ConceptNormalizationRun:
        """Group, merge and relate the concepts of every extracted paper, and store the result as one artifact.

        Nothing is saved until every stage has been accepted, so a normalization never reaches the data
        directory with the concepts of one run and no relations at all.
        """
        paper_ids = tuple(entry.id for entry in self._paper_repository.find_index())
        if not paper_ids:
            raise ValueError("no papers in the index")
        extractions = self._collect_extractions(paper_ids)
        pairs = collect_candidate_pairs(extractions, self._embedder, self._neighbors)
        grouping, grouping_attempts = self._group_concepts(extractions, pairs)
        groups = collect_groups(grouping, extractions)
        solos = collect_solo_concepts(grouping, extractions)
        merge, merge_calls, merge_rounds = self._merge_groups(groups, solos, extractions)
        # Only the accepted merge maps the local ids onto slugs, so what the papers state can be written in the
        # vocabulary of the relation stage only once the merge is through.
        paper_relations = collect_paper_stated_relations(merge.concepts, extractions)
        proposal, relation_attempts = self._propose_relations(merge.concepts, paper_relations)
        normalization = ConceptNormalization(
            schema_version=1, concepts=merge.concepts, concept_relations=proposal.concept_relations
        )
        self._normalization_repository.save(normalization)
        return ConceptNormalizationRun(
            normalization=normalization,
            grouping_attempts=grouping_attempts,
            groups=len(groups),
            merge_calls=merge_calls,
            merge_rounds=merge_rounds,
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

    def _group_concepts(
        self, extractions: tuple[PaperExtraction, ...], pairs: tuple[CandidatePair, ...]
    ) -> tuple[ConceptGrouping, int]:
        """Ask for the groups of merge candidates until they name concepts that exist, counting the attempts."""
        schema = build_concept_grouping_schema()
        attempts = 0
        previous: object | None = None
        issues: tuple[ConceptNormalizationIssue, ...] = ()
        while True:
            attempts += 1
            payload = self._agent.answer(build_concept_grouping_prompt(extractions, pairs, previous, issues), schema)
            try:
                grouping = self._validate_grouping(payload, extractions)
            except ConceptNormalizationValidationError as error:
                # The agent sees its own answer and what was wrong with it, so a retry corrects rather than reruns.
                if attempts >= self._max_attempts:
                    raise
                previous, issues = payload, error.issues
                continue
            return grouping, attempts

    def _merge_groups(
        self,
        groups: tuple[ConceptGroup, ...],
        solos: tuple[GroupedConcept, ...],
        extractions: tuple[PaperExtraction, ...],
    ) -> tuple[ConceptMerge, int, int]:
        """Merge every group, join the answers, and ask again about the groups that picked the same slug.

        Each concept the grouping left out counts as a group of one that is already answered for, so a slug it
        shares with a merged concept is found and repaired like any other collision, and joining it with the
        group it collided with gives a group of at least two that the merge stage can be asked about.
        """
        pending = tuple(groups) + tuple((item,) for item in solos)
        answered = tuple(() for _ in groups) + tuple((derive_solo_concept(item),) for item in solos)
        calls = 0
        rounds = 0
        while True:
            rounds += 1
            merged, made = self._answer_unmerged_groups(pending, answered)
            calls += made
            answered = merged
            collisions = find_slug_collisions(answered)
            if not collisions:
                break
            if rounds >= self._max_attempts:
                raise ConceptNormalizationValidationError("merge", build_collision_issues(collisions, pending))
            pending, answered = combine_colliding_groups(pending, answered)
        return self._validate_merge(answered, extractions), calls, rounds

    def _answer_unmerged_groups(
        self, groups: Sequence[ConceptGroup], answered: Sequence[tuple[NormalizedConcept, ...]]
    ) -> tuple[tuple[tuple[NormalizedConcept, ...], ...], int]:
        """Ask the merge stage about every group that has no concepts yet, several groups at a time.

        The results are read back in group order rather than as they finish, so the run reports the first group
        that failed and not whichever thread lost the race.
        """
        futures: dict[int, Future[tuple[tuple[NormalizedConcept, ...], int]]] = {}
        with ThreadPoolExecutor(max_workers=self._concurrency) as executor:
            for index, group in enumerate(groups):
                if not answered[index]:
                    futures[index] = executor.submit(self._merge_one_group, group)
            results = list(answered)
            calls = 0
            failure: tuple[int, ConceptNormalizationValidationError] | None = None
            for index in sorted(futures):
                try:
                    concepts, attempts = futures[index].result()
                except ConceptNormalizationValidationError as error:
                    if failure is None:
                        failure = (index, error)
                    continue
                calls += attempts
                results[index] = concepts
        if failure is not None:
            raise ConceptNormalizationValidationError("merge", _prefix_issues(failure[0], failure[1].issues))
        return tuple(results), calls

    def _merge_one_group(self, group: ConceptGroup) -> tuple[tuple[NormalizedConcept, ...], int]:
        """Ask for the merge of one group until it covers that group, and say how many attempts it took."""
        schema = build_concept_merge_schema()
        attempts = 0
        previous: object | None = None
        issues: tuple[ConceptNormalizationIssue, ...] = ()
        while True:
            attempts += 1
            payload = self._agent.answer(build_concept_merge_prompt(group, previous, issues), schema)
            try:
                merge = self._validate_group_merge(payload, group)
            except ConceptNormalizationValidationError as error:
                if attempts >= self._max_attempts:
                    raise
                previous, issues = payload, error.issues
                continue
            return merge.concepts, attempts

    def _propose_relations(
        self, concepts: tuple[NormalizedConcept, ...], paper_relations: tuple[PaperStatedRelation, ...]
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
            payload = self._agent.answer(
                build_general_knowledge_relations_prompt(concepts, paper_relations, previous, issues), schema
            )
            try:
                proposal = self._validate_proposal(payload, concepts, paper_relations)
            except ConceptNormalizationValidationError as error:
                if attempts >= self._max_attempts:
                    raise
                previous, issues = payload, error.issues
                continue
            return proposal, attempts

    def _validate_grouping(self, payload: object, extractions: tuple[PaperExtraction, ...]) -> ConceptGrouping:
        try:
            grouping = ConceptGrouping.model_validate(payload)
        except ValidationError as error:
            raise ConceptNormalizationValidationError("grouping", _build_issues(error)) from error
        issues = check_grouping_against_extractions(grouping, extractions)
        if issues:
            raise ConceptNormalizationValidationError("grouping", issues)
        return grouping

    def _validate_group_merge(self, payload: object, group: ConceptGroup) -> ConceptMerge:
        try:
            merge = ConceptMerge.model_validate(payload)
        except ValidationError as error:
            raise ConceptNormalizationValidationError("merge", _build_issues(error)) from error
        issues = check_group_merge(merge, group)
        if issues:
            raise ConceptNormalizationValidationError("merge", issues)
        return merge

    def _validate_merge(
        self, answered: Sequence[tuple[NormalizedConcept, ...]], extractions: tuple[PaperExtraction, ...]
    ) -> ConceptMerge:
        """Check the joined merge against the extractions, which is the only place the whole corpus is in view."""
        try:
            merge = build_merge_from_groups(answered)
        except ValidationError as error:
            raise ConceptNormalizationValidationError("merge", _build_issues(error)) from error
        issues = check_merge_against_extractions(merge, extractions)
        if issues:
            raise ConceptNormalizationValidationError("merge", issues)
        return merge

    def _validate_proposal(
        self,
        payload: object,
        concepts: tuple[NormalizedConcept, ...],
        paper_relations: tuple[PaperStatedRelation, ...],
    ) -> GeneralKnowledgeRelationProposal:
        try:
            proposal = GeneralKnowledgeRelationProposal.model_validate(payload)
        except ValidationError as error:
            raise ConceptNormalizationValidationError("relations", _build_issues(error)) from error
        issues = check_relations_against_concepts(proposal, concepts, paper_relations)
        if issues:
            raise ConceptNormalizationValidationError("relations", issues)
        return proposal
