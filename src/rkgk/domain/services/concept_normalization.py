"""Checks that a normalization really covers the extractions it claims to merge.

It also rewrites the relations the extractions state in the slugs of the merge, which is what the relation
stage is shown and judged against so that general knowledge does not repeat a relation a paper already states.
This module reads the normalization and extraction models but owns no data of its own.
"""

from rkgk.domain.models.concept_normalization import (
    ConceptMerge,
    ConceptNormalization,
    ConceptNormalizationIssue,
    GeneralKnowledgeRelationProposal,
    NormalizedConcept,
    PaperStatedRelation,
)
from rkgk.domain.models.paper_extraction import PaperExtraction
from rkgk.domain.models.vocabulary import ConceptRelationType, ConceptType


def _check_concepts_against_extractions(
    concepts: tuple[NormalizedConcept, ...], extractions: tuple[PaperExtraction, ...]
) -> tuple[ConceptNormalizationIssue, ...]:
    """Report every place where the merged concepts disagree with the extractions they merge.

    All issues are collected instead of raising on the first one, because an agent fixing its output needs the
    whole list to converge in one more attempt.
    """
    types_by_ref = {
        (extraction.paper_id, concept.local_id): concept.type
        for extraction in extractions
        for concept in extraction.concepts
    }
    known_papers = {extraction.paper_id for extraction in extractions}
    issues: list[ConceptNormalizationIssue] = []
    covered: set[tuple[int, str]] = set()
    for index, concept in enumerate(concepts):
        source_types: list[ConceptType] = []
        for position, ref in enumerate(concept.merged_from):
            key = (ref.paper_id, ref.local_id)
            source_type = types_by_ref.get(key)
            if source_type is None:
                problem = (
                    f"paper {ref.paper_id} has no concept {ref.local_id!r}"
                    if ref.paper_id in known_papers
                    else f"there is no extraction for paper {ref.paper_id}"
                )
                issues.append(
                    ConceptNormalizationIssue(path=f"concepts[{index}].merged_from[{position}]", message=problem)
                )
                continue
            covered.add(key)
            source_types.append(source_type)
        if source_types and concept.type not in source_types:
            issues.append(
                ConceptNormalizationIssue(
                    path=f"concepts[{index}].type",
                    message=(
                        f"is {concept.type.value!r}, which none of the merged concepts has; they are "
                        f"{', '.join(sorted({source.value for source in source_types}))}"
                    ),
                )
            )
    for extraction in extractions:
        for concept in extraction.concepts:
            if (extraction.paper_id, concept.local_id) not in covered:
                issues.append(
                    ConceptNormalizationIssue(
                        path="concepts",
                        message=f"paper {extraction.paper_id} {concept.local_id!r} is in no merged_from",
                    )
                )
    return tuple(issues)


def check_merge_against_extractions(
    merge: ConceptMerge, extractions: tuple[PaperExtraction, ...]
) -> tuple[ConceptNormalizationIssue, ...]:
    """Report every disagreement between what the merge stage answered and the extractions it was given."""
    return _check_concepts_against_extractions(merge.concepts, extractions)


def check_normalization_against_extractions(
    normalization: ConceptNormalization, extractions: tuple[PaperExtraction, ...]
) -> tuple[ConceptNormalizationIssue, ...]:
    """Report every disagreement between a stored normalization and the extractions it merges."""
    return _check_concepts_against_extractions(normalization.concepts, extractions)


def collect_paper_stated_relations(
    concepts: tuple[NormalizedConcept, ...], extractions: tuple[PaperExtraction, ...]
) -> tuple[PaperStatedRelation, ...]:
    """Rewrite every relation the extractions state in the slugs the merge gave the two concepts it joins.

    One entry per pair and relation, listing the papers that state it, so a relation two papers agree on is one
    thing to keep the general knowledge stage away from rather than two.
    Entries come out in the order the relations were first seen, so the prompt built from them does not change
    between two runs over the same input.
    """
    slug_of = {(ref.paper_id, ref.local_id): concept.id for concept in concepts for ref in concept.merged_from}
    paper_ids_of: dict[tuple[str, str, ConceptRelationType], list[int]] = {}
    for extraction in extractions:
        for edge in extraction.concept_relations:
            source_id = slug_of.get((extraction.paper_id, edge.source_id))
            target_id = slug_of.get((extraction.paper_id, edge.target_id))
            if source_id is None or target_id is None:
                # A local concept no merged concept covers is the merge's own error, which
                # `check_merge_against_extractions` reports with its path; failing here again would only stop
                # the caller from collecting the rest.
                continue
            if source_id == target_id:
                # The merge put both ends in one concept, so the relation now says a concept relates to itself
                # and `build_knowledge_graph` drops it; listing it would forbid a proposal the graph never
                # carries.
                continue
            paper_ids = paper_ids_of.setdefault((source_id, target_id, edge.relation), [])
            if extraction.paper_id not in paper_ids:
                paper_ids.append(extraction.paper_id)
    return tuple(
        PaperStatedRelation(
            source_id=source_id, target_id=target_id, relation=relation, paper_ids=tuple(sorted(paper_ids))
        )
        for (source_id, target_id, relation), paper_ids in paper_ids_of.items()
    )


def _describe_stating_papers(paper_ids: tuple[int, ...]) -> str:
    """Name the papers behind a stated relation, as the subject of the sentence the issue is written in."""
    listed = ", ".join(str(paper_id) for paper_id in paper_ids)
    return f"paper {listed} states" if len(paper_ids) == 1 else f"papers {listed} state"


def check_relations_against_concepts(
    proposal: GeneralKnowledgeRelationProposal,
    concepts: tuple[NormalizedConcept, ...],
    paper_relations: tuple[PaperStatedRelation, ...],
) -> tuple[ConceptNormalizationIssue, ...]:
    """Report every proposed relation that stands on a slug the merge never declared or that a paper states.

    The relation stage answers on its own, so pydantic sees no vocabulary to check the slugs against; the
    vocabulary only exists here, where both the proposal and the merged concepts are in hand.
    A relation a paper states is rejected rather than dropped, because the agent is told not to propose one and
    an answer that does may have taken it for a general-knowledge relation of its own.
    """
    declared = {concept.id for concept in concepts}
    stating_papers = {
        (relation.source_id, relation.target_id, relation.relation): relation.paper_ids for relation in paper_relations
    }
    issues: list[ConceptNormalizationIssue] = []
    for index, edge in enumerate(proposal.concept_relations):
        for field, slug in (("source_id", edge.source_id), ("target_id", edge.target_id)):
            if slug not in declared:
                issues.append(
                    ConceptNormalizationIssue(
                        path=f"concept_relations[{index}].{field}",
                        message=f"is {slug!r}, which is not one of the normalized concepts",
                    )
                )
        paper_ids = stating_papers.get((edge.source_id, edge.target_id, edge.relation))
        if paper_ids is not None:
            issues.append(
                ConceptNormalizationIssue(
                    path=f"concept_relations[{index}]",
                    message=(
                        f"repeats what {_describe_stating_papers(paper_ids)}: "
                        f"{edge.source_id} {edge.relation.value} {edge.target_id}"
                    ),
                )
            )
    return tuple(issues)
