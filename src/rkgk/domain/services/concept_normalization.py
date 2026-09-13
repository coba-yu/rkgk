"""Checks that a normalization really covers the extractions it claims to merge.

This module reads the normalization and extraction models but owns no data of its own.
"""

from rkgk.domain.models.concept_normalization import (
    ConceptMerge,
    ConceptNormalization,
    ConceptNormalizationIssue,
    GeneralKnowledgeRelationProposal,
    NormalizedConcept,
)
from rkgk.domain.models.paper_extraction import PaperExtraction
from rkgk.domain.models.vocabulary import ConceptType


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


def check_relations_against_concepts(
    proposal: GeneralKnowledgeRelationProposal, concepts: tuple[NormalizedConcept, ...]
) -> tuple[ConceptNormalizationIssue, ...]:
    """Report every proposed relation that stands on a slug the merge never declared.

    The relation stage answers on its own, so pydantic sees no vocabulary to check the slugs against; the
    vocabulary only exists here, where both the proposal and the merged concepts are in hand.
    """
    declared = {concept.id for concept in concepts}
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
    return tuple(issues)
