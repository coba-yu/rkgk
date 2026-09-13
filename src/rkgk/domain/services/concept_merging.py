"""Joins what the merge stage answered per group back into one vocabulary.

The grouping stage bundles the concepts that may be the same and the merge stage answers for one bundle at a
time, so no call ever sees the whole corpus and no call can tell that two groups picked the same slug for two
different concepts. That is what this module finds: a slug two groups claim means the grouping split concepts
that belong together, and the remedy is to join those groups and ask the merge stage again about the whole.
The concepts no group holds never reach an agent at all; they are normalized here, because a concept nothing
resembles has nothing to decide.
It reads the normalization and extraction models and owns no data.
"""

from collections.abc import Mapping, Sequence

from rkgk.domain.models.concept_normalization import (
    ConceptGrouping,
    ConceptMerge,
    ConceptNormalizationIssue,
    GroupedConcept,
    LocalConceptRef,
    NormalizedConcept,
)
from rkgk.domain.models.paper_extraction import PaperExtraction
from rkgk.domain.services.slugs import derive_slug

# One bundle of concepts the merge stage is asked about, in the order the grouping listed them.
ConceptGroup = tuple[GroupedConcept, ...]


def collect_groups(grouping: ConceptGrouping, extractions: Sequence[PaperExtraction]) -> tuple[ConceptGroup, ...]:
    """Look every reference of the grouping up in the extractions, so the merge stage sees whole concepts.

    A reference to a concept no paper extracted is `check_grouping_against_extractions`'s to report, so the
    grouping is expected to have passed that check before it gets here.
    """
    concepts_by_ref = _map_concepts_by_ref(extractions)
    return tuple(tuple(concepts_by_ref[(ref.paper_id, ref.local_id)] for ref in group) for group in grouping.groups)


def collect_solo_concepts(
    grouping: ConceptGrouping, extractions: Sequence[PaperExtraction]
) -> tuple[GroupedConcept, ...]:
    """Return the extracted concepts the grouping left out, in the order the extractions declare them."""
    grouped = {(ref.paper_id, ref.local_id) for group in grouping.groups for ref in group}
    return tuple(item for ref, item in _map_concepts_by_ref(extractions).items() if ref not in grouped)


def derive_solo_concept(item: GroupedConcept) -> NormalizedConcept:
    """Normalize a concept nothing was grouped with, keeping everything the extraction said about it.

    No agent is asked: with no other concept to compare it against there is nothing to merge, and the name the
    paper used is already the canonical one.
    """
    concept = item.concept
    # A name written outside `[a-z0-9]` leaves no slug, so the concept is addressed by where it was extracted.
    slug = derive_slug(concept.name) or f"concept-{item.paper_id}-{concept.local_id}"
    return NormalizedConcept(
        id=slug,
        canonical_name=concept.name,
        type=concept.type,
        aliases=concept.aliases,
        description=concept.description,
        merged_from=(_build_ref(item),),
    )


def find_slug_collisions(
    concepts_by_group: Sequence[Sequence[NormalizedConcept]],
) -> dict[str, tuple[int, ...]]:
    """Return every slug more than one group declared, with the groups that declared it, first seen first.

    Only a slug across groups is reported: two concepts of one group sharing a slug is already refused by
    `ConceptMerge`, and reporting it again would ask for a join of a group with itself.
    """
    groups_by_slug: dict[str, list[int]] = {}
    for index, concepts in enumerate(concepts_by_group):
        for concept in concepts:
            indexes = groups_by_slug.setdefault(concept.id, [])
            if index not in indexes:
                indexes.append(index)
    return {slug: tuple(indexes) for slug, indexes in groups_by_slug.items() if len(indexes) > 1}


def combine_colliding_groups(
    groups: Sequence[ConceptGroup], concepts_by_group: Sequence[Sequence[NormalizedConcept]]
) -> tuple[tuple[ConceptGroup, ...], tuple[tuple[NormalizedConcept, ...], ...]]:
    """Join the groups that claimed one slug into a single group each, and say what is still answered for.

    Groups no collision touched keep the concepts they produced; a joined group comes back with none, which is
    how the caller sees which groups the merge stage has to be asked about again.
    Two slugs that collide across overlapping sets of groups join into one group, because a group cannot be in
    two places at once.
    """
    collisions = find_slug_collisions(concepts_by_group)
    joined = _collect_collision_components(collisions)
    touched = {index for component in joined for index in component}
    kept_groups = [group for index, group in enumerate(groups) if index not in touched]
    kept_concepts = [tuple(concepts) for index, concepts in enumerate(concepts_by_group) if index not in touched]
    combined = [combine_groups([groups[index] for index in component]) for component in joined]
    return tuple(kept_groups) + tuple(combined), tuple(kept_concepts) + tuple(() for _ in combined)


def combine_groups(groups: Sequence[ConceptGroup]) -> ConceptGroup:
    """Lay the given groups end to end as one group, in the order they were given."""
    return tuple(item for group in groups for item in group)


def build_merge_from_groups(concepts_by_group: Sequence[Sequence[NormalizedConcept]]) -> ConceptMerge:
    """Lay the concepts of every group end to end as the merge of the whole corpus."""
    return ConceptMerge(concepts=tuple(concept for concepts in concepts_by_group for concept in concepts))


def build_collision_issues(
    collisions: Mapping[str, Sequence[int]], groups: Sequence[ConceptGroup]
) -> tuple[ConceptNormalizationIssue, ...]:
    """Name each repeated slug together with the concepts of the groups that claimed it.

    The groups themselves are spelled out rather than their indexes, because an index of a run that already
    joined groups says nothing to a reader of the output.
    """
    return tuple(
        ConceptNormalizationIssue(
            path="concepts",
            message=(
                f"{slug!r} is the id of a concept in {len(indexes)} groups that were merged apart: "
                + "; ".join(_describe_group(groups[index]) for index in indexes)
            ),
        )
        for slug, indexes in collisions.items()
    )


def _map_concepts_by_ref(extractions: Sequence[PaperExtraction]) -> dict[tuple[int, str], GroupedConcept]:
    return {
        (extraction.paper_id, concept.local_id): GroupedConcept(paper_id=extraction.paper_id, concept=concept)
        for extraction in extractions
        for concept in extraction.concepts
    }


def _build_ref(item: GroupedConcept) -> LocalConceptRef:
    return LocalConceptRef(paper_id=item.paper_id, local_id=item.concept.local_id)


def _describe_group(group: ConceptGroup) -> str:
    return ", ".join(f"paper {item.paper_id} {item.concept.local_id!r}" for item in group)


def _collect_collision_components(collisions: Mapping[str, Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Group the colliding groups into the sets that have to become one group, lowest group first.

    Sets that share a group are folded together, so a slug shared by groups 0 and 1 and another shared by 1
    and 2 lead to one group of three rather than to two groups that both hold 1.
    """
    components: list[set[int]] = []
    for indexes in collisions.values():
        overlapping = [component for component in components if component.intersection(indexes)]
        joined = set(indexes)
        for component in overlapping:
            joined |= component
            components.remove(component)
        components.append(joined)
    return tuple(tuple(sorted(component)) for component in sorted(components, key=min))
