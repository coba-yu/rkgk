from pathlib import Path

import pytest

from rkgk.domain.models.vocabulary import (
    ConceptRelationType,
    ConceptType,
    Origin,
    PaperConceptRelation,
    describe_vocabulary,
    traversable_concept_relations,
    traversable_concept_types,
    traversable_paper_relations,
)

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "vocabulary.md"


def test_traversable_concept_types_exclude_keyword() -> None:
    assert traversable_concept_types() == frozenset({ConceptType.PROBLEM, ConceptType.METHOD})


def test_traversable_paper_relations_exclude_mentions() -> None:
    assert traversable_paper_relations() == frozenset(
        {PaperConceptRelation.PROPOSES, PaperConceptRelation.USES, PaperConceptRelation.ADDRESSES}
    )


def test_all_concept_relations_are_traversable() -> None:
    assert traversable_concept_relations() == frozenset(ConceptRelationType)


@pytest.mark.parametrize("member", list(ConceptType) + list(PaperConceptRelation))
def test_concept_and_paper_relation_members_have_a_description(member: ConceptType | PaperConceptRelation) -> None:
    assert member.spec.description


@pytest.mark.parametrize("member", list(ConceptRelationType))
def test_concept_relation_members_have_a_description_and_direction(member: ConceptRelationType) -> None:
    assert member.spec.description
    assert member.spec.direction == "source -> target"


def test_paper_origin_requires_evidence_and_general_knowledge_does_not() -> None:
    assert Origin.PAPER.spec.requires_evidence is True
    assert Origin.GENERAL_KNOWLEDGE.spec.requires_evidence is False


def test_enum_values_are_plain_strings() -> None:
    assert ConceptType.PROBLEM == "problem"
    assert ConceptRelationType.IS_A == "is_a"


def test_describe_vocabulary_matches_snapshot() -> None:
    assert describe_vocabulary() == SNAPSHOT_PATH.read_text(encoding="utf-8")


def test_describe_vocabulary_is_deterministic() -> None:
    assert describe_vocabulary() == describe_vocabulary()
