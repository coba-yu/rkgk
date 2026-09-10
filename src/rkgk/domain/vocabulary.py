"""Vocabulary of the knowledge graph.

Node types, relation names, their meanings, and whether search traversal follows them are defined here.
Extraction prompts and JSON Schemas are generated from these definitions, so this module is the single source of truth.
"""

import enum
from dataclasses import dataclass


@dataclass(frozen=True)
class ConceptTypeSpec:
    description: str
    traversable: bool


@dataclass(frozen=True)
class RelationSpec:
    description: str
    traversable: bool


@dataclass(frozen=True)
class ConceptRelationSpec:
    description: str
    direction: str
    traversable: bool


@dataclass(frozen=True)
class OriginSpec:
    description: str
    requires_evidence: bool


class ConceptType(enum.StrEnum):
    PROBLEM = "problem"
    METHOD = "method"
    KEYWORD = "keyword"

    @property
    def spec(self) -> ConceptTypeSpec:
        return _CONCEPT_TYPE_SPECS[self]


# Keyword is not traversable because generic terms link unrelated papers and would drown the graph in noise.
_CONCEPT_TYPE_SPECS: dict[ConceptType, ConceptTypeSpec] = {
    ConceptType.PROBLEM: ConceptTypeSpec("A task, limitation, or research question a paper addresses", True),
    ConceptType.METHOD: ConceptTypeSpec("A technique, model, algorithm, or component used or proposed", True),
    ConceptType.KEYWORD: ConceptTypeSpec(
        "A term that is neither a problem nor a method; kept for display and explanation only", False
    ),
}


class PaperConceptRelation(enum.StrEnum):
    PROPOSES = "proposes"
    USES = "uses"
    ADDRESSES = "addresses"
    MENTIONS = "mentions"

    @property
    def spec(self) -> RelationSpec:
        return _PAPER_CONCEPT_RELATION_SPECS[self]


# Mentions is not traversable because a passing reference is no evidence that the paper is about the concept.
_PAPER_CONCEPT_RELATION_SPECS: dict[PaperConceptRelation, RelationSpec] = {
    PaperConceptRelation.PROPOSES: RelationSpec("The paper introduces the concept as its own contribution", True),
    PaperConceptRelation.USES: RelationSpec("The paper applies the concept in its method or experiments", True),
    PaperConceptRelation.ADDRESSES: RelationSpec("The paper targets the concept as a problem it tries to solve", True),
    PaperConceptRelation.MENTIONS: RelationSpec(
        "The paper refers to the concept without using or proposing it", False
    ),
}


class ConceptRelationType(enum.StrEnum):
    IS_A = "is_a"
    PART_OF = "part_of"
    USED_FOR = "used_for"
    RELATED_TO = "related_to"

    @property
    def spec(self) -> ConceptRelationSpec:
        return _CONCEPT_RELATION_SPECS[self]


_CONCEPT_RELATION_SPECS: dict[ConceptRelationType, ConceptRelationSpec] = {
    ConceptRelationType.IS_A: ConceptRelationSpec("source is a kind of target", "source -> target", True),
    ConceptRelationType.PART_OF: ConceptRelationSpec("source is a component of target", "source -> target", True),
    ConceptRelationType.USED_FOR: ConceptRelationSpec(
        "source is used for the problem or purpose target", "source -> target", True
    ),
    ConceptRelationType.RELATED_TO: ConceptRelationSpec(
        "source and target are related in a way the other relations cannot express; use sparingly",
        "source -> target",
        True,
    ),
}


class Origin(enum.StrEnum):
    PAPER = "paper"
    GENERAL_KNOWLEDGE = "general_knowledge"

    @property
    def spec(self) -> OriginSpec:
        return _ORIGIN_SPECS[self]


# General knowledge carries no evidence on purpose: a quote would claim a paper backs a statement it never made.
_ORIGIN_SPECS: dict[Origin, OriginSpec] = {
    Origin.PAPER: OriginSpec("Extracted from the paper text with quoted evidence", True),
    Origin.GENERAL_KNOWLEDGE: OriginSpec(
        "Added from general knowledge during normalization; unverified against any paper", False
    ),
}


def traversable_concept_types() -> frozenset[ConceptType]:
    return frozenset(member for member in ConceptType if member.spec.traversable)


def traversable_paper_relations() -> frozenset[PaperConceptRelation]:
    return frozenset(member for member in PaperConceptRelation if member.spec.traversable)


def traversable_concept_relations() -> frozenset[ConceptRelationType]:
    return frozenset(member for member in ConceptRelationType if member.spec.traversable)


def _member_line(value: str, description: str, *notes: str) -> str:
    return f"- `{value}`: {description}" + "".join(f" ({note})" for note in notes)


def _traversal_notes(traversable: bool) -> tuple[str, ...]:
    return () if traversable else ("not used for traversal",)


def describe_vocabulary() -> str:
    """Render every enum as Markdown, one section per enum, in definition order.

    The text is embedded in agent prompts, so wording and order must stay stable; a snapshot test guards it.
    """
    lines = ["## ConceptType", ""]
    for concept_type in ConceptType:
        spec = concept_type.spec
        lines.append(_member_line(concept_type.value, spec.description, *_traversal_notes(spec.traversable)))

    lines += ["", "## PaperConceptRelation", ""]
    for paper_relation in PaperConceptRelation:
        relation_spec = paper_relation.spec
        lines.append(
            _member_line(paper_relation.value, relation_spec.description, *_traversal_notes(relation_spec.traversable))
        )

    lines += ["", "## ConceptRelationType", ""]
    for concept_relation in ConceptRelationType:
        edge_spec = concept_relation.spec
        notes = (f"direction: {edge_spec.direction}", *_traversal_notes(edge_spec.traversable))
        lines.append(_member_line(concept_relation.value, edge_spec.description, *notes))

    lines += ["", "## Origin", ""]
    for origin in Origin:
        origin_spec = origin.spec
        note = "requires evidence" if origin_spec.requires_evidence else "no evidence"
        lines.append(_member_line(origin.value, origin_spec.description, note))

    return "\n".join(lines) + "\n"
