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
    ConceptType.PROBLEM: ConceptTypeSpec("論文が取り組む課題、制約、または研究上の問い", True),
    ConceptType.METHOD: ConceptTypeSpec("使用または提案される技術、モデル、アルゴリズム、または構成要素", True),
    ConceptType.KEYWORD: ConceptTypeSpec("problem でも method でもない用語。表示と説明のためだけに保持する", False),
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
    PaperConceptRelation.PROPOSES: RelationSpec("論文がその概念を自身の貢献として導入する", True),
    PaperConceptRelation.USES: RelationSpec("論文がその概念を手法または実験で適用する", True),
    PaperConceptRelation.ADDRESSES: RelationSpec("論文がその概念を、解こうとしている課題として扱う", True),
    PaperConceptRelation.MENTIONS: RelationSpec("論文がその概念に言及するだけで、使用も提案もしていない", False),
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
    ConceptRelationType.IS_A: ConceptRelationSpec("source は target の一種", "source -> target", True),
    ConceptRelationType.PART_OF: ConceptRelationSpec("source は target の構成要素", "source -> target", True),
    ConceptRelationType.USED_FOR: ConceptRelationSpec(
        "source は課題または目的である target のために使われる", "source -> target", True
    ),
    ConceptRelationType.RELATED_TO: ConceptRelationSpec(
        "source と target が、他の関係では表現できない形で関連する。多用しない",
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
    Origin.PAPER: OriginSpec("論文本文から、引用による根拠付きで抽出した", True),
    Origin.GENERAL_KNOWLEDGE: OriginSpec("正規化のときに一般知識から追加した。どの論文とも照合していない", False),
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
    return () if traversable else ("探索には使わない",)


def describe_vocabulary() -> str:
    """Render every enum as Markdown, one section per enum, in definition order.

    The text is embedded in agent prompts, so wording and order must stay stable; a snapshot test guards it.
    Each enum is an `###` because the prompts drop this text under their own `## Vocabulary` heading.
    """
    lines = ["### ConceptType", ""]
    for concept_type in ConceptType:
        spec = concept_type.spec
        lines.append(_member_line(concept_type.value, spec.description, *_traversal_notes(spec.traversable)))

    lines += ["", "### PaperConceptRelation", ""]
    for paper_relation in PaperConceptRelation:
        relation_spec = paper_relation.spec
        lines.append(
            _member_line(paper_relation.value, relation_spec.description, *_traversal_notes(relation_spec.traversable))
        )

    lines += ["", "### ConceptRelationType", ""]
    for concept_relation in ConceptRelationType:
        edge_spec = concept_relation.spec
        notes = (f"方向: {edge_spec.direction}", *_traversal_notes(edge_spec.traversable))
        lines.append(_member_line(concept_relation.value, edge_spec.description, *notes))

    lines += ["", "### Origin", ""]
    for origin in Origin:
        origin_spec = origin.spec
        note = "evidence が必須" if origin_spec.requires_evidence else "evidence なし"
        lines.append(_member_line(origin.value, origin_spec.description, note))

    return "\n".join(lines) + "\n"
