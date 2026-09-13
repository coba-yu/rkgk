"""Builds the prompt that asks an agent for the relations between concepts that are already merged.

The prose that is only this prompt's lives in the .md files next to this module, and the sections the
prompts share in ../vocabulary.py and ../retry.py; this module assembles them and loops over the concepts.
The merged concepts are shown with the relations the papers state between them, and nothing else of the papers,
so the agent answers from general knowledge and leaves out the relations the graph already carries.
"""

from pathlib import Path

from rkgk.domain.models.concept_normalization import (
    ConceptNormalizationIssue,
    NormalizedConcept,
    PaperStatedRelation,
)
from rkgk.domain.prompts.retry import build_retry_section
from rkgk.domain.prompts.vocabulary import build_vocabulary_section

_DIR = Path(__file__).parent


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").rstrip("\n")


def _describe_concepts(concepts: tuple[NormalizedConcept, ...]) -> list[str]:
    lines = []
    for concept in concepts:
        parts = [concept.id, concept.canonical_name, concept.type.value]
        if concept.aliases:
            parts.append(f"aliases: {', '.join(concept.aliases)}")
        if concept.description:
            parts.append(concept.description)
        lines.append("- " + " | ".join(parts))
    return lines


# The second sentence spells out the line format `_describe_concepts` writes, so the two sit in one module and
# cannot be changed apart.
_CONCEPTS_HEADING = (
    "# Concepts",
    "",
    "以下は統合済みの正規化概念であり、`source_id` と `target_id` にはここに並ぶ slug だけを書く。",
    "概念の行は `- slug | canonical_name | type | aliases: ... | description` という形式で、概念が aliases や "
    "description を持たない場合は途中で終わる。",
)


def _describe_paper_relations(paper_relations: tuple[PaperStatedRelation, ...]) -> list[str]:
    # Never an empty list: a section with a heading and nothing under it reads as a section the prompt forgot
    # to fill in rather than as papers that state no relation at all.
    if not paper_relations:
        return ["なし"]
    return [
        f"- {relation.source_id} | {relation.relation.value} | {relation.target_id}" for relation in paper_relations
    ]


# The second sentence spells out the line format `_describe_paper_relations` writes, so the two sit in one module
# and cannot be changed apart.
_PAPER_RELATIONS_HEADING = (
    "# Paper-stated relations",
    "",
    "以下は論文が本文で述べた関係で、構築時に論文由来の辺としてグラフに載るため、同じ `source_id`、`target_id`、"
    "`relation` の組は提案しない。",
    "関係の行は `- source_id | relation | target_id` という形式で、論文が述べた関係が 1 つもない場合は"
    "「なし」とだけ書く。",
)


def build_general_knowledge_relations_prompt(
    concepts: tuple[NormalizedConcept, ...],
    paper_relations: tuple[PaperStatedRelation, ...],
    previous: object | None = None,
    issues: tuple[ConceptNormalizationIssue, ...] = (),
) -> str:
    """Write the instructions, the merged concepts and the stated relations an agent needs to relate the rest.

    A retry gets the rejected JSON and the issues appended, so the agent corrects its own answer instead of
    starting over and losing the parts that were already right.
    """
    lines = [
        _read(_DIR / "task.md"),
        "",
        *build_vocabulary_section(),
        "",
        _read(_DIR / "rules.md"),
        "",
        *_CONCEPTS_HEADING,
        "",
        *_describe_concepts(concepts),
        "",
        *_PAPER_RELATIONS_HEADING,
        "",
        *_describe_paper_relations(paper_relations),
    ]
    if previous is not None:
        lines += build_retry_section(previous, issues)
    return "\n".join(lines) + "\n"
