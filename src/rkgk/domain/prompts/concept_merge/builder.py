"""Builds the prompt that asks an agent to merge one group of extracted concepts into normalized ones.

The prose that is only this prompt's lives in the .md files next to this module, and the sections the
prompts share in ../vocabulary.py and ../retry.py; this module assembles them and loops over the group.
A group holds concepts from any number of papers and is small enough to read whole, so the paper travels with
each concept instead of being the heading its concepts sit under.
"""

from collections.abc import Sequence
from pathlib import Path

from rkgk.domain.models.concept_normalization import ConceptNormalizationIssue, GroupedConcept
from rkgk.domain.prompts.retry import build_retry_section
from rkgk.domain.prompts.vocabulary import build_vocabulary_section

_DIR = Path(__file__).parent


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").rstrip("\n")


def _describe_group(group: Sequence[GroupedConcept]) -> list[str]:
    lines = []
    for item in group:
        concept = item.concept
        parts = [f"{item.paper_id}:{concept.local_id}", concept.name, concept.type.value]
        if concept.aliases:
            parts.append(f"aliases: {', '.join(concept.aliases)}")
        if concept.description:
            parts.append(concept.description)
        lines.append("- " + " | ".join(parts))
    return lines


# The second sentence spells out the line format `_describe_group` writes, so the two sit in one module and
# cannot be changed apart.
_CONCEPTS_HEADING = (
    "# Concepts",
    "",
    "`paper_id:local_id` は抽出元の論文 id と、その論文の中でのみ通用するローカル id である。",
    "概念の行は `- paper_id:local_id | name | type | aliases: ... | description` という形式で、論文が aliases や "
    "description を報告していない場合は途中で終わる。",
)


def build_concept_merge_prompt(
    group: tuple[GroupedConcept, ...],
    previous: object | None = None,
    issues: tuple[ConceptNormalizationIssue, ...] = (),
) -> str:
    """Write the instructions and the grouped concepts an agent needs to merge them into normalized concepts.

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
        *_describe_group(group),
    ]
    if previous is not None:
        lines += build_retry_section(previous, issues)
    return "\n".join(lines) + "\n"
