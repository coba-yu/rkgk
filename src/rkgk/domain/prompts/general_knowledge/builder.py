"""Builds the prompt that asks an agent for the relations between concepts that are already merged.

The prose that is only this prompt's lives in the .md files next to this module, and the sections the
prompts share in ../vocabulary.py and ../retry.py; this module assembles them and loops over the concepts.
Only the merged concepts are shown: what the papers stated about them is left out so the agent answers from
general knowledge and cannot copy an edge a paper already carries.
"""

from pathlib import Path

from rkgk.domain.models.concept_normalization import ConceptNormalizationIssue, NormalizedConcept
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


def build_general_knowledge_prompt(
    concepts: tuple[NormalizedConcept, ...],
    previous: object | None = None,
    issues: tuple[ConceptNormalizationIssue, ...] = (),
) -> str:
    """Write the instructions and the merged concepts an agent needs to relate them from general knowledge.

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
    ]
    if previous is not None:
        lines += build_retry_section(previous, issues)
    return "\n".join(lines) + "\n"
