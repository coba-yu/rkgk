"""Builds the prompt that asks an agent to merge the concepts extracted from every paper into one vocabulary.

The prose that is only this prompt's lives in the .md files next to this module, and the sections the
prompts share in ../vocabulary.py and ../retry.py; this module assembles them and loops over the extracted concepts.
"""

from pathlib import Path

from rkgk.domain.models.concept_normalization import ConceptNormalizationIssue
from rkgk.domain.models.paper_extraction import PaperExtraction
from rkgk.domain.prompts.retry import build_retry_section
from rkgk.domain.prompts.vocabulary import build_vocabulary_section

_DIR = Path(__file__).parent


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").rstrip("\n")


def _describe_concepts(extraction: PaperExtraction) -> list[str]:
    lines = []
    for concept in extraction.concepts:
        parts = [concept.local_id, concept.name, concept.type.value]
        if concept.aliases:
            parts.append(f"aliases: {', '.join(concept.aliases)}")
        if concept.description:
            parts.append(concept.description)
        lines.append("- " + " | ".join(parts))
    return lines


# The second sentence spells out the line format `_describe_concepts` writes, so the two sit in one module and
# cannot be changed apart.
_PAPERS_HEADING = (
    "# Papers",
    "",
    "論文ごとに個別に抽出したので、`c1`、`c2`、... はそれが記載されている論文の中でのみ通用する。",
    "概念の行は `- local id | name | type | aliases: ... | description` という形式で、論文が aliases や "
    "description を報告していない場合は途中で終わる。",
)


def build_concept_merge_prompt(
    extractions: tuple[PaperExtraction, ...],
    previous: object | None = None,
    issues: tuple[ConceptNormalizationIssue, ...] = (),
) -> str:
    """Write the instructions and the extracted concepts an agent needs to merge them in one pass.

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
        *_PAPERS_HEADING,
    ]
    # Tagged like the pages of the extraction prompt, so a paper's concepts are bounded rather than running
    # until the next heading.
    for extraction in extractions:
        lines += ["", f'<paper id="{extraction.paper_id}">', *_describe_concepts(extraction), "</paper>"]
    if previous is not None:
        lines += build_retry_section(previous, issues)
    return "\n".join(lines) + "\n"
