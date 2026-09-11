"""Builds the prompt that asks an agent to normalize the concepts extracted from every paper.

The static prose lives in the .md files next to this module and in ../shared; this module only assembles
the sections, loops over the extracted concepts, and appends the retry section.
"""

import json
from pathlib import Path

from rkgk.domain.models.concept_normalization import ConceptNormalizationIssue
from rkgk.domain.models.paper_extraction import PaperExtraction
from rkgk.domain.models.vocabulary import describe_vocabulary

_DIR = Path(__file__).parent
_SHARED_DIR = _DIR.parent / "shared"


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


def build_concept_normalization_prompt(
    extractions: tuple[PaperExtraction, ...],
    previous: object | None = None,
    issues: tuple[ConceptNormalizationIssue, ...] = (),
) -> str:
    """Write the instructions and the extracted concepts an agent needs to normalize them in one pass.

    A retry gets the rejected JSON and the issues appended, so the agent corrects its own answer instead of
    starting over and losing the parts that were already right.
    """
    lines = [
        _read(_DIR / "task.md"),
        "",
        _read(_DIR / "rules.md"),
        "",
        _read(_SHARED_DIR / "vocabulary.md"),
        "",
        describe_vocabulary().rstrip("\n"),
        "",
        _read(_DIR / "papers.md"),
    ]
    for extraction in extractions:
        lines += ["", f"## Paper {extraction.paper_id}", "", *_describe_concepts(extraction)]
    if previous is not None:
        lines += [
            "",
            _read(_SHARED_DIR / "previous_attempt.md"),
            "",
            "```json",
            json.dumps(previous, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Issues",
            "",
            *(f"- {issue.path}: {issue.message}" for issue in issues),
            "",
            _read(_SHARED_DIR / "issues.md"),
        ]
    lines += ["", _read(_SHARED_DIR / "closing.md")]
    return "\n".join(lines) + "\n"
