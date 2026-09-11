"""Builds the prompt that asks an agent to normalize the concepts extracted from every paper.

Kept next to `check_normalization_against_extractions` so the rules stated here and the rules enforced there
cannot drift apart.
"""

import json

from rkgk.domain.models.concept_normalization import (
    CONCEPT_NORMALIZATION_SCHEMA_VERSION,
    ConceptNormalizationIssue,
)
from rkgk.domain.models.paper_extraction import PaperExtraction
from rkgk.domain.models.vocabulary import describe_vocabulary

_RULES = (
    "Merge two concepts only when they mean the same thing, and keep concepts whose meaning you cannot tell "
    "apart as separate normalized concepts.",
    "Write `canonical_name` as the English canonical name of the concept, and derive `id` from it as its "
    "lowercase words joined by `-`, matching `^[a-z0-9]+(-[a-z0-9]+)*$`.",
    "Collect every spelling, abbreviation and Japanese name of the merged concepts in `aliases`.",
    "List every extracted concept in exactly one `merged_from`, as the paper id and the local id it was "
    "extracted under.",
    "Keep `type` the type the merged concepts were extracted with.",
    "Never add a concept that no paper extracted.",
    "Add concept relations from general knowledge between normalized concepts only, and name no other id.",
    "Prefer `is_a`, `part_of` and `used_for` for those relations, and use `related_to` only when none of the "
    "other three fits.",
    "Give every relation a `rationale` of one sentence saying why it holds, because a general-knowledge "
    "relation carries no evidence.",
    f"Set `schema_version` to {CONCEPT_NORMALIZATION_SCHEMA_VERSION}.",
)


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
        "# Task",
        "",
        "You merge the concepts extracted from several research papers into one shared vocabulary.",
        "Read the concepts below, report each of them once as a normalized concept, and say how those "
        "normalized concepts relate to each other.",
        "Answer with a single JSON object that follows the JSON Schema you were given.",
        "",
        "## Rules",
        "",
        *(f"- {rule}" for rule in _RULES),
        "",
        "## Vocabulary",
        "",
        "Use these node types and relation names, spelled exactly as shown.",
        "",
        describe_vocabulary().rstrip("\n"),
        "",
        "## Papers",
        "",
        "Each paper was extracted on its own, so `c1`, `c2`, ... are local to the paper they are listed under.",
        "A concept line reads `- local id | name | type | aliases: ... | description`, and it ends early when "
        "the paper reported no aliases or no description.",
    ]
    for extraction in extractions:
        lines += ["", f"## Paper {extraction.paper_id}", "", *_describe_concepts(extraction)]
    if previous is not None:
        lines += [
            "",
            "## Previous attempt",
            "",
            "This JSON was rejected.",
            "",
            "```json",
            json.dumps(previous, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Issues",
            "",
            *(f"- {issue.path}: {issue.message}" for issue in issues),
            "",
            "Return a complete corrected JSON object that fixes every issue above.",
        ]
    lines += ["", "Return only the JSON object."]
    return "\n".join(lines) + "\n"
