"""Builds the prompt that asks an agent to extract the knowledge graph of one paper.

Kept next to `check_extraction_against_paper` so the rules stated here and the rules enforced there cannot
drift apart.
"""

import json

from rkgk.domain.models.paper import Paper
from rkgk.domain.models.paper_extraction import EXTRACTION_SCHEMA_VERSION, PaperExtractionIssue
from rkgk.domain.models.vocabulary import describe_vocabulary

_RULES = (
    "Report only concepts that appear in the paper text; never add a concept from general knowledge.",
    "Write `name` as the English canonical name of the concept, and put abbreviations and spelling variants "
    "in `aliases`.",
    "Use `proposes` for a concept the paper introduces as its own contribution and `uses` for one it applies "
    "in its method or experiments.",
    "Use `addresses` for a problem the paper tries to solve and `mentions` when the paper only refers to the "
    "concept; the line between `mentions` and `uses` is whether the paper actually works with the concept.",
    "Copy every `quote` verbatim from the paper text; never summarize or paraphrase it.",
    "Set every `page` to the page the quote is printed on.",
    "Add a concept-to-concept relation only when the paper text backs it, and leave `concept_relations` empty "
    "otherwise.",
    "Write `summary_ja` in Japanese, between 200 and 400 characters, covering the problem, the method, and "
    "the results.",
)


def _identifier_rules(paper_id: int) -> tuple[str, ...]:
    return (
        "Give every concept a local id `c1`, `c2`, ... in declaration order.",
        "Refer to those local ids only; `paper_concepts` and `concept_relations` may use no other id.",
        "Never invent a global id or a slug, because normalization assigns them after this step.",
        f"Set `paper_id` to {paper_id}.",
        f"Set `schema_version` to {EXTRACTION_SCHEMA_VERSION}.",
    )


def build_paper_extraction_prompt(
    paper: Paper, previous: object | None = None, issues: tuple[PaperExtractionIssue, ...] = ()
) -> str:
    """Write the instructions and the paper text an agent needs to extract this one paper.

    A retry gets the rejected JSON and the issues appended, so the agent corrects its own answer instead of
    starting over and losing the parts that were already right.
    """
    lines = [
        "# Task",
        "",
        "You extract the knowledge graph of one research paper.",
        "Read the paper below and report the concepts it contains, how the paper relates to them, and how they "
        "relate to each other.",
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
        "## Identifiers",
        "",
        *(f"- {rule}" for rule in _identifier_rules(paper.meta.id)),
        "",
        "## Paper",
        "",
        f"Title: {paper.meta.title}",
        f"Year: {paper.meta.year}",
        f"Venue: {paper.meta.venue}",
    ]
    for page in paper.pages:
        lines += ["", f"## Page {page.number}", "", page.text.rstrip("\n")]
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
