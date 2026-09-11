"""Builds the prompt that asks an agent to extract the knowledge graph of one paper.

The prose that is only this prompt's lives in the .md files next to this module, and the sections both
prompts share in ../vocabulary.py and ../retry.py; this module assembles them and loops over the paper's pages.
"""

from pathlib import Path

from rkgk.domain.models.paper import Paper
from rkgk.domain.models.paper_extraction import PaperExtractionIssue
from rkgk.domain.prompts.retry import build_retry_section
from rkgk.domain.prompts.vocabulary import build_vocabulary_section

_DIR = Path(__file__).parent


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").rstrip("\n")


def build_paper_extraction_prompt(
    paper: Paper, previous: object | None = None, issues: tuple[PaperExtractionIssue, ...] = ()
) -> str:
    """Write the instructions and the paper text an agent needs to extract this one paper.

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
        "# Paper",
        "",
        f"Id: {paper.meta.id}",
        f"Title: {paper.meta.title}",
        f"Year: {paper.meta.year}",
        f"Venue: {paper.meta.venue}",
    ]
    # The page text is Markdown of its own, so it is fenced in a tag instead of a heading; a heading would
    # put the paper's own headings on the same footing as the prompt's and blur where a page starts and ends.
    for page in paper.pages:
        lines += ["", f'<page number="{page.number}">', page.text.rstrip("\n"), "</page>"]
    if previous is not None:
        lines += build_retry_section(previous, issues)
    return "\n".join(lines) + "\n"
