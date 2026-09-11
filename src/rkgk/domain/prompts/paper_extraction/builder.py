"""Builds the prompt that asks an agent to extract the knowledge graph of one paper.

The static prose lives in the .md files next to this module and in ../shared, and the retry wording in
../retry.py; this module only assembles the sections and loops over the paper's pages.
"""

from pathlib import Path

from rkgk.domain.models.paper import Paper
from rkgk.domain.models.paper_extraction import PaperExtractionIssue
from rkgk.domain.models.vocabulary import describe_vocabulary
from rkgk.domain.prompts.retry import build_retry_section

_DIR = Path(__file__).parent
_SHARED_DIR = _DIR.parent / "shared"


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
        _read(_SHARED_DIR / "vocabulary.md"),
        "",
        describe_vocabulary().rstrip("\n"),
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
    lines += ["", _read(_SHARED_DIR / "closing.md")]
    return "\n".join(lines) + "\n"
