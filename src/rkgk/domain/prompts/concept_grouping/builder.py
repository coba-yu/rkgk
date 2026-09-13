"""Builds the prompt that asks an agent to bundle the extracted concepts that may turn out to be the same.

The prose that is only this prompt's lives in the .md files next to this module, and the sections the
prompts share in ../vocabulary.py and ../retry.py; this module assembles them and loops over the concepts and
the candidate pairs.
Only the name and the type of a concept are shown here, because this stage decides what to look at together and
the merge stage is the one that reads the aliases and the descriptions.
"""

from collections.abc import Sequence
from pathlib import Path

from rkgk.domain.models.concept_normalization import CandidatePair, ConceptNormalizationIssue
from rkgk.domain.models.paper_extraction import PaperExtraction
from rkgk.domain.prompts.retry import build_retry_section
from rkgk.domain.prompts.vocabulary import build_vocabulary_section

_DIR = Path(__file__).parent


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").rstrip("\n")


def _describe_concepts(extraction: PaperExtraction) -> list[str]:
    return [f"- {concept.local_id} | {concept.name} | {concept.type.value}" for concept in extraction.concepts]


# The second sentence spells out the line format `_describe_concepts` writes, so the two sit in one module and
# cannot be changed apart.
_PAPERS_HEADING = (
    "# Papers",
    "",
    "論文ごとに個別に抽出したので、`c1`、`c2`、... はそれが記載されている論文の中でのみ通用する。",
    "概念の行は `- local_id | name | type` という形式である。",
)


def _describe_pairs(pairs: Sequence[CandidatePair]) -> list[str]:
    # Never an empty list: a section with a heading and nothing under it reads as a section the prompt forgot to
    # fill in rather than as a corpus in which no two concepts resemble each other.
    if not pairs:
        return ["なし"]
    return [
        f"- {pair.left.paper_id}:{pair.left.local_id} | {pair.right.paper_id}:{pair.right.local_id} | {pair.score:.3f}"
        for pair in pairs
    ]


# The second sentence spells out the line format `_describe_pairs` writes, so the two sit in one module and
# cannot be changed apart.
_PAIRS_HEADING = (
    "# Candidate pairs",
    "",
    "以下は概念の名前と説明を埋め込んで近かった組で、どの概念を一緒に見るかの手がかりにする。",
    "組の行は `- paper_id:local_id | paper_id:local_id | score` という形式で、`score` はコサイン類似度である。"
    "近い組が 1 つもない場合は「なし」とだけ書く。",
)


def build_concept_grouping_prompt(
    extractions: tuple[PaperExtraction, ...],
    pairs: tuple[CandidatePair, ...],
    previous: object | None = None,
    issues: tuple[ConceptNormalizationIssue, ...] = (),
) -> str:
    """Write the instructions, the extracted concepts and the close pairs an agent needs to bundle them.

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
    lines += ["", *_PAIRS_HEADING, "", *_describe_pairs(pairs)]
    if previous is not None:
        lines += build_retry_section(previous, issues)
    return "\n".join(lines) + "\n"
