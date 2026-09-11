import json
from pathlib import Path

from rkgk.domain.models.paper import Paper, PaperMeta, parse_page
from rkgk.domain.models.paper_extraction import PaperExtractionIssue
from rkgk.domain.models.vocabulary import describe_vocabulary
from rkgk.domain.prompts.paper_extraction.builder import build_paper_extraction_prompt

FIXTURE_DIR = Path(__file__).parent.parent.parent.parent / "fixtures"
PROMPT_SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "prompt.md"


def load_fixture_paper() -> Paper:
    paper_dir = FIXTURE_DIR / "papers" / "0001"
    meta = PaperMeta.model_validate(json.loads((paper_dir / "paper.json").read_text(encoding="utf-8")))
    pages = tuple(
        parse_page(number, (paper_dir / "pages" / f"{number:03d}.md").read_text(encoding="utf-8"))
        for number in range(1, meta.page_count + 1)
    )
    return Paper(meta=meta, pages=pages)


def test_the_prompt_for_the_fixture_paper_matches_the_snapshot() -> None:
    assert build_paper_extraction_prompt(load_fixture_paper()) == PROMPT_SNAPSHOT_PATH.read_text(encoding="utf-8")


def test_the_prompt_is_deterministic() -> None:
    paper = load_fixture_paper()
    assert build_paper_extraction_prompt(paper) == build_paper_extraction_prompt(paper)


def test_the_prompt_carries_the_vocabulary_and_the_paper_id() -> None:
    prompt = build_paper_extraction_prompt(load_fixture_paper())
    assert describe_vocabulary().rstrip("\n") in prompt
    assert "Id: 1" in prompt
    assert "`paper_id` には `# Paper` に示した Id を設定する。" in prompt


def test_the_prompt_holds_every_page_with_its_number_and_text() -> None:
    prompt = build_paper_extraction_prompt(load_fixture_paper())
    assert prompt.count('<page number="') == 3
    assert '<page number="2">\n## Method' in prompt
    assert "<!-- equation: 1 -->" in prompt


def test_the_prompt_ends_by_asking_for_the_json_alone() -> None:
    assert build_paper_extraction_prompt(load_fixture_paper()).endswith("JSON オブジェクトだけを返す。\n")


def test_a_first_attempt_mentions_neither_a_previous_answer_nor_issues() -> None:
    prompt = build_paper_extraction_prompt(load_fixture_paper())
    assert "Previous attempt" not in prompt
    assert "## Issues" not in prompt


def test_a_retry_repeats_the_rejected_json_and_the_issues() -> None:
    previous = {"paper_id": 1, "summary_ja": "要約"}
    issues = (PaperExtractionIssue(path="paper_concepts[0].evidence[0].quote", message="is not found on page 1"),)
    prompt = build_paper_extraction_prompt(load_fixture_paper(), previous, issues)
    assert '"summary_ja": "要約"' in prompt
    assert "- paper_concepts[0].evidence[0].quote: is not found on page 1" in prompt
    assert "上記のすべての問題を修正した、完全な JSON オブジェクトを返す。" in prompt
