from pathlib import Path

from rkgk.domain.models.concept_normalization import (
    CandidatePair,
    ConceptNormalizationIssue,
    LocalConceptRef,
)
from rkgk.domain.models.paper_extraction import ExtractedConcept, PaperExtraction
from rkgk.domain.models.vocabulary import ConceptType, describe_vocabulary
from rkgk.domain.prompts.concept_grouping.builder import build_concept_grouping_prompt

EXTRACTIONS = (
    PaperExtraction(
        schema_version=1,
        paper_id=1,
        summary_ja="この論文は検索拡張生成のパイプラインを提案する。",
        concepts=(
            ExtractedConcept(
                local_id="c1",
                name="Retrieval-Augmented Generation",
                type=ConceptType.METHOD,
                aliases=("RAG", "検索拡張生成"),
                description="Generation grounded in retrieved passages.",
            ),
            ExtractedConcept(local_id="c2", name="Page-Aligned Chunking", type=ConceptType.METHOD),
        ),
        paper_concepts=(),
    ),
    PaperExtraction(
        schema_version=1,
        paper_id=2,
        summary_ja="この論文は知識グラフを検索の骨格として使う。",
        concepts=(
            ExtractedConcept(local_id="c1", name="RAG", type=ConceptType.METHOD),
            ExtractedConcept(
                local_id="c2",
                name="Knowledge Graph",
                type=ConceptType.METHOD,
                aliases=("KG",),
                description="A graph of concepts and their relations.",
            ),
        ),
        paper_concepts=(),
    ),
)

PAIRS = (
    CandidatePair(
        left=LocalConceptRef(paper_id=1, local_id="c1"),
        right=LocalConceptRef(paper_id=2, local_id="c1"),
        score=0.9123,
    ),
    CandidatePair(
        left=LocalConceptRef(paper_id=1, local_id="c2"),
        right=LocalConceptRef(paper_id=2, local_id="c2"),
        score=0.4,
    ),
)

PROMPT_SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "prompt.md"


def test_the_prompt_for_the_extractions_and_the_pairs_matches_the_snapshot() -> None:
    assert build_concept_grouping_prompt(EXTRACTIONS, PAIRS) == PROMPT_SNAPSHOT_PATH.read_text(encoding="utf-8")


def test_the_prompt_is_deterministic() -> None:
    assert build_concept_grouping_prompt(EXTRACTIONS, PAIRS) == build_concept_grouping_prompt(EXTRACTIONS, PAIRS)


def test_the_prompt_carries_the_vocabulary() -> None:
    assert describe_vocabulary().rstrip("\n") in build_concept_grouping_prompt(EXTRACTIONS, PAIRS)


def test_the_prompt_lists_every_paper_with_the_name_and_the_type_of_its_concepts() -> None:
    prompt = build_concept_grouping_prompt(EXTRACTIONS, PAIRS)
    assert '<paper id="1">' in prompt
    assert '<paper id="2">' in prompt
    assert "- c1 | Retrieval-Augmented Generation | method\n" in prompt
    assert "- c2 | Knowledge Graph | method\n" in prompt


def test_the_prompt_leaves_the_aliases_and_the_descriptions_to_the_merge_stage() -> None:
    prompt = build_concept_grouping_prompt(EXTRACTIONS, PAIRS)
    assert "aliases" not in prompt
    assert "Generation grounded in retrieved passages." not in prompt


def test_the_prompt_lists_every_pair_with_its_score_to_three_decimals() -> None:
    prompt = build_concept_grouping_prompt(EXTRACTIONS, PAIRS)
    assert "- 1:c1 | 2:c1 | 0.912" in prompt
    assert "- 1:c2 | 2:c2 | 0.400" in prompt


def test_the_prompt_says_none_when_no_two_concepts_are_close() -> None:
    assert build_concept_grouping_prompt(EXTRACTIONS, ()).endswith("なし\n")


def test_a_first_attempt_mentions_neither_a_previous_answer_nor_issues() -> None:
    prompt = build_concept_grouping_prompt(EXTRACTIONS, PAIRS)
    assert "Previous attempt" not in prompt
    assert "## Issues" not in prompt


def test_a_retry_repeats_the_rejected_json_and_the_issues() -> None:
    previous = {"groups": [[{"paper_id": 2, "local_id": "c9"}]]}
    issues = (ConceptNormalizationIssue(path="groups[0][0]", message="paper 2 has no concept 'c9'"),)
    prompt = build_concept_grouping_prompt(EXTRACTIONS, PAIRS, previous, issues)
    assert '"local_id": "c9"' in prompt
    assert "- groups[0][0]: paper 2 has no concept 'c9'" in prompt
    assert "上記のすべての問題を修正した、完全な JSON オブジェクトを返す。" in prompt


def test_the_prompt_never_asks_for_a_merged_concept_or_a_relation() -> None:
    prompt = build_concept_grouping_prompt(EXTRACTIONS, PAIRS)
    assert "canonical_name" not in prompt
    assert "rationale" not in prompt
