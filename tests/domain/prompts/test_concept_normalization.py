from pathlib import Path

from rkgk.domain.models.concept_normalization import CONCEPT_NORMALIZATION_SCHEMA_VERSION, ConceptNormalizationIssue
from rkgk.domain.models.paper_extraction import ExtractedConcept, PaperExtraction
from rkgk.domain.models.vocabulary import ConceptType, describe_vocabulary
from rkgk.domain.prompts.concept_normalization import build_concept_normalization_prompt

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

PROMPT_SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "concept_normalization_prompt.md"


def test_the_prompt_for_the_extractions_matches_the_snapshot() -> None:
    assert build_concept_normalization_prompt(EXTRACTIONS) == PROMPT_SNAPSHOT_PATH.read_text(encoding="utf-8")


def test_the_prompt_is_deterministic() -> None:
    assert build_concept_normalization_prompt(EXTRACTIONS) == build_concept_normalization_prompt(EXTRACTIONS)


def test_the_prompt_carries_the_vocabulary_and_the_schema_version() -> None:
    prompt = build_concept_normalization_prompt(EXTRACTIONS)
    assert describe_vocabulary().rstrip("\n") in prompt
    assert f"`schema_version` に {CONCEPT_NORMALIZATION_SCHEMA_VERSION} を設定する。" in prompt


def test_the_prompt_lists_every_paper_with_its_concepts_aliases_and_description() -> None:
    prompt = build_concept_normalization_prompt(EXTRACTIONS)
    assert "## Paper 1" in prompt
    assert "## Paper 2" in prompt
    assert (
        "- c1 | Retrieval-Augmented Generation | method | aliases: RAG, 検索拡張生成 | "
        "Generation grounded in retrieved passages." in prompt
    )
    assert "- c2 | Page-Aligned Chunking | method\n" in prompt


def test_the_prompt_ends_by_asking_for_the_json_alone() -> None:
    assert build_concept_normalization_prompt(EXTRACTIONS).endswith("JSON オブジェクトだけを返す。\n")


def test_a_first_attempt_mentions_neither_a_previous_answer_nor_issues() -> None:
    prompt = build_concept_normalization_prompt(EXTRACTIONS)
    assert "Previous attempt" not in prompt
    assert "## Issues" not in prompt


def test_a_retry_repeats_the_rejected_json_and_the_issues() -> None:
    previous = {"schema_version": 1, "concepts": []}
    issues = (ConceptNormalizationIssue(path="concepts", message="paper 2 'c2' is in no merged_from"),)
    prompt = build_concept_normalization_prompt(EXTRACTIONS, previous, issues)
    assert '"schema_version": 1' in prompt
    assert "- concepts: paper 2 'c2' is in no merged_from" in prompt
    assert "上記のすべての問題を修正した、完全な JSON オブジェクトを返す。" in prompt
