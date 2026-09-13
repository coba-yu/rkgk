from pathlib import Path

from rkgk.domain.models.concept_normalization import ConceptNormalizationIssue, GroupedConcept
from rkgk.domain.models.paper_extraction import ExtractedConcept
from rkgk.domain.models.vocabulary import ConceptType, describe_vocabulary
from rkgk.domain.prompts.concept_merge.builder import build_concept_merge_prompt

GROUP = (
    GroupedConcept(
        paper_id=1,
        concept=ExtractedConcept(
            local_id="c1",
            name="Retrieval-Augmented Generation",
            type=ConceptType.METHOD,
            aliases=("RAG", "検索拡張生成"),
            description="Generation grounded in retrieved passages.",
        ),
    ),
    GroupedConcept(paper_id=2, concept=ExtractedConcept(local_id="c1", name="RAG", type=ConceptType.METHOD)),
    GroupedConcept(
        paper_id=2,
        concept=ExtractedConcept(
            local_id="c2",
            name="Knowledge Graph",
            type=ConceptType.METHOD,
            aliases=("KG",),
            description="A graph of concepts and their relations.",
        ),
    ),
)

PROMPT_SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "prompt.md"


def test_the_prompt_for_the_group_matches_the_snapshot() -> None:
    assert build_concept_merge_prompt(GROUP) == PROMPT_SNAPSHOT_PATH.read_text(encoding="utf-8")


def test_the_prompt_is_deterministic() -> None:
    assert build_concept_merge_prompt(GROUP) == build_concept_merge_prompt(GROUP)


def test_the_prompt_carries_the_vocabulary() -> None:
    assert describe_vocabulary().rstrip("\n") in build_concept_merge_prompt(GROUP)


def test_the_prompt_lists_every_concept_of_the_group_with_its_paper_aliases_and_description() -> None:
    prompt = build_concept_merge_prompt(GROUP)
    assert (
        "- 1:c1 | Retrieval-Augmented Generation | method | aliases: RAG, 検索拡張生成 | "
        "Generation grounded in retrieved passages." in prompt
    )
    assert "- 2:c1 | RAG | method\n" in prompt
    assert "- 2:c2 | Knowledge Graph | method | aliases: KG | A graph of concepts and their relations." in prompt


def test_the_prompt_groups_the_concepts_without_a_heading_per_paper() -> None:
    assert '<paper id="1">' not in build_concept_merge_prompt(GROUP)


def test_the_prompt_ends_with_the_last_concept_of_the_group() -> None:
    assert build_concept_merge_prompt(GROUP).endswith(
        "- 2:c2 | Knowledge Graph | method | aliases: KG | A graph of concepts and their relations.\n"
    )


def test_a_first_attempt_mentions_neither_a_previous_answer_nor_issues() -> None:
    prompt = build_concept_merge_prompt(GROUP)
    assert "Previous attempt" not in prompt
    assert "## Issues" not in prompt


def test_a_retry_repeats_the_rejected_json_and_the_issues() -> None:
    previous = {"concepts": []}
    issues = (ConceptNormalizationIssue(path="concepts", message="paper 2 'c2' is in no merged_from"),)
    prompt = build_concept_merge_prompt(GROUP, previous, issues)
    assert '"concepts": []' in prompt
    assert "- concepts: paper 2 'c2' is in no merged_from" in prompt
    assert "上記のすべての問題を修正した、完全な JSON オブジェクトを返す。" in prompt


def test_the_prompt_never_asks_for_a_relation() -> None:
    assert "rationale" not in build_concept_merge_prompt(GROUP)
