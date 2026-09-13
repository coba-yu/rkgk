from pathlib import Path

from rkgk.domain.models.concept_normalization import (
    ConceptNormalizationIssue,
    LocalConceptRef,
    NormalizedConcept,
    PaperStatedRelation,
)
from rkgk.domain.models.vocabulary import ConceptRelationType, ConceptType, describe_vocabulary
from rkgk.domain.prompts.general_knowledge_relations.builder import build_general_knowledge_relations_prompt

CONCEPTS = (
    NormalizedConcept(
        id="retrieval-augmented-generation",
        canonical_name="Retrieval-Augmented Generation",
        type=ConceptType.METHOD,
        aliases=("RAG", "検索拡張生成"),
        description="Generation grounded in retrieved passages.",
        merged_from=(LocalConceptRef(paper_id=1, local_id="c1"), LocalConceptRef(paper_id=2, local_id="c1")),
    ),
    NormalizedConcept(
        id="page-aligned-chunking",
        canonical_name="Page-Aligned Chunking",
        type=ConceptType.METHOD,
        merged_from=(LocalConceptRef(paper_id=1, local_id="c2"),),
    ),
    NormalizedConcept(
        id="knowledge-graph",
        canonical_name="Knowledge Graph",
        type=ConceptType.METHOD,
        aliases=("KG",),
        description="A graph of concepts and their relations.",
        merged_from=(LocalConceptRef(paper_id=2, local_id="c2"),),
    ),
)

PAPER_RELATIONS = (
    PaperStatedRelation(
        source_id="page-aligned-chunking",
        target_id="retrieval-augmented-generation",
        relation=ConceptRelationType.PART_OF,
        paper_ids=(1,),
    ),
    PaperStatedRelation(
        source_id="knowledge-graph",
        target_id="retrieval-augmented-generation",
        relation=ConceptRelationType.USED_FOR,
        paper_ids=(1, 2),
    ),
)

PROMPT_SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "prompt.md"


def test_the_prompt_for_the_concepts_and_the_stated_relations_matches_the_snapshot() -> None:
    assert build_general_knowledge_relations_prompt(CONCEPTS, PAPER_RELATIONS) == PROMPT_SNAPSHOT_PATH.read_text(
        encoding="utf-8"
    )


def test_the_prompt_is_deterministic() -> None:
    assert build_general_knowledge_relations_prompt(
        CONCEPTS, PAPER_RELATIONS
    ) == build_general_knowledge_relations_prompt(CONCEPTS, PAPER_RELATIONS)


def test_the_prompt_carries_the_vocabulary() -> None:
    assert describe_vocabulary().rstrip("\n") in build_general_knowledge_relations_prompt(CONCEPTS, PAPER_RELATIONS)


def test_the_prompt_lists_every_concept_with_its_slug_aliases_and_description() -> None:
    prompt = build_general_knowledge_relations_prompt(CONCEPTS, PAPER_RELATIONS)
    assert (
        "- retrieval-augmented-generation | Retrieval-Augmented Generation | method | aliases: RAG, 検索拡張生成 | "
        "Generation grounded in retrieved passages." in prompt
    )
    assert "- page-aligned-chunking | Page-Aligned Chunking | method\n" in prompt
    assert (
        "- knowledge-graph | Knowledge Graph | method | aliases: KG | A graph of concepts and their relations."
        in prompt
    )


def test_the_prompt_says_nothing_about_the_papers_the_concepts_came_from() -> None:
    prompt = build_general_knowledge_relations_prompt(CONCEPTS, PAPER_RELATIONS)
    assert "merged_from" not in prompt
    assert "<paper" not in prompt


def test_the_prompt_ends_with_the_last_relation_the_papers_state() -> None:
    assert build_general_knowledge_relations_prompt(CONCEPTS, PAPER_RELATIONS).endswith(
        "- knowledge-graph | used_for | retrieval-augmented-generation\n"
    )


def test_the_prompt_lists_every_relation_the_papers_state_without_naming_the_papers() -> None:
    prompt = build_general_knowledge_relations_prompt(CONCEPTS, PAPER_RELATIONS)
    assert "# Paper-stated relations" in prompt
    assert "- page-aligned-chunking | part_of | retrieval-augmented-generation\n" in prompt
    assert "- knowledge-graph | used_for | retrieval-augmented-generation\n" in prompt
    assert "paper_ids" not in prompt


def test_the_prompt_forbids_proposing_a_relation_the_papers_state() -> None:
    prompt = build_general_knowledge_relations_prompt(CONCEPTS, PAPER_RELATIONS)
    assert "- 論文が述べた関係として列挙された組は提案しない。" in prompt
    assert "同じ `source_id`、`target_id`、`relation` の組は提案しない。" in prompt


def test_the_prompt_says_none_when_no_paper_states_a_relation() -> None:
    prompt = build_general_knowledge_relations_prompt(CONCEPTS, ())
    assert "# Paper-stated relations" in prompt
    assert prompt.endswith("なし\n")
    assert " | part_of | " not in prompt


def test_a_first_attempt_mentions_neither_a_previous_answer_nor_issues() -> None:
    prompt = build_general_knowledge_relations_prompt(CONCEPTS, PAPER_RELATIONS)
    assert "Previous attempt" not in prompt
    assert "## Issues" not in prompt


def test_a_retry_repeats_the_rejected_json_and_the_issues() -> None:
    previous = {"concept_relations": [{"source_id": "dense-retrieval"}]}
    issues = (
        ConceptNormalizationIssue(
            path="concept_relations[0].source_id",
            message="is 'dense-retrieval', which is not one of the normalized concepts",
        ),
    )
    prompt = build_general_knowledge_relations_prompt(CONCEPTS, PAPER_RELATIONS, previous, issues)
    assert '"source_id": "dense-retrieval"' in prompt
    assert (
        "- concept_relations[0].source_id: is 'dense-retrieval', which is not one of the normalized concepts" in prompt
    )
    assert "上記のすべての問題を修正した、完全な JSON オブジェクトを返す。" in prompt
