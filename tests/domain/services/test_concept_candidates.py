import pytest

from rkgk.domain.models.concept_normalization import LocalConceptRef
from rkgk.domain.models.paper_extraction import ExtractedConcept, PaperExtraction
from rkgk.domain.models.vocabulary import ConceptType
from rkgk.domain.services.concept_candidates import (
    build_extracted_concept_embedding_text,
    collect_candidate_pairs,
)
from rkgk.infrastructure.fake_embedder import FakeEmbedder

NAMES = {
    1: ("Retrieval-Augmented Generation", "Page-Aligned Chunking", "Dense Retrieval"),
    # Paper 2 extracts the first concept of paper 1 under the very same name, so the two share a vector.
    2: ("Retrieval-Augmented Generation", "Knowledge Graph", "Sparse Retrieval"),
}


def build_extraction(paper_id: int) -> PaperExtraction:
    return PaperExtraction(
        schema_version=1,
        paper_id=paper_id,
        summary_ja="この論文は検索の手法を提案する。",
        concepts=tuple(
            ExtractedConcept(local_id=f"c{index}", name=name, type=ConceptType.METHOD)
            for index, name in enumerate(NAMES[paper_id], start=1)
        ),
        paper_concepts=(),
    )


EXTRACTIONS = (build_extraction(1), build_extraction(2))


def build_single_concept_extraction() -> PaperExtraction:
    return PaperExtraction(
        schema_version=1,
        paper_id=1,
        summary_ja="この論文は検索の手法を提案する。",
        concepts=(ExtractedConcept(local_id="c1", name="Knowledge Graph", type=ConceptType.METHOD),),
        paper_concepts=(),
    )


def collect_keys(extractions: tuple[PaperExtraction, ...], neighbors: int) -> list[tuple[str, str]]:
    pairs = collect_candidate_pairs(extractions, FakeEmbedder(), neighbors)
    return [
        (f"{pair.left.paper_id}:{pair.left.local_id}", f"{pair.right.paper_id}:{pair.right.local_id}")
        for pair in pairs
    ]


def test_a_concept_is_rendered_with_its_aliases_in_brackets_and_its_description_below() -> None:
    concept = ExtractedConcept(
        local_id="c1",
        name="Retrieval-Augmented Generation",
        type=ConceptType.METHOD,
        aliases=("RAG", "検索拡張生成"),
        description="Generation grounded in retrieved passages.",
    )
    assert build_extracted_concept_embedding_text(concept) == (
        "Retrieval-Augmented Generation (RAG, 検索拡張生成)\nGeneration grounded in retrieved passages."
    )


def test_a_concept_without_aliases_or_a_description_is_rendered_as_its_name_alone() -> None:
    concept = ExtractedConcept(local_id="c1", name="Knowledge Graph", type=ConceptType.METHOD)
    assert build_extracted_concept_embedding_text(concept) == "Knowledge Graph"


def test_every_concept_is_paired_with_as_many_others_as_the_neighbors_allow() -> None:
    pairs = collect_candidate_pairs(EXTRACTIONS, FakeEmbedder(), neighbors=1)
    # Six concepts each keep one neighbour, and a pair two concepts both kept is reported once.
    assert 3 <= len(pairs) <= 6
    for pair in pairs:
        assert pair.left != pair.right


def test_more_neighbors_never_yield_fewer_pairs() -> None:
    fewer = collect_candidate_pairs(EXTRACTIONS, FakeEmbedder(), neighbors=1)
    more = collect_candidate_pairs(EXTRACTIONS, FakeEmbedder(), neighbors=3)
    assert len(more) >= len(fewer)


def test_asking_for_more_neighbors_than_there_are_concepts_pairs_every_concept_with_every_other() -> None:
    assert len(collect_candidate_pairs(EXTRACTIONS, FakeEmbedder(), neighbors=99)) == 15


def test_a_pair_is_reported_once_whichever_of_its_two_concepts_found_the_other() -> None:
    keys = collect_keys(EXTRACTIONS, neighbors=99)
    assert len(keys) == len(set(keys))
    assert not any((right, left) in set(keys) for left, right in keys)


def test_two_concepts_of_one_paper_are_paired_like_any_others() -> None:
    keys = collect_keys(EXTRACTIONS, neighbors=99)
    assert ("1:c1", "1:c2") in keys


def test_the_pairs_come_out_from_the_closest_to_the_furthest() -> None:
    pairs = collect_candidate_pairs(EXTRACTIONS, FakeEmbedder(), neighbors=99)
    assert [pair.score for pair in pairs] == sorted((pair.score for pair in pairs), reverse=True)


def test_the_same_corpus_yields_the_same_pairs_in_the_same_order() -> None:
    assert collect_candidate_pairs(EXTRACTIONS, FakeEmbedder(), neighbors=3) == collect_candidate_pairs(
        EXTRACTIONS, FakeEmbedder(), neighbors=3
    )


def test_two_concepts_that_read_the_same_come_first_and_score_one() -> None:
    pairs = collect_candidate_pairs(EXTRACTIONS, FakeEmbedder(), neighbors=99)
    assert pairs[0].left == LocalConceptRef(paper_id=1, local_id="c1")
    assert pairs[0].right == LocalConceptRef(paper_id=2, local_id="c1")
    assert pairs[0].score == pytest.approx(1.0)


def test_a_corpus_of_one_concept_has_nothing_to_pair() -> None:
    assert collect_candidate_pairs((build_single_concept_extraction(),), FakeEmbedder()) == ()


@pytest.mark.parametrize("neighbors", [0, -1])
def test_fewer_than_one_neighbor_is_rejected(neighbors: int) -> None:
    with pytest.raises(ValueError, match="neighbors must be at least 1"):
        collect_candidate_pairs(EXTRACTIONS, FakeEmbedder(), neighbors)
