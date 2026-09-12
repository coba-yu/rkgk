import warnings
from collections.abc import Sequence

import numpy as np
import pytest

from rkgk.domain.models.embedding import EmbeddedItem, EmbeddedItemKind, EmbeddingTable
from rkgk.domain.models.graph import ChunkEvidence, Concept, KnowledgeGraph, PaperConceptEdge
from rkgk.domain.models.search import PaperHits
from rkgk.domain.models.vocabulary import ConceptType, PaperConceptRelation
from rkgk.domain.services.vector_search import compute_cosine_scores, search_vectors

TABLE = EmbeddingTable(
    items=(
        EmbeddedItem(kind=EmbeddedItemKind.CHUNK, ref="1:0", paper_id=1, text="We study retrieval."),
        EmbeddedItem(kind=EmbeddedItemKind.CHUNK, ref="1:1", paper_id=1, text="It helps a reader."),
        EmbeddedItem(kind=EmbeddedItemKind.CHUNK, ref="2:0", paper_id=2, text="We build a graph."),
        EmbeddedItem(kind=EmbeddedItemKind.SUMMARY, ref="1", paper_id=1, text="検索を研究する。"),
        EmbeddedItem(kind=EmbeddedItemKind.SUMMARY, ref="2", paper_id=2, text="グラフを作る。"),
        EmbeddedItem(kind=EmbeddedItemKind.CONCEPT, ref="retrieval-augmented-generation", text="RAG"),
        EmbeddedItem(kind=EmbeddedItemKind.CONCEPT, ref="graph-database", text="Graph Database"),
    ),
    vectors=np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    ),
)


def build_paper_concept_edge(paper_id: int, concept_id: str, relation: PaperConceptRelation) -> PaperConceptEdge:
    return PaperConceptEdge(
        paper_id=paper_id,
        concept_id=concept_id,
        relation=relation,
        evidence=(ChunkEvidence(page=1, quote="We study retrieval.", chunk_id=f"{paper_id}:0"),),
    )


GRAPH = KnowledgeGraph(
    paper_ids=(1, 2, 3),
    concepts=(
        Concept(
            id="retrieval-augmented-generation",
            canonical_name="Retrieval-Augmented Generation",
            type=ConceptType.METHOD,
            paper_count=2,
        ),
        Concept(id="graph-database", canonical_name="Graph Database", type=ConceptType.METHOD, paper_count=1),
    ),
    paper_concepts=(
        build_paper_concept_edge(1, "retrieval-augmented-generation", PaperConceptRelation.USES),
        build_paper_concept_edge(2, "retrieval-augmented-generation", PaperConceptRelation.USES),
        build_paper_concept_edge(3, "graph-database", PaperConceptRelation.MENTIONS),
    ),
    concept_relations=(),
)


def run_search(queries: Sequence[str], vectors: Sequence[Sequence[float]], top_k: int) -> tuple[PaperHits, ...]:
    return search_vectors(TABLE, queries, np.array(vectors, dtype=np.float32), GRAPH, top_k=top_k)


def list_refs(paper_hits: PaperHits) -> list[str]:
    return [hit.ref for hit in paper_hits.hits]


def test_the_same_direction_scores_one_and_an_orthogonal_vector_scores_zero() -> None:
    vectors = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    scores = compute_cosine_scores(vectors, np.array([[3.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    assert scores.tolist() == [[1.0, 0.0]]


def test_a_zero_vector_scores_zero_against_everything_without_a_warning() -> None:
    vectors = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    queries = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scores = compute_cosine_scores(vectors, queries)
    assert not np.isnan(scores).any()
    assert scores.tolist() == [[0.0, 1.0], [0.0, 0.0]]


def test_the_scores_have_one_row_per_query_and_one_column_per_item() -> None:
    queries = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    assert compute_cosine_scores(TABLE.vectors, queries).shape == (2, len(TABLE.items))


def test_top_k_limits_the_hits_of_each_query_and_not_the_hits_overall() -> None:
    result = run_search(("検索の質問", "要約の質問"), ([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]), top_k=1)
    assert [(hit.kind, hit.ref, hit.query) for hit in result[0].hits] == [
        (EmbeddedItemKind.CHUNK, "1:0", "検索の質問"),
        (EmbeddedItemKind.SUMMARY, "1", "要約の質問"),
    ]


def test_the_hits_of_one_paper_are_collapsed_into_one_entry_that_keeps_every_hit() -> None:
    result = run_search(("検索の質問", "要約の質問"), ([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]), top_k=1)
    assert [paper_hits.paper_id for paper_hits in result] == [1]
    assert len(result[0].hits) == 2


def test_two_queries_landing_on_the_same_chunk_keep_both_hits() -> None:
    result = run_search(("検索", "retrieval"), ([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]), top_k=1)
    assert list_refs(result[0]) == ["1:0", "1:0"]
    assert [hit.query for hit in result[0].hits] == ["検索", "retrieval"]


def test_a_tie_between_two_items_goes_to_the_one_in_the_lower_row() -> None:
    result = run_search(("要約の質問",), ([0.0, 0.0, 1.0, 0.0],), top_k=1)
    assert [(paper_hits.paper_id, list_refs(paper_hits)) for paper_hits in result] == [(1, ["1"])]


def test_a_concept_hit_reaches_every_paper_that_uses_the_concept() -> None:
    result = run_search(("概念の質問",), ([0.0, 0.0, 0.0, 1.0],), top_k=2)
    assert [(paper_hits.paper_id, list_refs(paper_hits)) for paper_hits in result] == [
        (1, ["retrieval-augmented-generation"]),
        (2, ["retrieval-augmented-generation"]),
    ]
    assert result[0].hits[0].score == pytest.approx(1.0)


def test_a_concept_a_paper_only_mentions_reaches_no_paper() -> None:
    result = run_search(("概念の質問",), ([0.0, 0.0, 0.0, 1.0],), top_k=2)
    assert 3 not in {paper_hits.paper_id for paper_hits in result}
    assert "graph-database" not in {hit.ref for paper_hits in result for hit in paper_hits.hits}


def test_the_papers_come_out_by_their_best_score_descending() -> None:
    result = run_search(("両方の質問",), ([1.0, 1.0, 0.0, 0.0],), top_k=3)
    assert [(paper_hits.paper_id, list_refs(paper_hits)) for paper_hits in result] == [
        (2, ["2:0"]),
        (1, ["1:0", "1:1"]),
    ]


def test_the_papers_with_the_same_best_score_come_out_by_paper_id() -> None:
    result = run_search(("両方の質問", "二つ目の質問"), ([1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]), top_k=2)
    assert [paper_hits.paper_id for paper_hits in result] == [1, 2]
    assert result[0].hits[0].score == pytest.approx(1.0)
    assert result[1].hits[0].score == pytest.approx(1.0)


def test_the_hits_of_one_paper_come_out_by_score_descending() -> None:
    result = run_search(("両方の質問", "二つ目の質問"), ([1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]), top_k=2)
    assert list_refs(result[0]) == ["1:1", "1:0"]
    assert [hit.score for hit in result[0].hits] == pytest.approx([1.0, 0.7071067811865475])


def test_a_top_k_larger_than_the_table_returns_every_item_that_belongs_to_a_paper() -> None:
    result = run_search(("検索の質問",), ([1.0, 0.0, 0.0, 0.0],), top_k=100)
    assert [(paper_hits.paper_id, sorted(list_refs(paper_hits))) for paper_hits in result] == [
        (1, ["1", "1:0", "1:1", "retrieval-augmented-generation"]),
        (2, ["2", "2:0", "retrieval-augmented-generation"]),
    ]


def test_no_query_is_rejected() -> None:
    with pytest.raises(ValueError, match="queries must not be empty"):
        search_vectors(TABLE, (), np.empty((0, 4), dtype=np.float32), GRAPH, top_k=1)


def test_a_row_count_that_does_not_match_the_queries_is_rejected() -> None:
    with pytest.raises(ValueError, match="1 rows for 2 queries"):
        search_vectors(TABLE, ("検索", "retrieval"), np.zeros((1, 4), dtype=np.float32), GRAPH, top_k=1)


def test_a_column_count_that_does_not_match_the_table_is_rejected() -> None:
    with pytest.raises(ValueError, match="3 columns for a table of dimension 4"):
        search_vectors(TABLE, ("検索",), np.zeros((1, 3), dtype=np.float32), GRAPH, top_k=1)


def test_query_vectors_that_are_not_two_dimensional_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be two-dimensional"):
        search_vectors(TABLE, ("検索",), np.zeros(4, dtype=np.float32), GRAPH, top_k=1)


def test_a_top_k_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        search_vectors(TABLE, ("検索",), np.zeros((1, 4), dtype=np.float32), GRAPH, top_k=0)
