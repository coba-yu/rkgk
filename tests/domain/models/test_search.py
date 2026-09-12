import pytest
from pydantic import ValidationError

from rkgk.domain.models.embedding import EmbeddedItemKind
from rkgk.domain.models.graph import ChunkEvidence, ConceptEdge, PaperConceptEdge
from rkgk.domain.models.search import (
    ConceptHop,
    EmbeddedItemHit,
    PaperCandidate,
    PaperHits,
    SearchConfig,
    SearchResult,
    TraversalPath,
)
from rkgk.domain.models.vocabulary import ConceptRelationType, Origin, PaperConceptRelation

QUERY_RETRIEVAL = "検索手法についての論文を探したい"
QUERY_DENSE = "密なベクトル検索の手法を知りたい"

RETRIEVAL = "retrieval"
DENSE_RETRIEVAL = "dense-retrieval"
SPARSE_RETRIEVAL = "sparse-retrieval"


def build_config(**overrides: object) -> SearchConfig:
    payload: dict[str, object] = {}
    payload.update(overrides)
    return SearchConfig.model_validate(payload)


def build_hit(**overrides: object) -> EmbeddedItemHit:
    payload: dict[str, object] = {
        "kind": EmbeddedItemKind.CHUNK,
        "ref": "1:0",
        "query": QUERY_RETRIEVAL,
        "score": 0.9,
        "text": "Retrieval improves recall by finding relevant passages.",
    }
    payload.update(overrides)
    return EmbeddedItemHit.model_validate(payload)


def build_paper_hits(**overrides: object) -> PaperHits:
    payload: dict[str, object] = {
        "paper_id": 1,
        "hits": (build_hit(),),
    }
    payload.update(overrides)
    return PaperHits.model_validate(payload)


def build_chunk_evidence(**overrides: object) -> ChunkEvidence:
    payload: dict[str, object] = {
        "page": 1,
        "quote": "Retrieval improves recall by finding relevant passages.",
        "chunk_id": "1:0",
    }
    payload.update(overrides)
    return ChunkEvidence.model_validate(payload)


def build_paper_edge(paper_id: int, concept_id: str = RETRIEVAL, **overrides: object) -> PaperConceptEdge:
    """A paper-to-concept edge for `paper_id`, with a chunk id that matches the paper by convention."""
    payload: dict[str, object] = {
        "paper_id": paper_id,
        "concept_id": concept_id,
        "relation": PaperConceptRelation.USES,
        "evidence": (build_chunk_evidence(chunk_id=f"{paper_id}:0"),),
    }
    payload.update(overrides)
    return PaperConceptEdge.model_validate(payload)


def build_paper_origin_concept_edge(**overrides: object) -> ConceptEdge:
    """ "dense-retrieval" is_a "retrieval", quoted from paper 1."""
    payload: dict[str, object] = {
        "source_id": DENSE_RETRIEVAL,
        "target_id": RETRIEVAL,
        "relation": ConceptRelationType.IS_A,
        "origin": Origin.PAPER,
        "paper_id": 1,
        "evidence": (build_chunk_evidence(chunk_id="1:0"),),
    }
    payload.update(overrides)
    return ConceptEdge.model_validate(payload)


def build_general_knowledge_concept_edge(**overrides: object) -> ConceptEdge:
    """ "sparse-retrieval" is_a "retrieval", added from general knowledge, no paper behind it."""
    payload: dict[str, object] = {
        "source_id": SPARSE_RETRIEVAL,
        "target_id": RETRIEVAL,
        "relation": ConceptRelationType.IS_A,
        "origin": Origin.GENERAL_KNOWLEDGE,
        "rationale": "疎な検索も密な検索と同様、検索という上位概念の一種であるという一般知識に基づく。",
    }
    payload.update(overrides)
    return ConceptEdge.model_validate(payload)


def build_concept_hop(**overrides: object) -> ConceptHop:
    payload: dict[str, object] = {
        "edge": build_paper_origin_concept_edge(),
        "reached_concept_id": RETRIEVAL,
    }
    payload.update(overrides)
    return ConceptHop.model_validate(payload)


def build_traversal_path(**overrides: object) -> TraversalPath:
    payload: dict[str, object] = {
        "source_edge": build_paper_edge(1, concept_id=RETRIEVAL),
        "hops": (),
        "target_edge": build_paper_edge(2, concept_id=RETRIEVAL),
    }
    payload.update(overrides)
    return TraversalPath.model_validate(payload)


def build_paper_candidate(**overrides: object) -> PaperCandidate:
    payload: dict[str, object] = {
        "paper_id": 1,
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "s3_uri": "s3://rkgk-papers/1.pdf",
        "summary_ja": "疎な検索ではなく密なベクトル表現を用いた検索手法を提案する。",
        "hits": (build_hit(),),
        "paths": (),
    }
    payload.update(overrides)
    return PaperCandidate.model_validate(payload)


def build_search_result(**overrides: object) -> SearchResult:
    payload: dict[str, object] = {
        "queries": (QUERY_RETRIEVAL,),
        "config": build_config(),
        "direct_candidates": (build_paper_candidate(),),
        "graph_candidates": (),
    }
    payload.update(overrides)
    return SearchResult.model_validate(payload)


def build_full_result() -> SearchResult:
    """A paper found directly, a paper found directly and reached again from it, and a paper reached from that."""
    hit_1 = build_hit(kind=EmbeddedItemKind.CHUNK, ref="1:0", query=QUERY_RETRIEVAL)
    hit_2 = build_hit(kind=EmbeddedItemKind.SUMMARY, ref="2", query=QUERY_DENSE)
    hop = build_concept_hop(
        edge=build_paper_origin_concept_edge(source_id=RETRIEVAL, target_id=DENSE_RETRIEVAL),
        reached_concept_id=DENSE_RETRIEVAL,
    )
    path_from_1 = build_traversal_path(
        source_edge=build_paper_edge(1, concept_id=RETRIEVAL),
        hops=(hop,),
        target_edge=build_paper_edge(2, concept_id=DENSE_RETRIEVAL),
    )
    path_from_2 = build_traversal_path(
        source_edge=build_paper_edge(2, concept_id=DENSE_RETRIEVAL),
        target_edge=build_paper_edge(3, concept_id=DENSE_RETRIEVAL),
    )
    candidate_1 = build_paper_candidate(paper_id=1, hits=(hit_1,), paths=())
    candidate_2 = build_paper_candidate(paper_id=2, hits=(hit_2,), paths=(path_from_1,))
    candidate_3 = build_paper_candidate(paper_id=3, hits=(), paths=(path_from_2,))
    return build_search_result(
        queries=(QUERY_RETRIEVAL, QUERY_DENSE),
        direct_candidates=(candidate_1, candidate_2),
        graph_candidates=(candidate_3,),
    )


# SearchConfig


def test_search_config_defaults_top_k_to_ten() -> None:
    assert build_config().top_k == 10


def test_search_config_rejects_a_non_positive_top_k() -> None:
    with pytest.raises(ValidationError):
        build_config(top_k=0)


# EmbeddedItemHit


def test_a_hit_builds_with_its_score_and_text() -> None:
    hit = build_hit()
    assert hit.score == 0.9
    assert hit.text.startswith("Retrieval")


def test_hit_rejects_blank_text() -> None:
    with pytest.raises(ValidationError, match="text must contain non-whitespace characters"):
        build_hit(text="   \n ")


def test_hit_rejects_blank_query() -> None:
    with pytest.raises(ValidationError):
        build_hit(query="")


def test_hit_rejects_blank_ref() -> None:
    with pytest.raises(ValidationError):
        build_hit(ref="")


# PaperHits


def test_paper_hits_collects_the_hits_of_one_paper() -> None:
    hits = build_paper_hits()
    assert hits.paper_id == 1
    assert len(hits.hits) == 1


def test_paper_hits_requires_at_least_one_hit() -> None:
    with pytest.raises(ValidationError):
        build_paper_hits(hits=())


def test_paper_hits_rejects_the_same_kind_ref_and_query_twice() -> None:
    hit = build_hit()
    with pytest.raises(ValidationError, match="hits repeats chunk '1:0' for the query"):
        build_paper_hits(hits=(hit, hit))


def test_paper_hits_accepts_the_same_item_hit_by_two_different_queries() -> None:
    hit_a = build_hit(query=QUERY_RETRIEVAL)
    hit_b = build_hit(query=QUERY_DENSE)
    hits = build_paper_hits(hits=(hit_a, hit_b))
    assert hits.hits == (hit_a, hit_b)


# ConceptHop


def test_concept_hop_reaches_the_target_end_of_the_edge() -> None:
    hop = build_concept_hop(reached_concept_id=RETRIEVAL)
    assert hop.reached_concept_id == RETRIEVAL


def test_concept_hop_reaches_the_source_end_of_the_edge() -> None:
    hop = build_concept_hop(reached_concept_id=DENSE_RETRIEVAL)
    assert hop.reached_concept_id == DENSE_RETRIEVAL


def test_concept_hop_rejects_a_reached_concept_outside_the_edge() -> None:
    with pytest.raises(ValidationError, match="reached_concept_id must be"):
        build_concept_hop(reached_concept_id=SPARSE_RETRIEVAL)


def test_concept_hop_departed_concept_is_the_source_when_reached_is_the_target() -> None:
    edge = build_paper_origin_concept_edge(source_id=RETRIEVAL, target_id=DENSE_RETRIEVAL)
    hop = build_concept_hop(edge=edge, reached_concept_id=DENSE_RETRIEVAL)
    assert hop.departed_concept_id == RETRIEVAL


def test_concept_hop_departed_concept_is_the_target_when_reached_is_the_source() -> None:
    edge = build_paper_origin_concept_edge(source_id=RETRIEVAL, target_id=DENSE_RETRIEVAL)
    hop = build_concept_hop(edge=edge, reached_concept_id=RETRIEVAL)
    assert hop.departed_concept_id == DENSE_RETRIEVAL


def test_concept_hop_accepts_a_general_knowledge_edge_and_keeps_its_rationale() -> None:
    edge = build_general_knowledge_concept_edge()
    hop = build_concept_hop(edge=edge, reached_concept_id=edge.target_id)
    assert hop.edge.rationale == edge.rationale


# TraversalPath


def test_traversal_path_without_hops_is_reached_at_the_shared_concept() -> None:
    path = build_traversal_path()
    assert path.reached_concept_id == RETRIEVAL
    assert path.source_paper_id == 1
    assert path.reached_paper_id == 2


def test_traversal_path_accepts_a_hop_that_follows_the_edge_forward() -> None:
    hop = build_concept_hop(
        edge=build_paper_origin_concept_edge(source_id=RETRIEVAL, target_id=DENSE_RETRIEVAL),
        reached_concept_id=DENSE_RETRIEVAL,
    )
    path = build_traversal_path(
        source_edge=build_paper_edge(1, concept_id=RETRIEVAL),
        hops=(hop,),
        target_edge=build_paper_edge(2, concept_id=DENSE_RETRIEVAL),
    )
    assert path.reached_concept_id == DENSE_RETRIEVAL


def test_traversal_path_accepts_a_hop_that_follows_the_edge_backward() -> None:
    hop = build_concept_hop(
        edge=build_paper_origin_concept_edge(source_id=DENSE_RETRIEVAL, target_id=RETRIEVAL),
        reached_concept_id=DENSE_RETRIEVAL,
    )
    path = build_traversal_path(
        source_edge=build_paper_edge(1, concept_id=RETRIEVAL),
        hops=(hop,),
        target_edge=build_paper_edge(2, concept_id=DENSE_RETRIEVAL),
    )
    assert path.reached_concept_id == DENSE_RETRIEVAL


def test_traversal_path_rejects_a_hop_that_departs_from_another_concept() -> None:
    hop = build_concept_hop(
        edge=build_paper_origin_concept_edge(source_id=DENSE_RETRIEVAL, target_id=SPARSE_RETRIEVAL),
        reached_concept_id=SPARSE_RETRIEVAL,
    )
    with pytest.raises(ValidationError, match="stands on 'retrieval'"):
        build_traversal_path(
            source_edge=build_paper_edge(1, concept_id=RETRIEVAL),
            hops=(hop,),
            target_edge=build_paper_edge(2, concept_id=SPARSE_RETRIEVAL),
        )


def test_traversal_path_rejects_a_second_hop_that_does_not_depart_from_the_first_hops_end() -> None:
    hop_1 = build_concept_hop(
        edge=build_paper_origin_concept_edge(source_id=RETRIEVAL, target_id=DENSE_RETRIEVAL),
        reached_concept_id=DENSE_RETRIEVAL,
    )
    hop_2 = build_concept_hop(
        edge=build_paper_origin_concept_edge(
            source_id=RETRIEVAL, target_id=SPARSE_RETRIEVAL, evidence=(build_chunk_evidence(chunk_id="1:1"),)
        ),
        reached_concept_id=SPARSE_RETRIEVAL,
    )
    with pytest.raises(ValidationError, match="stands on 'dense-retrieval'"):
        build_traversal_path(
            source_edge=build_paper_edge(1, concept_id=RETRIEVAL),
            hops=(hop_1, hop_2),
            target_edge=build_paper_edge(2, concept_id=SPARSE_RETRIEVAL),
        )


def test_traversal_path_rejects_a_target_edge_attached_to_the_wrong_concept() -> None:
    hop = build_concept_hop(
        edge=build_paper_origin_concept_edge(source_id=RETRIEVAL, target_id=DENSE_RETRIEVAL),
        reached_concept_id=DENSE_RETRIEVAL,
    )
    with pytest.raises(ValidationError, match="target_edge is attached to"):
        build_traversal_path(
            source_edge=build_paper_edge(1, concept_id=RETRIEVAL),
            hops=(hop,),
            target_edge=build_paper_edge(2, concept_id=SPARSE_RETRIEVAL),
        )


def test_traversal_path_rejects_leading_from_a_paper_back_to_itself() -> None:
    with pytest.raises(ValidationError, match="back to itself"):
        build_traversal_path(
            source_edge=build_paper_edge(1, concept_id=RETRIEVAL),
            target_edge=build_paper_edge(1, concept_id=RETRIEVAL),
        )


def test_traversal_path_chains_two_hops_to_the_last_hops_end() -> None:
    hop_1 = build_concept_hop(
        edge=build_paper_origin_concept_edge(source_id=RETRIEVAL, target_id=DENSE_RETRIEVAL),
        reached_concept_id=DENSE_RETRIEVAL,
    )
    hop_2 = build_concept_hop(
        edge=build_paper_origin_concept_edge(
            source_id=DENSE_RETRIEVAL, target_id=SPARSE_RETRIEVAL, evidence=(build_chunk_evidence(chunk_id="1:1"),)
        ),
        reached_concept_id=SPARSE_RETRIEVAL,
    )
    path = build_traversal_path(
        source_edge=build_paper_edge(1, concept_id=RETRIEVAL),
        hops=(hop_1, hop_2),
        target_edge=build_paper_edge(2, concept_id=SPARSE_RETRIEVAL),
    )
    assert path.reached_concept_id == SPARSE_RETRIEVAL


def test_two_paths_that_differ_only_in_the_hop_edges_paper_are_different_paths() -> None:
    """Parallel edges asserted by different papers must stay apart, not collapse into one path."""
    edge_from_paper_1 = build_paper_origin_concept_edge(paper_id=1, evidence=(build_chunk_evidence(chunk_id="1:0"),))
    edge_from_paper_3 = build_paper_origin_concept_edge(paper_id=3, evidence=(build_chunk_evidence(chunk_id="3:0"),))
    path_1 = build_traversal_path(
        source_edge=build_paper_edge(1, concept_id=DENSE_RETRIEVAL),
        hops=(build_concept_hop(edge=edge_from_paper_1, reached_concept_id=RETRIEVAL),),
        target_edge=build_paper_edge(2, concept_id=RETRIEVAL),
    )
    path_2 = build_traversal_path(
        source_edge=build_paper_edge(1, concept_id=DENSE_RETRIEVAL),
        hops=(build_concept_hop(edge=edge_from_paper_3, reached_concept_id=RETRIEVAL),),
        target_edge=build_paper_edge(2, concept_id=RETRIEVAL),
    )
    assert path_1 != path_2
    candidate = build_paper_candidate(paper_id=2, hits=(), paths=(path_1, path_2))
    assert candidate.paths == (path_1, path_2)


# PaperCandidate


def test_paper_candidate_accepts_hits_only() -> None:
    candidate = build_paper_candidate(hits=(build_hit(),), paths=())
    assert candidate.paths == ()


def test_paper_candidate_accepts_paths_only() -> None:
    path = build_traversal_path(
        source_edge=build_paper_edge(2, concept_id=RETRIEVAL),
        target_edge=build_paper_edge(1, concept_id=RETRIEVAL),
    )
    candidate = build_paper_candidate(paper_id=1, hits=(), paths=(path,))
    assert candidate.hits == ()


def test_paper_candidate_accepts_hits_and_paths() -> None:
    path = build_traversal_path(
        source_edge=build_paper_edge(2, concept_id=RETRIEVAL),
        target_edge=build_paper_edge(1, concept_id=RETRIEVAL),
    )
    candidate = build_paper_candidate(paper_id=1, hits=(build_hit(),), paths=(path,))
    assert candidate.hits and candidate.paths


def test_paper_candidate_rejects_neither_hits_nor_paths() -> None:
    with pytest.raises(ValidationError, match="a candidate needs at least one hit or one path"):
        build_paper_candidate(hits=(), paths=())


def test_paper_candidate_rejects_a_repeated_hit() -> None:
    hit = build_hit()
    with pytest.raises(ValidationError, match="hits repeats chunk '1:0' for the query"):
        build_paper_candidate(hits=(hit, hit))


def test_paper_candidate_rejects_a_path_that_does_not_end_at_it() -> None:
    path = build_traversal_path()  # source paper 1, reaches paper 2
    with pytest.raises(ValidationError, match="paths must end at the candidate paper 1, not 2"):
        build_paper_candidate(paper_id=1, hits=(), paths=(path,))


def test_paper_candidate_rejects_the_same_path_twice() -> None:
    path = build_traversal_path()  # source paper 1, reaches paper 2
    with pytest.raises(ValidationError, match="paths repeats a path from paper 1"):
        build_paper_candidate(paper_id=2, hits=(), paths=(path, path))


def test_paper_candidate_rejects_blank_title() -> None:
    with pytest.raises(ValidationError):
        build_paper_candidate(title="")


def test_paper_candidate_rejects_blank_s3_uri() -> None:
    with pytest.raises(ValidationError):
        build_paper_candidate(s3_uri="")


def test_paper_candidate_rejects_blank_summary_ja() -> None:
    with pytest.raises(ValidationError):
        build_paper_candidate(summary_ja="")


# SearchResult


def test_search_result_accepts_direct_and_graph_candidates_with_hits_and_paths() -> None:
    result = build_full_result()
    assert [candidate.paper_id for candidate in result.direct_candidates] == [1, 2]
    assert [candidate.paper_id for candidate in result.graph_candidates] == [3]


def test_search_result_rejects_empty_queries() -> None:
    with pytest.raises(ValidationError):
        build_search_result(queries=())


def test_search_result_rejects_a_blank_query() -> None:
    with pytest.raises(ValidationError, match="queries must contain non-whitespace characters"):
        build_search_result(queries=("   ",))


def test_search_result_rejects_a_repeated_query() -> None:
    with pytest.raises(ValidationError, match="queries repeats"):
        build_search_result(queries=(QUERY_RETRIEVAL, QUERY_RETRIEVAL))


def test_search_result_rejects_the_same_paper_in_both_lists() -> None:
    direct = build_paper_candidate(paper_id=1, hits=(build_hit(),), paths=())
    path = build_traversal_path(
        source_edge=build_paper_edge(2, concept_id=RETRIEVAL),
        target_edge=build_paper_edge(1, concept_id=RETRIEVAL),
    )
    graph = build_paper_candidate(paper_id=1, hits=(), paths=(path,))
    with pytest.raises(ValidationError, match="paper 1 is listed more than once"):
        build_search_result(direct_candidates=(direct,), graph_candidates=(graph,))


def test_search_result_rejects_the_same_paper_twice_in_one_list() -> None:
    candidate = build_paper_candidate(paper_id=1, hits=(build_hit(),), paths=())
    with pytest.raises(ValidationError, match="paper 1 is listed more than once"):
        build_search_result(direct_candidates=(candidate, candidate), graph_candidates=())


def test_search_result_rejects_a_direct_candidate_without_hits() -> None:
    path = build_traversal_path(
        source_edge=build_paper_edge(2, concept_id=RETRIEVAL),
        target_edge=build_paper_edge(1, concept_id=RETRIEVAL),
    )
    candidate = build_paper_candidate(paper_id=1, hits=(), paths=(path,))
    with pytest.raises(ValidationError, match="direct candidate 1 has no hit"):
        build_search_result(direct_candidates=(candidate,), graph_candidates=())


def test_search_result_rejects_a_graph_candidate_with_hits() -> None:
    direct = build_paper_candidate(paper_id=1, hits=(build_hit(),), paths=())
    path = build_traversal_path(
        source_edge=build_paper_edge(1, concept_id=RETRIEVAL),
        target_edge=build_paper_edge(2, concept_id=RETRIEVAL),
    )
    graph = build_paper_candidate(paper_id=2, hits=(build_hit(ref="2:0"),), paths=(path,))
    with pytest.raises(ValidationError, match="graph candidate 2 has hits and belongs to direct_candidates"):
        build_search_result(direct_candidates=(direct,), graph_candidates=(graph,))


def test_search_result_rejects_a_hit_whose_query_is_unknown() -> None:
    direct = build_paper_candidate(paper_id=1, hits=(build_hit(query="未知のクエリ"),), paths=())
    with pytest.raises(ValidationError, match="has a hit for the unknown query"):
        build_search_result(queries=(QUERY_RETRIEVAL,), direct_candidates=(direct,), graph_candidates=())


def test_search_result_rejects_a_path_from_a_non_direct_candidate() -> None:
    direct = build_paper_candidate(paper_id=1, hits=(build_hit(),), paths=())
    path = build_traversal_path(
        source_edge=build_paper_edge(99, concept_id=RETRIEVAL),
        target_edge=build_paper_edge(2, concept_id=RETRIEVAL),
    )
    graph = build_paper_candidate(paper_id=2, hits=(), paths=(path,))
    with pytest.raises(ValidationError, match="which is not a direct candidate"):
        build_search_result(direct_candidates=(direct,), graph_candidates=(graph,))


def test_search_result_round_trips_through_json() -> None:
    result = build_full_result()
    assert SearchResult.model_validate_json(result.model_dump_json()) == result
