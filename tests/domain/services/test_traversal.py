from collections.abc import Sequence

import pytest

from rkgk.domain.models.graph import ChunkEvidence, Concept, ConceptEdge, KnowledgeGraph, PaperConceptEdge
from rkgk.domain.models.search import PaperPaths, SearchConfig, TraversalPath
from rkgk.domain.models.vocabulary import ConceptRelationType, ConceptType, Origin, PaperConceptRelation
from rkgk.domain.services.traversal import (
    ConceptNeighborhoodTraversal,
    TraversalStrategy,
    rank_reached_papers,
    select_graph_candidates,
)

GRAPH_RETRIEVAL = "graph-retrieval"
DENSE_RETRIEVAL = "dense-retrieval"
SPARSE_RETRIEVAL = "sparse-retrieval"
LEXICAL_SEARCH = "lexical-search"
RE_RANKING = "re-ranking"
CROSS_ENCODER = "cross-encoder"
BENCHMARK = "benchmark"
MACHINE_LEARNING = "machine-learning"
TRANSFORMER = "transformer"
ATTENTION = "attention"

RATIONALE = "再ランキングは交差符号化器の代表的な用途であるという一般知識に基づく。"


def build_concept(concept_id: str, concept_type: ConceptType, paper_count: int) -> Concept:
    return Concept(id=concept_id, canonical_name=concept_id, type=concept_type, paper_count=paper_count)


def build_paper_concept_edge(paper_id: int, concept_id: str, relation: PaperConceptRelation) -> PaperConceptEdge:
    return PaperConceptEdge(
        paper_id=paper_id,
        concept_id=concept_id,
        relation=relation,
        evidence=(
            ChunkEvidence(page=1, quote=f"paper {paper_id} {relation.value} {concept_id}", chunk_id=f"{paper_id}:0"),
        ),
    )


def build_paper_concept_relation(
    source_id: str, target_id: str, relation: ConceptRelationType, paper_id: int
) -> ConceptEdge:
    return ConceptEdge(
        source_id=source_id,
        target_id=target_id,
        relation=relation,
        origin=Origin.PAPER,
        paper_id=paper_id,
        evidence=(ChunkEvidence(page=2, quote=f"{source_id} {relation.value} {target_id}", chunk_id=f"{paper_id}:1"),),
    )


# Five papers, so a concept held by two of them sits exactly on the default threshold of 0.4 and one held by
# three of them sits above it.
GRAPH = KnowledgeGraph(
    paper_ids=(1, 2, 3, 4, 5),
    concepts=(
        build_concept(GRAPH_RETRIEVAL, ConceptType.METHOD, paper_count=2),
        build_concept(DENSE_RETRIEVAL, ConceptType.METHOD, paper_count=2),
        build_concept(SPARSE_RETRIEVAL, ConceptType.METHOD, paper_count=2),
        build_concept(LEXICAL_SEARCH, ConceptType.METHOD, paper_count=1),
        build_concept(RE_RANKING, ConceptType.METHOD, paper_count=1),
        build_concept(CROSS_ENCODER, ConceptType.METHOD, paper_count=1),
        build_concept(BENCHMARK, ConceptType.KEYWORD, paper_count=2),
        build_concept(MACHINE_LEARNING, ConceptType.METHOD, paper_count=3),
        build_concept(TRANSFORMER, ConceptType.METHOD, paper_count=2),
        build_concept(ATTENTION, ConceptType.METHOD, paper_count=2),
    ),
    paper_concepts=(
        build_paper_concept_edge(1, GRAPH_RETRIEVAL, PaperConceptRelation.USES),
        build_paper_concept_edge(2, GRAPH_RETRIEVAL, PaperConceptRelation.USES),
        build_paper_concept_edge(1, DENSE_RETRIEVAL, PaperConceptRelation.USES),
        build_paper_concept_edge(4, DENSE_RETRIEVAL, PaperConceptRelation.PROPOSES),
        build_paper_concept_edge(4, DENSE_RETRIEVAL, PaperConceptRelation.USES),
        build_paper_concept_edge(1, SPARSE_RETRIEVAL, PaperConceptRelation.USES),
        build_paper_concept_edge(5, SPARSE_RETRIEVAL, PaperConceptRelation.USES),
        build_paper_concept_edge(3, LEXICAL_SEARCH, PaperConceptRelation.USES),
        build_paper_concept_edge(1, RE_RANKING, PaperConceptRelation.USES),
        build_paper_concept_edge(2, CROSS_ENCODER, PaperConceptRelation.USES),
        build_paper_concept_edge(1, BENCHMARK, PaperConceptRelation.USES),
        build_paper_concept_edge(2, BENCHMARK, PaperConceptRelation.USES),
        build_paper_concept_edge(1, MACHINE_LEARNING, PaperConceptRelation.USES),
        build_paper_concept_edge(2, MACHINE_LEARNING, PaperConceptRelation.USES),
        build_paper_concept_edge(3, MACHINE_LEARNING, PaperConceptRelation.USES),
        build_paper_concept_edge(1, TRANSFORMER, PaperConceptRelation.MENTIONS),
        build_paper_concept_edge(2, TRANSFORMER, PaperConceptRelation.USES),
        build_paper_concept_edge(1, ATTENTION, PaperConceptRelation.USES),
        build_paper_concept_edge(3, ATTENTION, PaperConceptRelation.MENTIONS),
    ),
    concept_relations=(
        build_paper_concept_relation(SPARSE_RETRIEVAL, LEXICAL_SEARCH, ConceptRelationType.IS_A, paper_id=1),
        build_paper_concept_relation(LEXICAL_SEARCH, SPARSE_RETRIEVAL, ConceptRelationType.RELATED_TO, paper_id=3),
        ConceptEdge(
            source_id=RE_RANKING,
            target_id=CROSS_ENCODER,
            relation=ConceptRelationType.RELATED_TO,
            origin=Origin.GENERAL_KNOWLEDGE,
            rationale=RATIONALE,
        ),
    ),
)

TRAVERSAL = ConceptNeighborhoodTraversal()


def traverse_from(source_paper_ids: Sequence[int], **overrides: object) -> tuple[TraversalPath, ...]:
    return TRAVERSAL.traverse(GRAPH, source_paper_ids, SearchConfig.model_validate(overrides))


def describe_path(path: TraversalPath) -> tuple[int, str, tuple[str, ...], int]:
    """One path as (source paper, concept left by, concepts hopped to, paper reached)."""
    return (
        path.source_paper_id,
        path.source_edge.concept_id,
        tuple(hop.reached_concept_id for hop in path.hops),
        path.reached_paper_id,
    )


def describe_paths(paths: Sequence[TraversalPath]) -> list[tuple[int, str, tuple[str, ...], int]]:
    return [describe_path(path) for path in paths]


def find_paths_to(paths: Sequence[TraversalPath], paper_id: int) -> list[TraversalPath]:
    return [path for path in paths if path.reached_paper_id == paper_id]


# Walking the neighbourhood


def test_the_default_walk_leaves_by_every_traversable_concept_of_the_source_paper() -> None:
    assert describe_paths(traverse_from((1,))) == [
        (1, GRAPH_RETRIEVAL, (), 2),
        (1, DENSE_RETRIEVAL, (), 4),
        (1, DENSE_RETRIEVAL, (), 4),
        (1, SPARSE_RETRIEVAL, (), 5),
        (1, SPARSE_RETRIEVAL, (LEXICAL_SEARCH,), 3),
        (1, SPARSE_RETRIEVAL, (LEXICAL_SEARCH,), 3),
        (1, RE_RANKING, (CROSS_ENCODER,), 2),
    ]


def test_a_paper_sharing_a_concept_is_reached_without_a_hop_and_keeps_both_edges_whole() -> None:
    (path,) = [path for path in traverse_from((1,)) if path.source_edge.concept_id == GRAPH_RETRIEVAL]
    assert path.hops == ()
    assert path.reached_paper_id == 2
    assert path.source_edge.relation is PaperConceptRelation.USES
    assert path.target_edge.relation is PaperConceptRelation.USES
    assert path.source_edge.evidence[0].chunk_id == "1:0"
    assert path.target_edge.evidence[0].chunk_id == "2:0"


def test_a_paper_one_concept_relation_away_is_reached_by_following_the_relation_forward() -> None:
    paths = [path for path in traverse_from((1,)) if path.hops and path.hops[0].edge.source_id == SPARSE_RETRIEVAL]
    (path,) = paths
    hop = path.hops[0]
    assert hop.edge.relation is ConceptRelationType.IS_A
    assert hop.edge.target_id == LEXICAL_SEARCH
    assert hop.reached_concept_id == LEXICAL_SEARCH
    assert hop.edge.evidence[0].chunk_id == "1:1"
    assert path.reached_paper_id == 3


def test_a_concept_relation_is_followed_backward_too() -> None:
    paths = [path for path in traverse_from((1,)) if path.hops and path.hops[0].edge.source_id == LEXICAL_SEARCH]
    (path,) = paths
    hop = path.hops[0]
    assert hop.edge.relation is ConceptRelationType.RELATED_TO
    assert hop.edge.target_id == SPARSE_RETRIEVAL
    assert hop.reached_concept_id == LEXICAL_SEARCH
    assert path.reached_paper_id == 3


def test_a_general_knowledge_hop_carries_its_origin_and_rationale() -> None:
    (path,) = [path for path in traverse_from((1,)) if path.source_edge.concept_id == RE_RANKING]
    hop = path.hops[0]
    assert hop.edge.origin is Origin.GENERAL_KNOWLEDGE
    assert hop.edge.rationale == RATIONALE
    assert hop.edge.evidence == ()
    assert path.reached_paper_id == 2


def test_a_paper_that_proposes_and_uses_the_shared_concept_is_reached_by_each_edge() -> None:
    paths = find_paths_to(traverse_from((1,)), 4)
    assert [path.target_edge.relation for path in paths] == [
        PaperConceptRelation.PROPOSES,
        PaperConceptRelation.USES,
    ]


def test_no_path_leads_back_to_the_source_paper() -> None:
    assert all(path.reached_paper_id != 1 for path in traverse_from((1,), max_hops=2))


def test_no_path_stands_on_the_same_concept_twice() -> None:
    """Two concepts point at each other, so a second hop could only come back to where the path started."""
    paths = traverse_from((1,), max_hops=2)
    assert all(hop.reached_concept_id != SPARSE_RETRIEVAL for path in paths for hop in path.hops)
    assert all(path.hops == () for path in find_paths_to(paths, 5))


def test_every_source_paper_is_walked_in_the_given_order_including_paths_to_another_source_paper() -> None:
    paths = traverse_from((2, 1), max_hops=0)
    assert [path.source_paper_id for path in paths] == [2, 1, 1, 1, 1]
    assert (2, GRAPH_RETRIEVAL, (), 1) in describe_paths(paths)
    assert (1, GRAPH_RETRIEVAL, (), 2) in describe_paths(paths)


def test_an_unknown_source_paper_is_rejected() -> None:
    with pytest.raises(ValueError, match="paper 99 is not in the graph"):
        traverse_from((99,))


# What the configuration lets the walk follow


def test_max_hops_of_zero_keeps_only_the_papers_that_share_a_concept() -> None:
    assert describe_paths(traverse_from((1,), max_hops=0)) == [
        (1, GRAPH_RETRIEVAL, (), 2),
        (1, DENSE_RETRIEVAL, (), 4),
        (1, DENSE_RETRIEVAL, (), 4),
        (1, SPARSE_RETRIEVAL, (), 5),
    ]


def test_a_keyword_concept_is_not_walked_by_default() -> None:
    assert all(path.source_edge.concept_id != BENCHMARK for path in traverse_from((1,)))


def test_a_keyword_concept_is_walked_when_the_config_lists_its_type() -> None:
    paths = traverse_from((1,), concept_types=(ConceptType.PROBLEM, ConceptType.METHOD, ConceptType.KEYWORD))
    assert (1, BENCHMARK, (), 2) in describe_paths(paths)


def test_a_concept_held_by_more_papers_than_the_threshold_allows_is_skipped() -> None:
    assert all(path.source_edge.concept_id != MACHINE_LEARNING for path in traverse_from((1,)))


def test_a_concept_exactly_at_the_threshold_is_walked() -> None:
    paths = traverse_from((1,), generic_concept_threshold=0.6)
    assert (1, MACHINE_LEARNING, (), 2) in describe_paths(paths)
    assert (1, MACHINE_LEARNING, (), 3) in describe_paths(paths)


def test_a_generic_concept_is_walked_when_the_threshold_is_raised_to_one() -> None:
    paths = traverse_from((1,), generic_concept_threshold=1.0)
    assert (1, MACHINE_LEARNING, (), 2) in describe_paths(paths)


def test_a_mentions_edge_is_not_left_by_and_not_arrived_at_by_default() -> None:
    paths = traverse_from((1,))
    assert all(path.source_edge.concept_id != TRANSFORMER for path in paths)
    assert all(path.source_edge.concept_id != ATTENTION for path in paths)


def test_a_mentions_edge_is_followed_when_the_config_lists_the_relation() -> None:
    paths = traverse_from(
        (1,),
        paper_relations=(
            PaperConceptRelation.PROPOSES,
            PaperConceptRelation.USES,
            PaperConceptRelation.ADDRESSES,
            PaperConceptRelation.MENTIONS,
        ),
    )
    assert (1, TRANSFORMER, (), 2) in describe_paths(paths)
    assert (1, ATTENTION, (), 3) in describe_paths(paths)


def test_the_walk_satisfies_the_traversal_strategy_protocol() -> None:
    strategy: TraversalStrategy = ConceptNeighborhoodTraversal()
    assert strategy.traverse(GRAPH, (1,), SearchConfig()) == traverse_from((1,))


# Ranking and selecting


def test_rank_reached_papers_orders_by_path_count_then_by_paper_id() -> None:
    ranked = rank_reached_papers(traverse_from((1,)))
    assert [(entry.paper_id, len(entry.paths)) for entry in ranked] == [(2, 2), (3, 2), (4, 2), (5, 1)]


def test_rank_reached_papers_keeps_the_order_of_the_paths_of_one_paper() -> None:
    paths = traverse_from((1,))
    ranked = rank_reached_papers(paths)
    assert [entry.paths for entry in ranked] == [tuple(find_paths_to(paths, entry.paper_id)) for entry in ranked]


def test_rank_reached_papers_returns_nothing_when_no_path_was_found() -> None:
    assert rank_reached_papers(()) == ()


def test_select_graph_candidates_drops_the_papers_vector_search_already_found() -> None:
    ranked = rank_reached_papers(traverse_from((1,)))
    selected = select_graph_candidates(ranked, direct_paper_ids=(1, 2), limit=10)
    assert [entry.paper_id for entry in selected] == [3, 4, 5]


def test_select_graph_candidates_keeps_the_first_entries_up_to_the_limit() -> None:
    ranked = rank_reached_papers(traverse_from((1,)))
    selected = select_graph_candidates(ranked, direct_paper_ids=(2,), limit=2)
    assert [entry.paper_id for entry in selected] == [3, 4]


def test_select_graph_candidates_with_a_limit_of_zero_keeps_nothing() -> None:
    ranked = rank_reached_papers(traverse_from((1,)))
    assert select_graph_candidates(ranked, direct_paper_ids=(), limit=0) == ()


def test_select_graph_candidates_returns_paper_paths_that_can_become_candidates() -> None:
    ranked = rank_reached_papers(traverse_from((1,)))
    (first,) = select_graph_candidates(ranked, direct_paper_ids=(2, 3, 4), limit=10)
    assert isinstance(first, PaperPaths)
    assert first.paper_id == 5
