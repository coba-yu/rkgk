"""Finds the papers a set of queries lands on, by comparing the query vectors with the embedding table.

A query hits an item, not a paper, so the hits are gathered per paper here: a chunk or a summary belongs to the
paper it was written from, and a concept belongs to every paper the graph attaches it to.
Every hit is kept as it was produced, because a hit is the evidence a later stage shows for the paper it chose.
It reads the embedding and graph models and owns no data.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from rkgk.domain.models.embedding import EmbeddedItem, EmbeddedItemKind, EmbeddingTable
from rkgk.domain.models.graph import KnowledgeGraph
from rkgk.domain.models.search import PaperHits, SearchHit
from rkgk.domain.models.vocabulary import traversable_paper_relations


def normalize_rows(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Divide each row by its L2 norm, leaving a zero row as it is.

    A zero row has no direction, so it keeps its zeros and scores 0 against everything; dividing it would give
    a NaN that then beats every real score in a sort.
    """
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def compute_cosine_scores(vectors: NDArray[np.float32], query_vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return the cosine of every query against every item, one row per query and one column per item.

    Both sides are normalized first, so the dot product of a query row and an item row is the cosine itself.
    """
    return normalize_rows(query_vectors) @ normalize_rows(vectors).T


def search_vectors(
    table: EmbeddingTable,
    queries: Sequence[str],
    query_vectors: NDArray[np.float32],
    graph: KnowledgeGraph,
    top_k: int,
) -> tuple[PaperHits, ...]:
    """Take the `top_k` items of each query, turn them into hits, and group the hits by paper.

    Each query is ranked on its own and the results are unioned, so an item two queries land on yields two hits,
    each with its own query and score.
    """
    check_search_inputs(table, queries, query_vectors, top_k)
    scores = compute_cosine_scores(np.asarray(table.vectors, dtype=np.float32), query_vectors)
    papers_by_concept = map_concepts_to_papers(graph)

    hits_by_paper: dict[int, list[SearchHit]] = {}
    for query, row in zip(queries, scores, strict=True):
        for index in rank_top_rows(row, top_k):
            item = table.items[index]
            hit = SearchHit(kind=item.kind, ref=item.ref, query=query, score=float(row[index]), text=item.text)
            for paper_id in find_papers_of_item(item, papers_by_concept):
                hits_by_paper.setdefault(paper_id, []).append(hit)

    grouped: list[PaperHits] = []
    for paper_id, hits in hits_by_paper.items():
        ranked = sorted(hits, key=lambda hit: -hit.score)
        grouped.append(PaperHits(paper_id=paper_id, hits=tuple(ranked)))
    return tuple(sorted(grouped, key=lambda entry: (-max(hit.score for hit in entry.hits), entry.paper_id)))


def check_search_inputs(
    table: EmbeddingTable, queries: Sequence[str], query_vectors: NDArray[np.float32], top_k: int
) -> None:
    """Reject the shapes that would otherwise pair a query with the wrong row or the wrong table."""
    if not queries:
        raise ValueError("queries must not be empty")
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    if query_vectors.ndim != 2:
        raise ValueError(f"query_vectors must be two-dimensional, got {query_vectors.ndim} dimensions")
    if query_vectors.shape[0] != len(queries):
        raise ValueError(f"query_vectors has {query_vectors.shape[0]} rows for {len(queries)} queries")
    if query_vectors.shape[1] != table.dimension:
        raise ValueError(
            f"query_vectors has {query_vectors.shape[1]} columns for a table of dimension {table.dimension}"
        )


def rank_top_rows(scores: NDArray[np.float32], top_k: int) -> tuple[int, ...]:
    """Return the rows of the `top_k` highest scores, best first, a tie going to the lower row.

    The sort is on the negated scores and stable, so equal scores stay in row order and the same table always
    answers the same query the same way.
    """
    return tuple(int(index) for index in np.argsort(-scores, kind="stable")[:top_k])


def map_concepts_to_papers(graph: KnowledgeGraph) -> dict[str, tuple[int, ...]]:
    """Group the papers of the traversable paper-concept edges by concept, each paper listed once.

    `mentions` is left out because a passing reference does not make the paper about the concept, so a hit on
    the concept is no evidence for it.
    A paper that holds the concept through two relations is still one paper, hence the duplicate check.
    """
    traversable = traversable_paper_relations()
    papers_by_concept: dict[str, list[int]] = {}
    for edge in graph.paper_concepts:
        if edge.relation not in traversable:
            continue
        papers = papers_by_concept.setdefault(edge.concept_id, [])
        if edge.paper_id not in papers:
            papers.append(edge.paper_id)
    return {concept_id: tuple(papers) for concept_id, papers in papers_by_concept.items()}


def find_papers_of_item(item: EmbeddedItem, papers_by_concept: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
    """Return the papers a hit on the item belongs to: its own paper, or every paper the graph attaches it to.

    A concept no paper holds through a traversable relation belongs to no paper, and its hit is dropped.
    """
    if item.kind is EmbeddedItemKind.CONCEPT:
        return papers_by_concept.get(item.ref, ())
    # `EmbeddedItem` requires a paper on a chunk and on a summary, so the guard only satisfies the type checker.
    return () if item.paper_id is None else (item.paper_id,)
