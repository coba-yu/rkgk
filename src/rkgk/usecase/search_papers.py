"""Use case that answers one set of themes with the papers to read next, and with why each one was chosen.

Vector search and graph traversal each know one half of the answer and neither knows the papers, so three jobs
are left here.
The embedding model is checked against the manifest first, because a query embedded with another model than the
vectors is compared against numbers it does not belong with and the result would look plausible while being
meaningless.
A path that reaches a paper vector search already found is moved onto that paper's direct candidate, because a
paper is listed once in the result and traversal has no way of knowing what vector search found.
The titles, the S3 URIs and the Japanese summaries are joined in last, because they live in the paper and
extraction artifacts rather than in the index.
Only the paper metadata is read for that join, never the pages, because a result lists many candidates and
reading every page of each one would be corpus-wide I/O for details the result never shows.
"""

from collections.abc import Sequence

from rkgk.domain.embedders import Embedder
from rkgk.domain.models.manifest import EmbeddingModelMismatchError
from rkgk.domain.models.search import EmbeddedItemHit, PaperCandidate, SearchConfig, SearchResult, TraversalPath
from rkgk.domain.repositories.index import IndexRepository
from rkgk.domain.repositories.paper import PaperRepository
from rkgk.domain.repositories.paper_extraction import PaperExtractionRepository
from rkgk.domain.services.traversal import (
    ConceptNeighborhoodTraversal,
    TraversalStrategy,
    rank_reached_papers,
    select_graph_candidates,
)
from rkgk.domain.services.vector_search import search_vectors


class SearchPapersUseCase:
    def __init__(
        self,
        paper_repository: PaperRepository,
        extraction_repository: PaperExtractionRepository,
        index_repository: IndexRepository,
        embedder: Embedder,
        traversal: TraversalStrategy | None = None,
    ) -> None:
        self._paper_repository = paper_repository
        self._extraction_repository = extraction_repository
        self._index_repository = index_repository
        self._embedder = embedder
        # The concept neighbourhood is the only strategy implemented so far, so it is the default rather than
        # something every caller has to name.
        self._traversal = ConceptNeighborhoodTraversal() if traversal is None else traversal

    def execute(self, queries: Sequence[str], config: SearchConfig) -> SearchResult:
        """Find the papers the queries land on, walk the graph around them, and describe both from the artifacts."""
        self._check_queries(queries)
        index = self._index_repository.find_index()
        # Before embedding, not after: loading a model and encoding the queries costs seconds that a caller who
        # pointed at the wrong index should not pay.
        if self._embedder.model_name != index.manifest.embedding_model:
            raise EmbeddingModelMismatchError(index.manifest.embedding_model, self._embedder.model_name)

        query_vectors = self._embedder.embed_queries(queries)
        paper_hits = search_vectors(index.embeddings, queries, query_vectors, index.graph, config.top_k)
        direct_ids = [entry.paper_id for entry in paper_hits]

        paths = self._traversal.traverse(index.graph, direct_ids, config)
        ranked = rank_reached_papers(paths)
        graph_entries = select_graph_candidates(ranked, direct_ids, config.max_graph_candidates)
        found_directly = set(direct_ids)
        paths_by_direct_paper = {entry.paper_id: entry.paths for entry in ranked if entry.paper_id in found_directly}

        return SearchResult(
            queries=tuple(queries),
            config=config,
            direct_candidates=tuple(
                self._build_candidate(entry.paper_id, entry.hits, paths_by_direct_paper.get(entry.paper_id, ()))
                for entry in paper_hits
            ),
            graph_candidates=tuple(self._build_candidate(entry.paper_id, (), entry.paths) for entry in graph_entries),
        )

    @staticmethod
    def _check_queries(queries: Sequence[str]) -> None:
        """Reject the query lists `SearchResult` would reject, before an index is read or a vector is made.

        `SearchResult` is built last, so leaving the check to it would report a typo only after the whole search
        ran; the messages are the same so that a caller sees one wording either way.
        """
        if not queries:
            raise ValueError("queries must not be empty")
        seen: set[str] = set()
        for query in queries:
            if not query.strip():
                raise ValueError("queries must contain non-whitespace characters")
            if query in seen:
                raise ValueError(f"queries repeats {query!r}")
            seen.add(query)

    def _build_candidate(
        self, paper_id: int, hits: tuple[EmbeddedItemHit, ...], paths: tuple[TraversalPath, ...]
    ) -> PaperCandidate:
        """Join what led to the paper with what the paper is, reading each artifact of it once.

        `find_meta` rather than `find`: only the title and the S3 URI are needed here, and this runs once per
        candidate, so reading every page of every candidate would make a result list cost corpus-wide I/O and
        would fail the whole search over a page missing for a paper the search never needed to open.
        """
        meta = self._paper_repository.find_meta(paper_id)
        extraction = self._extraction_repository.find(paper_id)
        return PaperCandidate(
            paper_id=paper_id,
            title=meta.title,
            s3_uri=meta.s3_uri,
            summary_ja=extraction.summary_ja,
            hits=hits,
            paths=paths,
        )
