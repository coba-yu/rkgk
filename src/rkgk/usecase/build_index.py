"""Use case that turns the stored papers, extractions and normalization into the index search reads.

No agent runs here: every artifact this needs has already been written by the extract and normalize commands, so
the build only chunks, resolves, embeds and joins them.
"""

from collections.abc import Mapping, Sequence

from rkgk.domain import DOMAIN_MODEL_VERSION
from rkgk.domain.embedders import Embedder
from rkgk.domain.models.chunk import Chunk
from rkgk.domain.models.concept_normalization import ConceptNormalizationValidationError, MissingPaperExtractionsError
from rkgk.domain.models.embedding import EmbeddingTable
from rkgk.domain.models.graph import ChunkEvidence, EvidenceResolutionError, UnresolvedEvidence
from rkgk.domain.models.manifest import INDEX_SCHEMA_VERSION, IndexBuildRun, IndexManifest
from rkgk.domain.models.paper import Paper
from rkgk.domain.models.paper_extraction import (
    PageEvidence,
    PaperExtraction,
    PaperExtractionIssue,
    PaperExtractionMismatchError,
)
from rkgk.domain.repositories.concept_normalization import ConceptNormalizationRepository
from rkgk.domain.repositories.index import IndexRepository
from rkgk.domain.repositories.paper import PaperRepository
from rkgk.domain.repositories.paper_extraction import PaperExtractionNotFoundError, PaperExtractionRepository
from rkgk.domain.services.chunking import DEFAULT_MAX_TOKENS, chunk_paper
from rkgk.domain.services.concept_normalization import check_normalization_against_extractions
from rkgk.domain.services.embedding_items import build_embedding_items
from rkgk.domain.services.evidence_resolver import resolve_extraction_evidence
from rkgk.domain.services.graph_builder import build_knowledge_graph
from rkgk.domain.services.paper_extraction import check_extraction_against_paper
from rkgk.domain.tokenizers import Tokenizer


class BuildIndexUseCase:
    def __init__(
        self,
        paper_repository: PaperRepository,
        extraction_repository: PaperExtractionRepository,
        normalization_repository: ConceptNormalizationRepository,
        tokenizer: Tokenizer,
        embedder: Embedder,
        index_repository: IndexRepository,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._paper_repository = paper_repository
        self._extraction_repository = extraction_repository
        self._normalization_repository = normalization_repository
        self._tokenizer = tokenizer
        self._embedder = embedder
        self._index_repository = index_repository
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be at least 1, got {max_tokens}")
        self._max_tokens = max_tokens

    def execute(self) -> IndexBuildRun:
        """Build the whole index at once, checking every stored artifact before anything is embedded or written."""
        paper_ids = tuple(entry.id for entry in self._paper_repository.find_index())
        if not paper_ids:
            raise ValueError("no papers in the index")
        papers = tuple(self._paper_repository.find(paper_id) for paper_id in paper_ids)
        extractions = self._collect_extractions(paper_ids)
        self._check_extractions(extractions, papers)

        normalization = self._normalization_repository.find()
        issues = check_normalization_against_extractions(normalization, extractions)
        if issues:
            # The issues name every local concept the normalization leaves out, which is how a normalization that
            # predates the newest extraction shows up.
            raise ConceptNormalizationValidationError(issues)

        chunks_by_paper = {paper.meta.id: chunk_paper(paper, self._tokenizer, self._max_tokens) for paper in papers}
        chunk_evidence = self._resolve_evidence(extractions, chunks_by_paper)
        chunks = tuple(chunk for paper_id in paper_ids for chunk in chunks_by_paper[paper_id])

        items = build_embedding_items(chunks, extractions, normalization.concepts)
        table = EmbeddingTable(items=items, vectors=self._embedder.embed_documents([item.text for item in items]))
        graph = build_knowledge_graph(extractions, normalization, chunk_evidence)
        manifest = IndexManifest(
            schema_version=INDEX_SCHEMA_VERSION,
            domain_model_version=DOMAIN_MODEL_VERSION,
            embedding_model=self._embedder.model_name,
            embedding_dimension=table.dimension,
            chunk_max_tokens=self._max_tokens,
            paper_ids=paper_ids,
        )

        self._index_repository.save_chunks(chunks)
        self._index_repository.save_embeddings(table)
        self._index_repository.save_graph(graph)
        # The manifest goes last: a build that dies halfway leaves an index without one, and search can read that
        # as "not built yet" instead of searching artifacts that do not belong together.
        self._index_repository.save_manifest(manifest)
        return IndexBuildRun(manifest=manifest, chunks=chunks, embeddings=table, graph=graph)

    def _collect_extractions(self, paper_ids: tuple[int, ...]) -> tuple[PaperExtraction, ...]:
        """Read every extraction, reporting all papers that still need one instead of only the first."""
        extractions: list[PaperExtraction] = []
        missing: list[int] = []
        for paper_id in paper_ids:
            try:
                extractions.append(self._extraction_repository.find(paper_id))
            except PaperExtractionNotFoundError:
                missing.append(paper_id)
        if missing:
            raise MissingPaperExtractionsError(tuple(missing))
        return tuple(extractions)

    def _check_extractions(self, extractions: Sequence[PaperExtraction], papers: Sequence[Paper]) -> None:
        """Compare every stored extraction with the pages of its paper, reporting all papers that disagree.

        An extraction passed this check when it was written, so a disagreement here means the extraction or the
        pages were edited by hand afterwards; the whole list is reported so one pass fixes all of them.
        """
        paper_by_id = {paper.meta.id: paper for paper in papers}
        issues_by_paper: dict[int, tuple[PaperExtractionIssue, ...]] = {}
        for extraction in extractions:
            issues = check_extraction_against_paper(extraction, paper_by_id[extraction.paper_id])
            if issues:
                issues_by_paper[extraction.paper_id] = issues
        if issues_by_paper:
            raise PaperExtractionMismatchError(issues_by_paper)

    def _resolve_evidence(
        self, extractions: Sequence[PaperExtraction], chunks_by_paper: Mapping[int, tuple[Chunk, ...]]
    ) -> dict[int, dict[PageEvidence, ChunkEvidence]]:
        """Tie the quotes of every extraction to the chunks of its paper, reporting all quotes that fit in none.

        A quote fails here only when the chunk budget cut it in two, so the unresolved quotes of every paper are
        gathered into one error: whoever raises the budget wants the whole list, not the first paper of it.
        """
        resolved: dict[int, dict[PageEvidence, ChunkEvidence]] = {}
        unresolved: list[UnresolvedEvidence] = []
        for extraction in extractions:
            try:
                resolved[extraction.paper_id] = resolve_extraction_evidence(
                    extraction, chunks_by_paper[extraction.paper_id]
                )
            except EvidenceResolutionError as error:
                unresolved.extend(error.unresolved)
        if unresolved:
            raise EvidenceResolutionError(tuple(unresolved))
        return resolved
