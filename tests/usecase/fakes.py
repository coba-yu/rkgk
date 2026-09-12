"""Fakes and builders shared by the use case tests."""

from collections.abc import Sequence
from datetime import UTC, datetime

from rkgk.domain.models.chunk import Chunk
from rkgk.domain.models.concept_normalization import ConceptNormalization
from rkgk.domain.models.embedding import EmbeddingTable
from rkgk.domain.models.graph import KnowledgeGraph
from rkgk.domain.models.manifest import IndexBuildRun, IndexManifest
from rkgk.domain.models.paper import Page, Paper, PaperIndexEntry, PaperMeta, PaperPreprocessInfo
from rkgk.domain.models.paper_extraction import PaperExtraction
from rkgk.domain.repositories.concept_normalization import ConceptNormalizationNotFoundError
from rkgk.domain.repositories.index import IndexNotFoundError
from rkgk.domain.repositories.paper import PaperNotFoundError
from rkgk.domain.repositories.paper_extraction import PaperExtractionNotFoundError

PREPROCESS = PaperPreprocessInfo(tool="pymupdf", version="1.24.0", processed_at=datetime(2026, 1, 1, tzinfo=UTC))


def build_paper(paper_id: int, *page_texts: str) -> Paper:
    pages = tuple(Page(number=number, text=text) for number, text in enumerate(page_texts, start=1))
    return Paper(
        meta=PaperMeta(
            id=paper_id,
            title="Retrieval-Augmented Generation for Conference Paper Search",
            authors=("Ada Lovelace",),
            year=2026,
            venue="NeurIPS",
            page_count=len(pages),
            preprocess=PREPROCESS,
        ),
        pages=pages,
    )


class FakePaperRepository:
    def __init__(self, papers: dict[int, Paper]) -> None:
        self._papers = papers

    def find_index(self) -> tuple[PaperIndexEntry, ...]:
        return tuple(PaperIndexEntry(id=paper.meta.id, title=paper.meta.title) for paper in self._papers.values())

    def find(self, paper_id: int) -> Paper:
        if paper_id not in self._papers:
            raise PaperNotFoundError(f"paper {paper_id}: paper directory not found", paper_id=paper_id)
        return self._papers[paper_id]


class FakePaperExtractionRepository:
    """Holds the extractions a test starts with, and records every extraction the use case saves."""

    def __init__(self, *extractions: PaperExtraction) -> None:
        self.saved: list[PaperExtraction] = []
        self._stored = {extraction.paper_id: extraction for extraction in extractions}

    def save(self, result: PaperExtraction) -> None:
        self.saved.append(result)
        self._stored[result.paper_id] = result

    def find(self, paper_id: int) -> PaperExtraction:
        if paper_id not in self._stored:
            raise PaperExtractionNotFoundError(f"paper {paper_id}: extraction not found", paper_id=paper_id)
        return self._stored[paper_id]


class FakeConceptNormalizationRepository:
    def __init__(self) -> None:
        self.saved: list[ConceptNormalization] = []

    def save(self, normalization: ConceptNormalization) -> None:
        self.saved.append(normalization)

    def find(self) -> ConceptNormalization:
        if not self.saved:
            raise ConceptNormalizationNotFoundError("normalization not found")
        return self.saved[-1]


class FakeIndexRepository:
    """Records what a build saves, and in `saved` the order it saved it in, so a test can check both."""

    def __init__(self) -> None:
        self.saved: list[str] = []
        self.chunks: tuple[Chunk, ...] | None = None
        self.table: EmbeddingTable | None = None
        self.graph: KnowledgeGraph | None = None
        self.manifest: IndexManifest | None = None

    def save_chunks(self, chunks: Sequence[Chunk]) -> None:
        self.saved.append("chunks")
        self.chunks = tuple(chunks)

    def find_chunks(self) -> tuple[Chunk, ...]:
        if self.chunks is None:
            raise IndexNotFoundError("chunks not found")
        return self.chunks

    def save_embeddings(self, table: EmbeddingTable) -> None:
        self.saved.append("embeddings")
        self.table = table

    def find_embeddings(self) -> EmbeddingTable:
        if self.table is None:
            raise IndexNotFoundError("embeddings not found")
        return self.table

    def save_graph(self, graph: KnowledgeGraph) -> None:
        self.saved.append("graph")
        self.graph = graph

    def find_graph(self) -> KnowledgeGraph:
        if self.graph is None:
            raise IndexNotFoundError("graph not found")
        return self.graph

    def save_manifest(self, manifest: IndexManifest) -> None:
        self.saved.append("manifest")
        self.manifest = manifest

    def find_manifest(self) -> IndexManifest:
        if self.manifest is None:
            raise IndexNotFoundError("manifest not found")
        return self.manifest

    def find_index(self) -> IndexBuildRun:
        # The manifest is checked first, the way the file-backed repository reads it first.
        if self.manifest is None:
            raise IndexNotFoundError("manifest not found")
        if self.chunks is None:
            raise IndexNotFoundError("chunks not found")
        if self.table is None:
            raise IndexNotFoundError("embeddings not found")
        if self.graph is None:
            raise IndexNotFoundError("graph not found")
        return IndexBuildRun(manifest=self.manifest, chunks=self.chunks, embeddings=self.table, graph=self.graph)
