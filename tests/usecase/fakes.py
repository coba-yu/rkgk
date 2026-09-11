"""Fakes and builders shared by the use case tests."""

from datetime import UTC, datetime

from rkgk.domain.models.concept_normalization import ConceptNormalization
from rkgk.domain.models.paper import Page, Paper, PaperIndexEntry, PaperMeta, PaperPreprocessInfo
from rkgk.domain.models.paper_extraction import PaperExtraction
from rkgk.domain.repositories.concept_normalization import ConceptNormalizationNotFoundError
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
