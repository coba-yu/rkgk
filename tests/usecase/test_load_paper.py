from datetime import UTC, datetime

import pytest

from rkgk.domain.paper import Page, Paper, PaperIndexEntry, PaperMeta, PaperPreprocessInfo
from rkgk.domain.repositories import PaperNotFoundError, PaperRepositoryError
from rkgk.usecase.load_paper import LoadPaperUseCase

PREPROCESS = PaperPreprocessInfo(tool="pymupdf", version="1.24.0", processed_at=datetime(2026, 1, 1, tzinfo=UTC))
META = PaperMeta(
    id=1,
    title="Retrieval-Augmented Generation for Conference Paper Search",
    authors=("Ada Lovelace",),
    year=2026,
    venue="NeurIPS",
    page_count=1,
    preprocess=PREPROCESS,
)
PAPER = Paper(meta=META, pages=(Page(number=1, text="intro\n"),))


class FakePaperRepository:
    def __init__(self, papers: dict[int, Paper]) -> None:
        self._papers = papers

    def find_index(self) -> tuple[PaperIndexEntry, ...]:
        return tuple(PaperIndexEntry(id=paper.meta.id, title=paper.meta.title) for paper in self._papers.values())

    def find(self, paper_id: int) -> Paper:
        if paper_id not in self._papers:
            raise PaperNotFoundError(f"paper {paper_id}: paper directory not found", paper_id=paper_id)
        return self._papers[paper_id]


def test_execute_returns_the_paper_the_repository_holds() -> None:
    use_case = LoadPaperUseCase(FakePaperRepository({1: PAPER}))
    assert use_case.execute(1) == PAPER


def test_execute_propagates_the_repository_error_for_an_unknown_paper() -> None:
    use_case = LoadPaperUseCase(FakePaperRepository({1: PAPER}))
    with pytest.raises(PaperRepositoryError, match="paper 2"):
        use_case.execute(2)
