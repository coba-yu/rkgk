import pytest

from rkgk.domain.models.paper import Paper, PaperIndexEntry
from rkgk.domain.repositories.paper import PaperNotFoundError, PaperRepositoryError
from rkgk.usecase.list_papers import ListPapersUseCase

ENTRIES = (
    PaperIndexEntry(id=1, title="Retrieval-Augmented Generation for Conference Paper Search"),
    PaperIndexEntry(id=2, title="Knowledge Graphs as Retrieval Backbones"),
)


class FakePaperRepository:
    def __init__(self, entries: tuple[PaperIndexEntry, ...] | PaperRepositoryError) -> None:
        self._entries = entries

    def find_index(self) -> tuple[PaperIndexEntry, ...]:
        if isinstance(self._entries, PaperRepositoryError):
            raise self._entries
        return self._entries

    def find(self, paper_id: int) -> Paper:
        raise PaperNotFoundError(f"paper {paper_id}: paper directory not found", paper_id=paper_id)


def test_execute_returns_the_entries_the_repository_holds() -> None:
    use_case = ListPapersUseCase(FakePaperRepository(ENTRIES))
    assert use_case.execute() == ENTRIES


def test_execute_returns_no_entries_for_an_empty_index() -> None:
    use_case = ListPapersUseCase(FakePaperRepository(()))
    assert use_case.execute() == ()


def test_execute_propagates_the_repository_error_for_an_unreadable_index() -> None:
    use_case = ListPapersUseCase(FakePaperRepository(PaperRepositoryError("index.json: is not valid JSON")))
    with pytest.raises(PaperRepositoryError, match="index.json"):
        use_case.execute()
