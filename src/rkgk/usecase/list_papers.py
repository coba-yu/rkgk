"""Use case that lists the papers known to the repository."""

from rkgk.domain.models.paper import PaperIndexEntry
from rkgk.domain.repositories.paper import PaperRepository


class ListPapersUseCase:
    def __init__(self, repository: PaperRepository) -> None:
        self._repository = repository

    def execute(self) -> tuple[PaperIndexEntry, ...]:
        return self._repository.find_index()
