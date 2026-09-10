"""Use case that reads one paper with its pages."""

from rkgk.domain.paper import Paper
from rkgk.domain.repositories import PaperRepository


class LoadPaperUseCase:
    def __init__(self, repository: PaperRepository) -> None:
        self._repository = repository

    def execute(self, paper_id: int) -> Paper:
        return self._repository.find(paper_id)
