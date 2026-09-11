"""Port that the domain needs to read and write extractions from the outside world."""

from typing import Protocol

from rkgk.domain.models.paper_extraction import PaperExtraction
from rkgk.domain.repositories.base import RepositoryError


class PaperExtractionRepositoryError(RepositoryError):
    """Base of every failure the extraction repository reports."""


class PaperExtractionNotFoundError(PaperExtractionRepositoryError):
    """The paper has not been extracted yet, so running the extraction is the remedy."""


class PaperExtractionArtifactUnreadableError(PaperExtractionRepositoryError):
    """The artifact exists but cannot be read or written, so the environment is the remedy."""


class PaperExtractionArtifactInvalidError(PaperExtractionRepositoryError):
    """The artifact was read but is not a valid extraction, so a new extraction run is the remedy."""


class PaperExtractionRepository(Protocol):
    def save(self, result: PaperExtraction) -> None: ...

    def find(self, paper_id: int) -> PaperExtraction: ...
