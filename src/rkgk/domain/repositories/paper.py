"""Port that the domain needs to read papers from the outside world."""

from typing import Protocol

from rkgk.domain.models.paper import Paper, PaperIndexEntry, PaperMeta
from rkgk.domain.repositories.base import RepositoryError


class PaperRepositoryError(RepositoryError):
    """Base of every failure the paper repository reports.

    `paper_id` is an attribute so a caller can render which paper failed without parsing the message.
    """

    def __init__(self, message: str, *, location: str | None = None, paper_id: int | None = None) -> None:
        super().__init__(message, location=location)
        self.paper_id = paper_id


class PaperNotFoundError(PaperRepositoryError):
    """The paper has no artifacts yet, so preprocessing or a pull from storage is the remedy."""


class PaperArtifactUnreadableError(PaperRepositoryError):
    """The artifact exists but cannot be read or listed, so the environment is the remedy."""


class PaperArtifactInvalidError(PaperRepositoryError):
    """The artifact was read but violates the handoff contract, so the preprocessing output is the remedy."""


class PaperRepository(Protocol):
    def find_index(self) -> tuple[PaperIndexEntry, ...]: ...

    def find(self, paper_id: int) -> Paper: ...

    # Separate from `find` because some callers (e.g. assembling a search result) only need the title and the
    # S3 URI; reading every page for each of them would turn a candidate list into corpus-wide I/O, and a page
    # missing for an unrelated reason would fail a search that never needed the page.
    def find_meta(self, paper_id: int) -> PaperMeta: ...
