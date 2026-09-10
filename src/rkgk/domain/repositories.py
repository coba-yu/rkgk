"""Ports that the domain needs from the outside world.

The domain states what it wants to read, not where the bytes live, so an implementation may back these calls
with files, S3, or a database without the use cases noticing.
"""

from typing import Protocol

from rkgk.domain.paper import Paper, PaperIndexEntry


class PaperRepositoryError(Exception):
    """Base of every failure a repository reports.

    Subclasses tell the caller what kind of failure it is, not which step of an implementation hit it,
    so a file-backed and an S3-backed repository raise the same classes.
    `location` and `paper_id` are attributes so a caller can render them without parsing the message.
    """

    def __init__(self, message: str, *, location: str | None = None, paper_id: int | None = None) -> None:
        super().__init__(message)
        self.location = location
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
