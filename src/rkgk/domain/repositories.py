"""Ports that the domain needs from the outside world.

The domain states what it wants to read, not where the bytes live, so an implementation may back these calls
with files, S3, or a database without the use cases noticing.
"""

from typing import Protocol

from rkgk.domain.paper import Paper, PaperIndexEntry


class PaperRepositoryError(Exception):
    """Raised when an artifact is missing, unreadable, or inconsistent with the metadata that describes it."""


class PaperRepository(Protocol):
    def find_index(self) -> tuple[PaperIndexEntry, ...]: ...

    def find(self, paper_id: int) -> Paper: ...
