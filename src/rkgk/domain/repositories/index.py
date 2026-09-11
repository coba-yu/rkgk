"""Port that the domain needs to read and write the built index from the outside world.

The index is the set of artifacts the build command writes and the search command reads; for now that is the
embedding table, and the graph and the manifest join it as they are built.
"""

from typing import Protocol

from rkgk.domain.models.embedding import EmbeddingTable
from rkgk.domain.repositories.base import RepositoryError


class IndexRepositoryError(RepositoryError):
    """Base of every failure the index repository reports."""


class IndexNotFoundError(IndexRepositoryError):
    """The index has not been built yet, so running the build is the remedy."""


class IndexArtifactUnreadableError(IndexRepositoryError):
    """The artifact exists but cannot be read or written, so the environment is the remedy."""


class IndexArtifactInvalidError(IndexRepositoryError):
    """The artifact was read but is not a valid part of an index, so a rebuild is the remedy."""


class IndexRepository(Protocol):
    def save_embeddings(self, table: EmbeddingTable) -> None: ...

    def find_embeddings(self) -> EmbeddingTable: ...
