"""Port that the domain needs to read and write the built index from the outside world.

The index is the set of artifacts the build command writes and the search command reads: the manifest, the
chunks, the embedding table, and the knowledge graph.
The manifest is saved last, so an index without one is an index whose build did not finish.
"""

from collections.abc import Sequence
from typing import Protocol

from rkgk.domain.models.chunk import Chunk
from rkgk.domain.models.embedding import EmbeddingTable
from rkgk.domain.models.graph import KnowledgeGraph
from rkgk.domain.models.manifest import IndexManifest
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
    def save_chunks(self, chunks: Sequence[Chunk]) -> None: ...

    def find_chunks(self) -> tuple[Chunk, ...]: ...

    def save_embeddings(self, table: EmbeddingTable) -> None: ...

    def find_embeddings(self) -> EmbeddingTable: ...

    def save_graph(self, graph: KnowledgeGraph) -> None: ...

    def find_graph(self) -> KnowledgeGraph: ...

    def save_manifest(self, manifest: IndexManifest) -> None: ...

    def find_manifest(self) -> IndexManifest: ...
