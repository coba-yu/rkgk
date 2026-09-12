"""What one build recorded about the index it produced.

The manifest says with which settings an index was made: the schema of its artifacts, the version of the domain
model behind them, the embedding model and the dimension of its vectors, the chunk budget, and the papers it
covers.
Search reads the manifest before it reads anything else, so an index made with another model or another schema
becomes a clear error instead of a query compared against vectors it does not belong with.
"""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from rkgk.domain.models.base import Entity
from rkgk.domain.models.chunk import Chunk
from rkgk.domain.models.embedding import EmbeddingTable
from rkgk.domain.models.graph import KnowledgeGraph

INDEX_SCHEMA_VERSION = 1


class IndexManifest(Entity):
    # Literal pins the version so an artifact written against another schema fails validation instead of being
    # read as if it were the current one.
    schema_version: Literal[1]
    domain_model_version: int = Field(ge=1)
    embedding_model: str = Field(min_length=1)
    embedding_dimension: int = Field(ge=1)
    chunk_max_tokens: int = Field(ge=1)
    paper_ids: tuple[Annotated[int, Field(ge=1)], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_repeated_paper_ids(self) -> Self:
        declared: set[int] = set()
        for paper_id in self.paper_ids:
            if paper_id in declared:
                raise ValueError(f"paper_ids declares {paper_id} more than once")
            declared.add(paper_id)
        return self


class IndexBuildRun(Entity):
    """What one build produced: the manifest and the artifacts it wrote."""

    manifest: IndexManifest
    chunks: tuple[Chunk, ...]
    embeddings: EmbeddingTable
    graph: KnowledgeGraph

    @model_validator(mode="after")
    def _check_artifacts_against_the_manifest(self) -> Self:
        """The manifest describes the artifacts of this very run, so what it claims is checked where they meet."""
        if self.manifest.paper_ids != self.graph.paper_ids:
            raise ValueError(
                f"manifest covers papers {list(self.manifest.paper_ids)} "
                f"while the graph holds {list(self.graph.paper_ids)}"
            )
        if self.manifest.embedding_dimension != self.embeddings.dimension:
            raise ValueError(
                f"manifest declares dimension {self.manifest.embedding_dimension} "
                f"while the vectors have {self.embeddings.dimension}"
            )
        expected = len(self.chunks) + len(self.manifest.paper_ids) + len(self.graph.concepts)
        if len(self.embeddings.items) != expected:
            raise ValueError(
                f"embeddings holds {len(self.embeddings.items)} items for {len(self.chunks)} chunks, "
                f"{len(self.manifest.paper_ids)} summaries and {len(self.graph.concepts)} concepts"
            )
        return self
