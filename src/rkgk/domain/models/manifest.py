"""What one build recorded about the index it produced.

The manifest says with which settings an index was made: the schema of its artifacts, the version of the domain
model behind them, the embedding model and the dimension of its vectors, the chunk budget, and the papers it
covers.
Search reads the manifest before it reads anything else, so an index made with another model or another schema
becomes a clear error instead of a query compared against vectors it does not belong with.
"""

from collections.abc import Collection
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from rkgk.domain.models.base import Entity
from rkgk.domain.models.chunk import Chunk
from rkgk.domain.models.embedding import EmbeddedItemKind, EmbeddingTable
from rkgk.domain.models.graph import KnowledgeGraph

INDEX_SCHEMA_VERSION = 1

# A mixed index can differ in thousands of ids, and a message that lists them all is read by nobody, so only the
# first few are named and the rest are counted.
_MAX_REPORTED_IDS = 5


def _describe_ids(ids: Collection[object]) -> str:
    """Render the ids sorted as text, naming at most `_MAX_REPORTED_IDS` of them and counting the rest."""
    ordered = sorted(str(value) for value in ids)
    named = ", ".join(repr(value) for value in ordered[:_MAX_REPORTED_IDS])
    left_out = len(ordered) - _MAX_REPORTED_IDS
    if left_out > 0:
        return f"{named} and {left_out} more"
    return named


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
    """What one build produced and what search reads back: the manifest and the artifacts beside it.

    Both directions go through the same object, so the cross-checks below also catch a stored index whose files
    do not belong together, such as a manifest left over from a build that wrote other vectors.
    The checks compare the artifacts by identity, naming the offending ids, rather than by count: a build that
    was interrupted after it rewrote the chunks and the vectors but before the graph and the manifest leaves an
    index that mixes two generations of artifacts while the counts still agree.
    """

    manifest: IndexManifest
    chunks: tuple[Chunk, ...]
    embeddings: EmbeddingTable
    graph: KnowledgeGraph

    @model_validator(mode="after")
    def _check_artifacts_against_the_manifest(self) -> Self:
        """The manifest describes the artifacts of this very run, so what it claims is checked where they meet.

        Each check below compares one pair of artifacts by the ids they hold, so a vector paired with a chunk,
        a paper or a concept of another generation of the index is reported instead of searched.
        """
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
        self._check_chunks_belong_to_the_manifest()
        self._check_chunk_items()
        self._check_summary_items()
        self._check_concept_items()
        return self

    def _check_chunks_belong_to_the_manifest(self) -> None:
        """A chunk of a paper the manifest leaves out comes from another build, so its paper must be covered."""
        stray = {chunk.paper_id for chunk in self.chunks} - set(self.manifest.paper_ids)
        if stray:
            raise ValueError(f"chunks hold papers the manifest does not cover: {_describe_ids(stray)}")

    def _check_chunk_items(self) -> None:
        """Every chunk needs its vector and every chunk vector its chunk, embedded from the text the chunk shows."""
        text_by_chunk_id = {chunk.id: chunk.text for chunk in self.chunks}
        items = self._collect_items(EmbeddedItemKind.CHUNK)
        self._reject_ref_mismatch(EmbeddedItemKind.CHUNK, set(items), set(text_by_chunk_id), "chunks")
        # The refs match the chunk ids at this point, so every ref resolves and only the text can still differ.
        # A differing text means the vector was made from another text than the chunk a hit will show.
        differing = {ref for ref, text in items.items() if text_by_chunk_id[ref] != text}
        if differing:
            raise ValueError(
                f"chunk items were embedded from another text than their chunk: {_describe_ids(differing)}"
            )

    def _check_summary_items(self) -> None:
        """One summary vector per paper of the manifest: a summary hit answers for the paper it names."""
        refs = set(self._collect_items(EmbeddedItemKind.SUMMARY))
        expected = {str(paper_id) for paper_id in self.manifest.paper_ids}
        self._reject_ref_mismatch(EmbeddedItemKind.SUMMARY, refs, expected, "papers")

    def _check_concept_items(self) -> None:
        """One concept vector per node of the graph: a concept hit is the entry point into traversal."""
        refs = set(self._collect_items(EmbeddedItemKind.CONCEPT))
        expected = {concept.id for concept in self.graph.concepts}
        self._reject_ref_mismatch(EmbeddedItemKind.CONCEPT, refs, expected, "concepts")

    def _collect_items(self, kind: EmbeddedItemKind) -> dict[str, str]:
        """The text of every item of one kind by its ref; `EmbeddingTable` already rejects a repeated ref."""
        return {item.ref: item.text for item in self.embeddings.items if item.kind is kind}

    @staticmethod
    def _reject_ref_mismatch(kind: EmbeddedItemKind, refs: set[str], expected: set[str], referenced: str) -> None:
        """Report what the items point at that no longer exists, and what exists without an item."""
        unknown = refs - expected
        if unknown:
            raise ValueError(
                f"{kind.value} items point at {referenced} the index does not hold: {_describe_ids(unknown)}"
            )
        missing = expected - refs
        if missing:
            raise ValueError(f"{referenced} without a {kind.value} item: {_describe_ids(missing)}")
