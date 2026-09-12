"""Index repository backed by files under `index/` in the data directory.

The vectors live in `embeddings.npy` and the items in `items.jsonl`, one item per line in the row order of the
vectors, so the items stay diffable and greppable while the vectors stay a plain binary matrix.
`find_embeddings` reassembles one `EmbeddingTable` from the pair, so a pair whose counts disagree is reported
instead of being searched.
The chunks live in `chunks.jsonl` the same way, one chunk per line, because a retrieval hit names a chunk id and
the text behind it has to be readable without rebuilding the index.
The graph lives in `graph.json`, the pydantic JSON of `KnowledgeGraph`, rather than a networkx node-link dump:
that way the stored graph gets the same schema validation on read as every other artifact, so a hand-edited or
stale file fails loudly instead of being traversed.
The manifest lives in `manifest.json` and is written after the artifacts it describes, so an index directory
without it is one whose build did not finish.
`find_index` reassembles the whole index into one `IndexBuildRun`, reading the manifest first so an index built
with another schema or another domain model is reported before any chunk, vector or graph is loaded.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from rkgk.domain import DOMAIN_MODEL_VERSION
from rkgk.domain.models.chunk import Chunk
from rkgk.domain.models.embedding import EmbeddedItem, EmbeddingTable
from rkgk.domain.models.graph import KnowledgeGraph
from rkgk.domain.models.manifest import INDEX_SCHEMA_VERSION, IndexBuildRun, IndexManifest
from rkgk.domain.repositories.index import (
    IndexArtifactInvalidError,
    IndexArtifactUnreadableError,
    IndexIncompatibleError,
    IndexNotFoundError,
    IndexRepositoryError,
)

INDEX_DIR_NAME = "index"
MANIFEST_FILE_NAME = "manifest.json"
CHUNKS_FILE_NAME = "chunks.jsonl"
ITEMS_FILE_NAME = "items.jsonl"
EMBEDDINGS_FILE_NAME = "embeddings.npy"
GRAPH_FILE_NAME = "graph.json"


def _fail(kind: type[IndexRepositoryError], path: Path, problem: str) -> IndexRepositoryError:
    return kind(f"{path}: {problem}", location=str(path))


class FileIndexRepository:
    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir / INDEX_DIR_NAME

    @property
    def manifest_path(self) -> Path:
        return self._dir / MANIFEST_FILE_NAME

    @property
    def chunks_path(self) -> Path:
        return self._dir / CHUNKS_FILE_NAME

    @property
    def items_path(self) -> Path:
        return self._dir / ITEMS_FILE_NAME

    @property
    def embeddings_path(self) -> Path:
        return self._dir / EMBEDDINGS_FILE_NAME

    @property
    def graph_path(self) -> Path:
        return self._dir / GRAPH_FILE_NAME

    def paths(self) -> tuple[Path, ...]:
        """Every file of the index, the manifest first, so a command can report what a build wrote."""
        return (self.manifest_path, self.chunks_path, self.items_path, self.embeddings_path, self.graph_path)

    def save_chunks(self, chunks: Sequence[Chunk]) -> None:
        self._write_text(
            self.chunks_path,
            "".join(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n" for chunk in chunks),
        )

    def find_chunks(self) -> tuple[Chunk, ...]:
        path = self.chunks_path
        raw = self._read_text(path)
        chunks: list[Chunk] = []
        seen: set[str] = set()
        for number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                chunk = Chunk.model_validate_json(line)
            except ValidationError as error:
                raise _fail(IndexArtifactInvalidError, path, f"line {number} is not a valid Chunk: {error}") from error
            if chunk.id in seen:
                # A repeated id would make a retrieval hit point at two different texts, so the file is rejected.
                raise _fail(IndexArtifactInvalidError, path, f"line {number} repeats the chunk {chunk.id!r}")
            seen.add(chunk.id)
            chunks.append(chunk)
        return tuple(chunks)

    def save_embeddings(self, table: EmbeddingTable) -> None:
        lines = "".join(json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n" for item in table.items)
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            self.items_path.write_text(lines, encoding="utf-8")
            with self.embeddings_path.open("wb") as stream:
                np.save(stream, table.vectors)
        except OSError as error:
            # A failed write is an environment problem, the same kind as a failed read, so it uses the same class.
            raise _fail(
                IndexArtifactUnreadableError, self._dir, f"cannot be written: {error.strerror or error}"
            ) from error

    def find_embeddings(self) -> EmbeddingTable:
        items = self._read_items()
        vectors = self._read_vectors()
        try:
            return EmbeddingTable(items=items, vectors=vectors)
        except ValidationError as error:
            raise _fail(IndexArtifactInvalidError, self._dir, f"is not a valid EmbeddingTable: {error}") from error

    def save_graph(self, graph: KnowledgeGraph) -> None:
        self._write_text(self.graph_path, graph.model_dump_json(indent=2) + "\n")

    def find_graph(self) -> KnowledgeGraph:
        path = self.graph_path
        raw = self._read_text(path)
        try:
            return KnowledgeGraph.model_validate_json(raw)
        except ValidationError as error:
            raise _fail(IndexArtifactInvalidError, path, f"is not a valid KnowledgeGraph: {error}") from error

    def save_manifest(self, manifest: IndexManifest) -> None:
        self._write_text(self.manifest_path, manifest.model_dump_json(indent=2) + "\n")

    def find_manifest(self) -> IndexManifest:
        path = self.manifest_path
        raw = self._read_text(path)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise _fail(IndexArtifactInvalidError, path, f"is not valid JSON: {error}") from error
        if isinstance(payload, dict):
            # The versions are compared before pydantic runs because `schema_version` is a Literal: another version
            # would only produce a generic validation error, hiding that a rebuild is needed rather than a repair.
            self._check_version(path, payload, "schema_version", "index schema", INDEX_SCHEMA_VERSION)
            self._check_version(path, payload, "domain_model_version", "domain model", DOMAIN_MODEL_VERSION)
        try:
            return IndexManifest.model_validate(payload)
        except ValidationError as error:
            raise _fail(IndexArtifactInvalidError, path, f"is not a valid IndexManifest: {error}") from error

    def find_index(self) -> IndexBuildRun:
        """Read the whole index, the manifest first so a version mismatch is reported before anything is loaded."""
        manifest = self.find_manifest()
        chunks = self.find_chunks()
        embeddings = self.find_embeddings()
        graph = self.find_graph()
        try:
            return IndexBuildRun(manifest=manifest, chunks=chunks, embeddings=embeddings, graph=graph)
        except ValidationError as error:
            raise _fail(
                IndexArtifactInvalidError, self._dir, f"holds artifacts that do not belong together: {error}"
            ) from error

    @staticmethod
    def _check_version(path: Path, payload: dict[str, object], field: str, label: str, expected: int) -> None:
        """Reject a version the current code does not read, naming what was found and what is read instead."""
        found = payload.get(field)
        if field in payload and found != expected:
            raise _fail(
                IndexIncompatibleError,
                path,
                f"built with {label} {found} while this rkgk reads {expected}; rebuild the index",
            )

    def _read_items(self) -> tuple[EmbeddedItem, ...]:
        path = self.items_path
        raw = self._read_text(path)
        items: list[EmbeddedItem] = []
        for number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                items.append(EmbeddedItem.model_validate_json(line))
            except ValidationError as error:
                raise _fail(
                    IndexArtifactInvalidError, path, f"line {number} is not a valid EmbeddedItem: {error}"
                ) from error
        return tuple(items)

    def _read_vectors(self) -> np.ndarray:
        path = self.embeddings_path
        try:
            with path.open("rb") as stream:
                # allow_pickle stays off so a tampered file can only fail to parse, never run code.
                vectors = np.load(stream, allow_pickle=False)
        except FileNotFoundError as error:
            raise _fail(IndexNotFoundError, path, "not found") from error
        except OSError as error:
            raise _fail(IndexArtifactUnreadableError, path, f"cannot be read: {error.strerror or error}") from error
        except ValueError as error:
            raise _fail(IndexArtifactInvalidError, path, f"is not a valid npy file: {error}") from error
        if not isinstance(vectors, np.ndarray):
            raise _fail(IndexArtifactInvalidError, path, f"holds a {type(vectors).__name__} instead of an array")
        return vectors

    def _read_text(self, path: Path) -> str:
        """Read one text artifact, telling a missing index from an unreadable file and from unreadable bytes."""
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError as error:
            raise _fail(IndexNotFoundError, path, "not found") from error
        except OSError as error:
            raise _fail(IndexArtifactUnreadableError, path, f"cannot be read: {error.strerror or error}") from error
        except UnicodeDecodeError as error:
            raise _fail(IndexArtifactInvalidError, path, f"is not valid UTF-8: {error}") from error

    def _write_text(self, path: Path, text: str) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError as error:
            # A failed write is an environment problem, the same kind as a failed read, so it uses the same class.
            raise _fail(
                IndexArtifactUnreadableError, self._dir, f"cannot be written: {error.strerror or error}"
            ) from error
