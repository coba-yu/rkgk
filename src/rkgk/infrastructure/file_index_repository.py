"""Index repository backed by files under `index/` in the data directory.

The vectors live in `embeddings.npy` and the items in `items.jsonl`, one item per line in the row order of the
vectors, so the items stay diffable and greppable while the vectors stay a plain binary matrix.
`find_embeddings` reassembles one `EmbeddingTable` from the pair, so a pair whose counts disagree is reported
instead of being searched.
"""

import json
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from rkgk.domain.models.embedding import EmbeddedItem, EmbeddingTable
from rkgk.domain.repositories.index import (
    IndexArtifactInvalidError,
    IndexArtifactUnreadableError,
    IndexNotFoundError,
    IndexRepositoryError,
)

INDEX_DIR_NAME = "index"
EMBEDDINGS_FILE_NAME = "embeddings.npy"
ITEMS_FILE_NAME = "items.jsonl"


def _fail(kind: type[IndexRepositoryError], path: Path, problem: str) -> IndexRepositoryError:
    return kind(f"{path}: {problem}", location=str(path))


class FileIndexRepository:
    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir / INDEX_DIR_NAME

    @property
    def embeddings_path(self) -> Path:
        return self._dir / EMBEDDINGS_FILE_NAME

    @property
    def items_path(self) -> Path:
        return self._dir / ITEMS_FILE_NAME

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

    def _read_items(self) -> tuple[EmbeddedItem, ...]:
        path = self.items_path
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError as error:
            raise _fail(IndexNotFoundError, path, "not found") from error
        except OSError as error:
            raise _fail(IndexArtifactUnreadableError, path, f"cannot be read: {error.strerror or error}") from error
        except UnicodeDecodeError as error:
            raise _fail(IndexArtifactInvalidError, path, f"is not valid UTF-8: {error}") from error
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
