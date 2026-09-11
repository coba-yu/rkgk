import json
from pathlib import Path

import numpy as np
import pytest

from rkgk.domain.models.embedding import EmbeddedItem, EmbeddedItemKind, EmbeddingTable
from rkgk.domain.repositories.index import IndexArtifactInvalidError, IndexNotFoundError, IndexRepositoryError
from rkgk.infrastructure.file_index_repository import FileIndexRepository

TABLE = EmbeddingTable(
    items=(
        EmbeddedItem(kind=EmbeddedItemKind.CHUNK, ref="1:0", paper_id=1, text="We study retrieval."),
        EmbeddedItem(kind=EmbeddedItemKind.SUMMARY, ref="1", paper_id=1, text="検索を研究する。"),
        EmbeddedItem(kind=EmbeddedItemKind.CONCEPT, ref="retrieval", text="Retrieval (検索)"),
    ),
    vectors=np.arange(12, dtype=np.float32).reshape(3, 4),
)


def test_a_saved_table_is_read_back_unchanged(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_embeddings(TABLE)
    assert repository.find_embeddings() == TABLE


def test_the_vectors_keep_their_dtype_and_shape(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_embeddings(TABLE)
    vectors = repository.find_embeddings().vectors
    assert vectors.dtype == np.float32
    assert vectors.shape == (3, 4)


def test_the_items_are_written_one_per_line_in_row_order(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_embeddings(TABLE)
    assert repository.items_path == tmp_path / "index" / "items.jsonl"
    assert repository.embeddings_path == tmp_path / "index" / "embeddings.npy"
    lines = repository.items_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["ref"] for line in lines] == ["1:0", "1", "retrieval"]
    assert json.loads(lines[2]) == {
        "kind": "concept",
        "ref": "retrieval",
        "paper_id": None,
        "text": "Retrieval (検索)",
    }


def test_the_items_are_stored_as_japanese_characters(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_embeddings(TABLE)
    assert "検索を研究する。" in repository.items_path.read_text(encoding="utf-8")


def test_saving_twice_replaces_the_previous_table(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_embeddings(TABLE)
    smaller = EmbeddingTable(items=TABLE.items[:1], vectors=TABLE.vectors[:1])
    repository.save_embeddings(smaller)
    assert repository.find_embeddings() == smaller


def test_an_empty_table_survives_the_round_trip(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    empty = EmbeddingTable(items=(), vectors=np.empty((0, 4), dtype=np.float32))
    repository.save_embeddings(empty)
    assert repository.find_embeddings() == empty


def test_a_data_directory_without_an_index_is_reported_as_not_found(tmp_path: Path) -> None:
    with pytest.raises(IndexNotFoundError, match="items.jsonl: not found"):
        FileIndexRepository(tmp_path).find_embeddings()


def test_an_index_whose_vectors_are_gone_is_reported_as_not_found(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_embeddings(TABLE)
    repository.embeddings_path.unlink()
    with pytest.raises(IndexNotFoundError, match="embeddings.npy: not found"):
        repository.find_embeddings()


def test_a_corrupted_item_line_is_reported_with_its_line_number(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_embeddings(TABLE)
    lines = repository.items_path.read_text(encoding="utf-8").splitlines()
    lines[1] = '{"kind": "summary", "ref": "1"}'
    repository.items_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(IndexArtifactInvalidError, match="line 2 is not a valid EmbeddedItem"):
        repository.find_embeddings()


def test_a_non_utf8_items_file_is_reported_as_invalid(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_embeddings(TABLE)
    repository.items_path.write_bytes(b"\xff\xfe")
    with pytest.raises(IndexArtifactInvalidError, match="is not valid UTF-8"):
        repository.find_embeddings()


def test_a_vectors_file_that_is_not_npy_is_reported_as_invalid(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_embeddings(TABLE)
    repository.embeddings_path.write_bytes(b"not an npy file")
    with pytest.raises(IndexArtifactInvalidError, match="is not a valid npy file"):
        repository.find_embeddings()


def test_items_and_vectors_of_different_lengths_are_reported_as_invalid(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_embeddings(TABLE)
    with repository.embeddings_path.open("wb") as stream:
        np.save(stream, TABLE.vectors[:2])
    with pytest.raises(IndexArtifactInvalidError, match="vectors has 2 rows for 3 items"):
        repository.find_embeddings()


def test_errors_carry_the_location_as_an_attribute(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    with pytest.raises(IndexRepositoryError) as caught:
        repository.find_embeddings()
    assert caught.value.location == str(repository.items_path)
