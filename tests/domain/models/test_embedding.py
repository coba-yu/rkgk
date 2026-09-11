import numpy as np
import pytest
from pydantic import ValidationError

from rkgk.domain.models.embedding import EmbeddedItem, EmbeddedItemKind, EmbeddingTable

CHUNK = EmbeddedItem(kind=EmbeddedItemKind.CHUNK, ref="1:0", paper_id=1, text="We study retrieval.")
SUMMARY = EmbeddedItem(kind=EmbeddedItemKind.SUMMARY, ref="1", paper_id=1, text="検索を研究する。")
CONCEPT = EmbeddedItem(kind=EmbeddedItemKind.CONCEPT, ref="retrieval", text="Retrieval")


def test_a_chunk_item_needs_the_paper_it_belongs_to() -> None:
    with pytest.raises(ValidationError, match="paper_id is required when kind is chunk"):
        EmbeddedItem(kind=EmbeddedItemKind.CHUNK, ref="1:0", text="We study retrieval.")


def test_a_chunk_item_must_reference_a_chunk_of_its_own_paper() -> None:
    with pytest.raises(ValidationError, match="ref must be a chunk id of paper 2"):
        EmbeddedItem(kind=EmbeddedItemKind.CHUNK, ref="1:0", paper_id=2, text="We study retrieval.")


def test_a_summary_item_needs_the_paper_it_belongs_to() -> None:
    with pytest.raises(ValidationError, match="paper_id is required when kind is summary"):
        EmbeddedItem(kind=EmbeddedItemKind.SUMMARY, ref="1", text="検索を研究する。")


def test_a_summary_item_must_reference_its_own_paper() -> None:
    with pytest.raises(ValidationError, match="ref must be '2' when kind is summary"):
        EmbeddedItem(kind=EmbeddedItemKind.SUMMARY, ref="1", paper_id=2, text="検索を研究する。")


def test_a_concept_item_belongs_to_no_single_paper() -> None:
    with pytest.raises(ValidationError, match="paper_id must be None when kind is concept"):
        EmbeddedItem(kind=EmbeddedItemKind.CONCEPT, ref="retrieval", paper_id=1, text="Retrieval")


def test_a_blank_text_is_rejected() -> None:
    with pytest.raises(ValidationError, match="text must contain non-whitespace characters"):
        EmbeddedItem(kind=EmbeddedItemKind.CONCEPT, ref="retrieval", text=" \n")


def test_an_empty_ref_is_rejected() -> None:
    with pytest.raises(ValidationError, match="ref"):
        EmbeddedItem(kind=EmbeddedItemKind.CONCEPT, ref="", text="Retrieval")


def test_the_kind_is_serialized_as_its_value() -> None:
    assert CHUNK.model_dump(mode="json")["kind"] == "chunk"


def test_a_table_pairs_each_item_with_one_row() -> None:
    table = EmbeddingTable(items=(CHUNK, SUMMARY, CONCEPT), vectors=np.zeros((3, 4), dtype=np.float32))
    assert table.dimension == 4
    assert table.vectors.shape == (3, 4)


def test_a_table_with_more_rows_than_items_is_rejected() -> None:
    with pytest.raises(ValidationError, match="vectors has 3 rows for 2 items"):
        EmbeddingTable(items=(CHUNK, SUMMARY), vectors=np.zeros((3, 4), dtype=np.float32))


def test_a_table_needs_a_matrix_not_a_flat_vector() -> None:
    with pytest.raises(ValidationError, match="vectors must be two-dimensional, got 1 dimensions"):
        EmbeddingTable(items=(CHUNK,), vectors=np.zeros(4, dtype=np.float32))


def test_a_table_holds_each_item_once() -> None:
    with pytest.raises(ValidationError, match="items holds chunk '1:0' more than once"):
        EmbeddingTable(items=(CHUNK, CHUNK), vectors=np.zeros((2, 4), dtype=np.float32))


def test_the_same_ref_under_two_kinds_is_two_items() -> None:
    paper = EmbeddedItem(kind=EmbeddedItemKind.SUMMARY, ref="1", paper_id=1, text="要約")
    concept = EmbeddedItem(kind=EmbeddedItemKind.CONCEPT, ref="1", text="One")
    assert len(EmbeddingTable(items=(paper, concept), vectors=np.zeros((2, 2), dtype=np.float32)).items) == 2


def test_an_empty_table_keeps_its_dimension() -> None:
    assert EmbeddingTable(items=(), vectors=np.empty((0, 8), dtype=np.float32)).dimension == 8


def test_tables_compare_by_items_and_vector_values() -> None:
    vectors = np.arange(8, dtype=np.float32).reshape(2, 4)
    table = EmbeddingTable(items=(CHUNK, SUMMARY), vectors=vectors)
    assert table == EmbeddingTable(items=(CHUNK, SUMMARY), vectors=vectors.copy())
    assert table != EmbeddingTable(items=(CHUNK, SUMMARY), vectors=vectors + 1)
    assert table != EmbeddingTable(items=(CHUNK, CONCEPT), vectors=vectors)
    assert table != "not a table"
