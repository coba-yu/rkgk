import numpy as np
import pytest

from rkgk.infrastructure.fake_embedder import FakeEmbedder


def test_the_same_text_always_gets_the_same_vector() -> None:
    first = FakeEmbedder().embed_documents(["retrieval"])
    second = FakeEmbedder().embed_documents(["retrieval"])
    assert np.array_equal(first, second)


def test_different_texts_get_different_vectors() -> None:
    vectors = FakeEmbedder().embed_documents(["retrieval", "generation"])
    assert not np.array_equal(vectors[0], vectors[1])


def test_the_vectors_have_one_row_per_text_and_the_requested_dimension() -> None:
    vectors = FakeEmbedder(dimension=5).embed_documents(["a", "b", "c"])
    assert vectors.shape == (3, 5)
    assert vectors.dtype == np.float32


def test_the_vectors_have_unit_length() -> None:
    vectors = FakeEmbedder().embed_documents(["a", "b"])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_no_texts_give_an_empty_matrix_of_the_right_width() -> None:
    assert FakeEmbedder(dimension=3).embed_documents([]).shape == (0, 3)


def test_a_query_gets_the_same_vector_as_a_document_with_the_same_text() -> None:
    embedder = FakeEmbedder()
    assert np.array_equal(embedder.embed_queries(["retrieval"]), embedder.embed_documents(["retrieval"]))


def test_the_vector_is_the_same_whether_the_text_is_embedded_alone_or_in_a_batch() -> None:
    embedder = FakeEmbedder()
    assert np.array_equal(embedder.embed_documents(["a", "b"])[1], embedder.embed_documents(["b"])[0])


def test_the_model_name_records_the_dimension() -> None:
    assert FakeEmbedder(dimension=16).model_name == "fake-16"


def test_a_dimension_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="dimension must be at least 1, got 0"):
        FakeEmbedder(dimension=0)
