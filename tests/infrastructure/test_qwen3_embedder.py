from typing import Any

import numpy as np
import pytest

from rkgk.domain.embedders import EmbedderError
from rkgk.infrastructure.qwen3_embedder import Qwen3Embedder


class _RecordingModel:
    """Stands in for SentenceTransformer to capture how it is constructed and called."""

    instances: list["_RecordingModel"] = []

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.kwargs = kwargs
        self.encode_calls: list[dict[str, Any]] = []
        _RecordingModel.instances.append(self)

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        self.encode_calls.append(kwargs)
        return np.ones((len(texts), 4), dtype=np.float32)

    def get_embedding_dimension(self) -> int:
        return 4


@pytest.fixture
def recording_model(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingModel]:
    sentence_transformers = pytest.importorskip("sentence_transformers")
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _RecordingModel)
    _RecordingModel.instances.clear()
    return _RecordingModel


def test_the_model_name_is_reported_before_the_model_is_loaded() -> None:
    assert Qwen3Embedder().model_name == "Qwen/Qwen3-Embedding-0.6B"


def test_documents_are_encoded_four_at_a_time_by_default(recording_model: type[_RecordingModel]) -> None:
    Qwen3Embedder(device="cpu").embed_documents(["a", "b"])
    assert recording_model.instances[0].encode_calls[0]["batch_size"] == 4


def test_the_gpu_loads_the_model_in_half_precision(recording_model: type[_RecordingModel]) -> None:
    import torch

    Qwen3Embedder(device="mps").embed_documents(["a"])
    assert recording_model.instances[0].kwargs["model_kwargs"] == {"dtype": torch.float16}


def test_the_cpu_keeps_full_precision(recording_model: type[_RecordingModel]) -> None:
    import torch

    Qwen3Embedder(device="cpu").embed_documents(["a"])
    assert recording_model.instances[0].kwargs["model_kwargs"] == {"dtype": torch.float32}


def test_a_model_that_cannot_be_loaded_is_reported(tmp_path: str) -> None:
    pytest.importorskip("sentence_transformers")
    with pytest.raises(EmbedderError, match="could not be loaded"):
        Qwen3Embedder(model_name=str(tmp_path), device="cpu").embed_documents(["retrieval"])


@pytest.mark.slow
def test_the_real_model_ranks_the_relevant_document_first_for_a_japanese_query() -> None:
    pytest.importorskip("sentence_transformers")
    embedder = Qwen3Embedder()
    documents = embedder.embed_documents(
        [
            "Retrieval-augmented generation grounds a language model on passages fetched from a corpus.",
            "The recipe calls for two eggs, a cup of flour, and a pinch of salt.",
        ]
    )
    query = embedder.embed_queries(["検索で取得した文書を使って言語モデルの生成を補強する手法"])
    assert documents.shape == (2, 1024)
    assert query.shape == (1, 1024)
    assert documents.dtype == np.float32
    scores = documents @ query[0]
    assert scores[0] > scores[1]


@pytest.mark.slow
def test_the_real_model_embeds_nothing_into_an_empty_matrix_of_its_width() -> None:
    pytest.importorskip("sentence_transformers")
    assert Qwen3Embedder().embed_documents([]).shape == (0, 1024)
