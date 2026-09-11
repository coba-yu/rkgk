"""Embedder backed by Qwen3-Embedding through sentence-transformers.

The model is loaded on the first call, not in the constructor, so that a command which fails before it embeds
anything does not pay for the load.
sentence-transformers and torch are an optional extra (`embedding`) because they weigh far more than the rest of
the package and only the build and search commands need them.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rkgk.domain.embedders import EmbedderError

DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_BATCH_SIZE = 16
# The name of the instruction prompt that the Qwen3-Embedding model cards ship in their sentence-transformers
# configuration; documents are encoded without a prompt.
QUERY_PROMPT_NAME = "query"


class Qwen3Embedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return self._encode(texts, prompt_name=None)

    def embed_queries(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return self._encode(texts, prompt_name=QUERY_PROMPT_NAME)

    def _encode(self, texts: Sequence[str], prompt_name: str | None) -> NDArray[np.float32]:
        model = self._load()
        if not texts:
            return np.empty((0, model.get_embedding_dimension()), dtype=np.float32)
        try:
            vectors = model.encode(
                list(texts),
                prompt_name=prompt_name,
                batch_size=self._batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as error:
            raise EmbedderError(f"{self._model_name} failed to embed {len(texts)} texts: {error}") from error
        return np.asarray(vectors, dtype=np.float32)

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise EmbedderError(
                "sentence-transformers is not installed; install the `embedding` extra to use Qwen3Embedder"
            ) from error
        try:
            self._model = SentenceTransformer(self._model_name, device=self._device or _default_device())
        except Exception as error:
            raise EmbedderError(f"{self._model_name} could not be loaded: {error}") from error
        return self._model


def _default_device() -> str:
    # Apple silicon runs the model on the GPU through MPS; anywhere else falls back to the CPU rather than
    # assuming CUDA, because the design targets a local Mac.
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"
