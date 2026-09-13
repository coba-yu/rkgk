"""Embedder backed by Qwen3-Embedding through sentence-transformers.

The model is loaded on the first call, not in the constructor, so that a command which fails before it embeds
anything does not pay for the load.
sentence-transformers and torch are an optional extra (`embedding`) because they weigh far more than the rest of
the package and only the build and search commands need them.
The defaults are chosen for the memory of a 16 GB Mac rather than for throughput: on MPS the PyTorch allocator
keeps the working buffers of the largest batch, and with batches of 16 chunks of up to 512 words that reached
13 GB of GPU memory and pushed the rest of the system into swap (2026-09-13, 2,047 texts).
Batches of 4 in half precision peaked at 5.7 GB for the same texts, at the same speed, because the model is
compute-bound on the GPU either way.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rkgk.domain.embedders import EmbedderError

DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_BATCH_SIZE = 4
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
        device = self._device or _default_device()
        try:
            self._model = SentenceTransformer(
                self._model_name, device=device, model_kwargs={"dtype": _default_dtype(device)}
            )
        except Exception as error:
            raise EmbedderError(f"{self._model_name} could not be loaded: {error}") from error
        return self._model


def _default_device() -> str:
    # Apple silicon runs the model on the GPU through MPS; anywhere else falls back to the CPU rather than
    # assuming CUDA, because the design targets a local Mac.
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _default_dtype(device: str) -> Any:
    # Half precision halves the weights and the working buffers on the GPU and matches the full-precision vectors
    # to four decimals of cosine similarity; the CPU keeps full precision because its half-precision kernels are
    # the slow path.
    import torch

    # torch.device parses an indexed spelling such as "cpu:0" down to its type, which a string comparison would
    # miss and send the CPU through half precision.
    return torch.float32 if torch.device(device).type == "cpu" else torch.float16
