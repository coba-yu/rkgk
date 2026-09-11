"""Embedder that derives a vector from a hash of the text.

It stands in for a real model in tests and local runs: the same text always gets the same vector, on any
machine, and different texts get vectors that are not related to their meaning.
An index built with it therefore proves that the pipeline is wired correctly, never that retrieval works.
"""

import hashlib
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

DEFAULT_DIMENSION = 8


class FakeEmbedder:
    def __init__(self, dimension: int = DEFAULT_DIMENSION) -> None:
        if dimension < 1:
            raise ValueError(f"dimension must be at least 1, got {dimension}")
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return f"fake-{self._dimension}"

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        return np.stack([self._vector(text) for text in texts])

    def embed_queries(self, texts: Sequence[str]) -> NDArray[np.float32]:
        # A fake has no instruction to prepend, so a query and a document with the same text share a vector.
        return self.embed_documents(texts)

    def _vector(self, text: str) -> NDArray[np.float32]:
        # sha256 rather than hash(): the latter is salted per process and would change the vectors between runs.
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8])
        vector = np.random.default_rng(seed).standard_normal(self._dimension).astype(np.float32)
        return vector / np.linalg.norm(vector)
