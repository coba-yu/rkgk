"""Port for the text embedding the domain needs.

Retrieval compares vectors, so the domain has to turn texts into vectors, but it must not know which model or
device does it: the real runs use Qwen3 on sentence-transformers, while the tests use deterministic fake vectors.
Documents and queries are embedded through two methods because instruction-tuned models such as Qwen3 prepend
an instruction to a query and nothing to a document, and the caller is the only one who knows which is which.
"""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Embedder(Protocol):
    """Something that can turn texts into one vector per text, all of the same dimension."""

    @property
    def model_name(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]: ...

    def embed_queries(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


class EmbedderError(Exception):
    """Raised when the embedder could not be set up or could not produce vectors."""
