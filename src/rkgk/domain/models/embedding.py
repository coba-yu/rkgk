"""What gets embedded for retrieval, and the vectors that came out of it.

Retrieval hits an item, not a paper, so each item records the kind of thing it stands for and the id of that
thing: a chunk hit points at a page of a paper, a summary hit at a paper, and a concept hit at a node of the
graph that leads to several papers.
"""

from enum import StrEnum
from typing import Self

import numpy as np
from pydantic import ConfigDict, Field, field_validator, model_validator

from rkgk.domain.models.base import Entity


class EmbeddedItemKind(StrEnum):
    CHUNK = "chunk"
    SUMMARY = "summary"
    CONCEPT = "concept"


class EmbeddedItem(Entity):
    """One text that was embedded, with the id of the thing it stands for.

    `ref` is the chunk id, the paper id, or the concept slug depending on `kind`, so `(kind, ref)` identifies
    the item and the referenced thing can be looked up without parsing the text.
    """

    kind: EmbeddedItemKind
    ref: str = Field(min_length=1)
    paper_id: int | None = Field(default=None, ge=1)
    text: str

    @field_validator("text")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain non-whitespace characters")
        return value

    @model_validator(mode="after")
    def _check_ref_against_kind(self) -> Self:
        """A concept spans papers while a chunk or a summary belongs to exactly one, and `ref` must agree with it."""
        match self.kind:
            case EmbeddedItemKind.CONCEPT:
                if self.paper_id is not None:
                    raise ValueError("paper_id must be None when kind is concept")
            case EmbeddedItemKind.SUMMARY:
                if self.paper_id is None:
                    raise ValueError("paper_id is required when kind is summary")
                if self.ref != str(self.paper_id):
                    raise ValueError(f"ref must be {str(self.paper_id)!r} when kind is summary")
            case EmbeddedItemKind.CHUNK:
                if self.paper_id is None:
                    raise ValueError("paper_id is required when kind is chunk")
                if not self.ref.startswith(f"{self.paper_id}:"):
                    raise ValueError(f"ref must be a chunk id of paper {self.paper_id}")
        return self


class EmbeddingTable(Entity):
    """The embedded items and their vectors, row `i` of `vectors` belonging to `items[i]`.

    The pairing lives in one object so that a count mismatch between the two is caught where they meet, not in
    a search that returns the wrong paper.
    """

    # The vectors are a numpy array rather than nested tuples because retrieval does linear algebra on them, and
    # numpy is the one third-party type the domain accepts for that.
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    items: tuple[EmbeddedItem, ...]
    vectors: np.ndarray

    @model_validator(mode="after")
    def _check_rows_match_items(self) -> Self:
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors must be two-dimensional, got {self.vectors.ndim} dimensions")
        if self.vectors.shape[0] != len(self.items):
            raise ValueError(f"vectors has {self.vectors.shape[0]} rows for {len(self.items)} items")
        seen: set[tuple[EmbeddedItemKind, str]] = set()
        for item in self.items:
            key = (item.kind, item.ref)
            if key in seen:
                raise ValueError(f"items holds {item.kind.value} {item.ref!r} more than once")
            seen.add(key)
        return self

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])

    def __eq__(self, other: object) -> bool:
        # pydantic compares fields with `==`, which on an array yields an array and cannot become a bool.
        if not isinstance(other, EmbeddingTable):
            return NotImplemented
        return self.items == other.items and np.array_equal(self.vectors, other.vectors)
