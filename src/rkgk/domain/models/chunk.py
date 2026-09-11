"""A chunk of paper text kept for retrieval."""

from typing import Self

from pydantic import Field, field_validator, model_validator

from rkgk.domain.models.base import Entity

# The paper id, a colon, and the index of the chunk within the paper, both without leading zeros.
CHUNK_ID_PATTERN = r"^[1-9][0-9]*:(0|[1-9][0-9]*)$"


class Chunk(Entity):
    id: str = Field(pattern=CHUNK_ID_PATTERN)
    paper_id: int = Field(ge=1)
    idx: int = Field(ge=0)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str

    @field_validator("text")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain non-whitespace characters")
        return value

    @model_validator(mode="after")
    def _check_pages_and_id(self) -> Self:
        """The id is derived from paper_id and idx so a chunk read from disk is traceable without a lookup."""
        if self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        expected_id = self.build_id(self.paper_id, self.idx)
        if self.id != expected_id:
            raise ValueError(f"id must be {expected_id!r}")
        return self

    @staticmethod
    def build_id(paper_id: int, idx: int) -> str:
        return f"{paper_id}:{idx}"

    @classmethod
    def create(cls, paper_id: int, idx: int, page_start: int, page_end: int, text: str) -> Self:
        return cls(
            id=cls.build_id(paper_id, idx),
            paper_id=paper_id,
            idx=idx,
            page_start=page_start,
            page_end=page_end,
            text=text,
        )
