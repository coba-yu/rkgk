import pytest
from pydantic import ValidationError

from rkgk.domain.models.chunk import Chunk


def test_chunk_make_builds_the_id() -> None:
    chunk = Chunk.create(paper_id=7, idx=3, page_start=2, page_end=3, text="body text")
    assert chunk.id == "7:3"


def test_chunk_rejects_page_end_before_page_start() -> None:
    with pytest.raises(ValidationError, match="page_end"):
        Chunk(id="1:0", paper_id=1, idx=0, page_start=5, page_end=4, text="body text")


def test_chunk_rejects_mismatched_id() -> None:
    with pytest.raises(ValidationError, match="id must be"):
        Chunk(id="2:0", paper_id=1, idx=0, page_start=1, page_end=1, text="body text")


def test_chunk_rejects_blank_text() -> None:
    with pytest.raises(ValidationError):
        Chunk(id="1:0", paper_id=1, idx=0, page_start=1, page_end=1, text="  ")
