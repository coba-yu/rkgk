import pytest

from rkgk.domain.models.chunk import Chunk
from rkgk.domain.services.chunking import DEFAULT_MAX_TOKENS, chunk_paper
from rkgk.infrastructure.whitespace_tokenizer import WhitespaceTokenizer
from tests.usecase.fakes import build_paper

TOKENIZER = WhitespaceTokenizer()


class CharacterTokenizer:
    """Counts characters, so that a test can make a single word exceed the budget."""

    def count_tokens(self, text: str) -> int:
        return len(text)


def texts_of(chunks: tuple[Chunk, ...]) -> list[str]:
    return [chunk.text for chunk in chunks]


def test_chunk_paper_joins_paragraphs_that_fit_together() -> None:
    paper = build_paper(1, "The pipeline reads a paper.\n\nIt then answers a question.\n")

    chunks = chunk_paper(paper, TOKENIZER)

    assert texts_of(chunks) == ["The pipeline reads a paper.\n\nIt then answers a question."]


def test_chunk_paper_never_joins_two_pages() -> None:
    paper = build_paper(1, "The pipeline reads a paper.\n", "It then answers a question.\n")

    chunks = chunk_paper(paper, TOKENIZER)

    assert texts_of(chunks) == ["The pipeline reads a paper.", "It then answers a question."]


def test_chunk_paper_records_the_page_a_chunk_came_from() -> None:
    paper = build_paper(1, "Alpha beta gamma delta.\n", "Epsilon zeta eta theta.\n")

    chunks = chunk_paper(paper, TOKENIZER, max_tokens=2)

    assert [(chunk.page_start, chunk.page_end) for chunk in chunks] == [(1, 1), (1, 1), (2, 2), (2, 2)]


def test_chunk_paper_starts_a_new_block_at_a_heading_without_blank_lines() -> None:
    paper = build_paper(1, "## Method\nWe train the model on paired data.\n")

    chunks = chunk_paper(paper, TOKENIZER, max_tokens=7)

    assert texts_of(chunks) == ["## Method", "We train the model on paired data."]


def test_chunk_paper_splits_a_paragraph_that_exceeds_the_budget() -> None:
    words = [f"word{index}" for index in range(10)]
    paper = build_paper(1, " ".join(words))

    chunks = chunk_paper(paper, TOKENIZER, max_tokens=4)

    assert len(chunks) == 3
    assert all(TOKENIZER.count_tokens(chunk.text) <= 4 for chunk in chunks)
    assert " ".join(texts_of(chunks)) == " ".join(words)


def test_chunk_paper_keeps_a_word_longer_than_the_budget_as_its_own_chunk() -> None:
    paper = build_paper(1, "alpha supercalifragilistic beta")

    chunks = chunk_paper(paper, CharacterTokenizer(), max_tokens=10)

    assert texts_of(chunks) == ["alpha", "supercalifragilistic", "beta"]


def test_chunk_paper_leaves_marker_lines_out_of_the_text() -> None:
    paper = build_paper(1, "We rely on the figure.\n<!-- figure: 1 -->\nIt shows the pipeline.\n")

    chunks = chunk_paper(paper, TOKENIZER)

    assert texts_of(chunks) == ["We rely on the figure.\nIt shows the pipeline."]


def test_chunk_paper_skips_a_page_that_holds_only_markers_and_blank_lines() -> None:
    paper = build_paper(1, "The first page.\n", "<!-- equation: 2 -->\n\n   \n", "The third page.\n")

    chunks = chunk_paper(paper, TOKENIZER)

    assert [(chunk.id, chunk.page_start) for chunk in chunks] == [("1:0", 1), ("1:1", 3)]


def test_chunk_paper_numbers_chunks_across_the_whole_paper() -> None:
    paper = build_paper(7, "Alpha beta gamma delta.\n", "Epsilon zeta eta theta.\n")

    chunks = chunk_paper(paper, TOKENIZER, max_tokens=2)

    assert [chunk.idx for chunk in chunks] == [0, 1, 2, 3]
    assert [chunk.id for chunk in chunks] == ["7:0", "7:1", "7:2", "7:3"]


def test_chunk_paper_rejects_a_budget_below_one_token() -> None:
    paper = build_paper(1, "The first page.\n")

    with pytest.raises(ValueError, match="max_tokens"):
        chunk_paper(paper, TOKENIZER, max_tokens=0)


def test_chunk_paper_defaults_to_512_tokens() -> None:
    assert DEFAULT_MAX_TOKENS == 512
    fitting = build_paper(1, " ".join(f"word{index}" for index in range(512)))
    overflowing = build_paper(1, " ".join(f"word{index}" for index in range(513)))

    assert len(chunk_paper(fitting, TOKENIZER)) == 1
    assert len(chunk_paper(overflowing, TOKENIZER)) == 2
