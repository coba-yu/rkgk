"""Cuts a paper into the chunks that retrieval searches over.

This module reads the paper model and builds chunks; it owns no data and touches no storage.
A chunk never crosses a page boundary, so every hit can be quoted with the page it came from, which is what an
answer needs to cite evidence.
"""

from rkgk.domain.models.chunk import Chunk
from rkgk.domain.models.paper import MARKER_PATTERN, Paper
from rkgk.domain.tokenizers import Tokenizer

DEFAULT_MAX_TOKENS = 512

_BLOCK_SEPARATOR = "\n\n"


def _split_into_blocks(text: str) -> list[str]:
    """Cut a page into paragraphs and heading lines, dropping the preprocessor's markers.

    A heading ends the paragraph above it and stands alone even without a blank line around it, because it names
    the section the following paragraphs belong to rather than continuing the one before.
    Marker lines are dropped: they point at a figure or an equation and carry no text to retrieve, so keeping
    them would only add noise to the embedding of the chunk.
    """
    blocks: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            block = "\n".join(paragraph).strip()
            if block:
                blocks.append(block)
            paragraph.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if MARKER_PATTERN.match(stripped):
            continue
        if stripped.startswith("#"):
            flush()
            blocks.append(stripped)
            continue
        paragraph.append(line)
    flush()
    return blocks


def _split_oversized_block(block: str, tokenizer: Tokenizer, max_tokens: int) -> list[str]:
    """Pack the words of a block that is too long on its own into pieces that fit the budget."""
    pieces: list[str] = []
    words: list[str] = []
    for word in block.split():
        # A word alone over the budget is kept whole: a tokenizer that only counts cannot say where inside a
        # word the budget runs out, and a cut by characters would produce text that is in no page as written.
        if words and tokenizer.count_tokens(" ".join([*words, word])) > max_tokens:
            pieces.append(" ".join(words))
            words = [word]
            continue
        words.append(word)
    if words:
        pieces.append(" ".join(words))
    return pieces


def _fit_blocks(blocks: list[str], tokenizer: Tokenizer, max_tokens: int) -> list[str]:
    fitted: list[str] = []
    for block in blocks:
        if tokenizer.count_tokens(block) > max_tokens:
            fitted += _split_oversized_block(block, tokenizer, max_tokens)
            continue
        fitted.append(block)
    return fitted


def _pack_page(blocks: list[str], tokenizer: Tokenizer, max_tokens: int) -> list[str]:
    """Group the blocks of one page greedily, so a chunk holds as much context as the budget allows."""
    texts: list[str] = []
    current: list[str] = []
    for block in blocks:
        if current and tokenizer.count_tokens(_BLOCK_SEPARATOR.join([*current, block])) > max_tokens:
            texts.append(_BLOCK_SEPARATOR.join(current))
            current = [block]
            continue
        current.append(block)
    if current:
        texts.append(_BLOCK_SEPARATOR.join(current))
    return texts


def chunk_paper(paper: Paper, tokenizer: Tokenizer, max_tokens: int = DEFAULT_MAX_TOKENS) -> tuple[Chunk, ...]:
    """Cut a paper into chunks of at most `max_tokens` tokens, numbered from 0 across the whole paper."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be greater than or equal to 1")
    chunks: list[Chunk] = []
    for page in paper.pages:
        blocks = _fit_blocks(_split_into_blocks(page.text), tokenizer, max_tokens)
        for text in _pack_page(blocks, tokenizer, max_tokens):
            chunks.append(
                Chunk.create(
                    paper_id=paper.meta.id,
                    idx=len(chunks),
                    page_start=page.number,
                    page_end=page.number,
                    text=text,
                )
            )
    return tuple(chunks)
