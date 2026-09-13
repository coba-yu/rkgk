"""Cuts a paper into the chunks that retrieval searches over.

This module reads the paper model and builds chunks; it owns no data and touches no storage.
A chunk never crosses a page boundary, so every hit can be quoted with the page it came from, which is what an
answer needs to cite evidence.
The References section is left out of the chunks: it is a list of other papers' titles, so it matches a query
about any of them while saying nothing about the paper that cites them.
"""

import re

from rkgk.domain.models.chunk import Chunk
from rkgk.domain.models.paper import MARKER_PATTERN, Page, Paper, PaperPosition, ReferencesSpan
from rkgk.domain.tokenizers import Tokenizer

DEFAULT_MAX_TOKENS = 512

_BLOCK_SEPARATOR = "\n\n"

# The preprocessor writes the heading with the emphasis the PDF had, so `## **References**` and `### References`
# are the same heading; the trailing `[\s*]*` also absorbs the space some pages leave after the title.
_REFERENCES_HEADING_PATTERN = re.compile(r"^(#+)[\s*]*(?:references|bibliography)[\s*]*$", re.IGNORECASE)

_HEADING_LEVEL_PATTERN = re.compile(r"^(#+)")


def _read_heading_level(line: str) -> int | None:
    """Count the leading `#` of a heading line, or return None when the line is not a heading."""
    matched = _HEADING_LEVEL_PATTERN.match(line)
    if matched is None:
        return None
    return len(matched.group(1))


def locate_references(paper: Paper) -> ReferencesSpan | None:
    """Find the first References heading of the paper and the heading that closes the section, if any.

    The section ends at the next heading of the same or a shallower level, which is how a paper resumes with an
    appendix; a deeper heading belongs to the references themselves and is walked over.
    """
    start: PaperPosition | None = None
    start_level = 0
    for page in paper.pages:
        for line_number, line in enumerate(page.text.splitlines(), start=1):
            stripped = line.strip()
            if start is None:
                matched = _REFERENCES_HEADING_PATTERN.match(stripped)
                if matched is not None:
                    start = PaperPosition(page=page.number, line=line_number)
                    start_level = len(matched.group(1))
                continue
            level = _read_heading_level(stripped)
            if level is not None and level <= start_level:
                return ReferencesSpan(start=start, end=PaperPosition(page=page.number, line=line_number))
    if start is None:
        return None
    return ReferencesSpan(start=start)


def _drop_references(page: Page, span: ReferencesSpan | None) -> str:
    """Cut the lines of one page that fall inside the References section, keeping the rest in order."""
    if span is None:
        return page.text
    lines = page.text.splitlines()
    if page.number < span.start.page:
        return page.text
    if span.end is not None and page.number > span.end.page:
        return page.text
    # A page fully inside the section keeps nothing; on the page the section starts or ends on, only the lines
    # from the heading up to the closing heading go, so the body around them is still chunked.
    first = span.start.line if page.number == span.start.page else 1
    last = span.end.line if span.end is not None and page.number == span.end.page else len(lines) + 1
    kept = [line for number, line in enumerate(lines, start=1) if number < first or number >= last]
    return "\n".join(kept)


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
    """Cut a paper into chunks of at most `max_tokens` tokens, numbered from 0 across the whole paper.

    The References section is located over the whole paper first, because it runs across pages, and its lines
    are cut before a page is split into blocks; a page left with no text yields no chunk.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be greater than or equal to 1")
    span = locate_references(paper)
    chunks: list[Chunk] = []
    for page in paper.pages:
        blocks = _fit_blocks(_split_into_blocks(_drop_references(page, span)), tokenizer, max_tokens)
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
