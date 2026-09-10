"""Paper text as the preprocessing project hands it over.

A paper is its metadata plus the pages in ascending order, and a page keeps the Markdown text of the preprocessor
together with the markers found in it.
The marker format is part of the handoff contract, so parsing it belongs to the domain rather than to a reader.
"""

import enum
import re

from pydantic import Field

from rkgk.domain.entities import Entity, PaperMeta

# The preprocessor writes these markers on their own line, so anything else on the line is body text.
MARKER_PATTERN = re.compile(r"^<!-- (figure|equation): (\d+) -->$")


class MarkerKind(enum.StrEnum):
    FIGURE = "figure"
    EQUATION = "equation"


class PaperIndexEntry(Entity):
    id: int = Field(ge=1)
    title: str = Field(min_length=1)


class PageMarker(Entity):
    kind: MarkerKind
    number: int = Field(ge=1)
    line: int = Field(ge=1)


class Page(Entity):
    number: int = Field(ge=1)
    text: str
    markers: tuple[PageMarker, ...] = ()


class Paper(Entity):
    meta: PaperMeta
    pages: tuple[Page, ...]


def parse_page(number: int, text: str) -> Page:
    """Collect the markers of one page, keeping the marker lines in the text for later stages to skip or use."""
    markers: list[PageMarker] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        matched = MARKER_PATTERN.match(line.strip())
        if matched is None:
            continue
        kind, marker_number = matched.groups()
        markers.append(PageMarker(kind=MarkerKind(kind), number=int(marker_number), line=line_number))
    return Page(number=number, text=text, markers=tuple(markers))
