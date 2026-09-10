from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rkgk.domain.models.paper import MarkerKind, PaperMeta, PaperPreprocessInfo, parse_page

PREPROCESS = PaperPreprocessInfo(tool="pymupdf", version="1.24.0", processed_at=datetime(2026, 1, 1, tzinfo=UTC))


def make_paper_meta(paper_id: int) -> PaperMeta:
    return PaperMeta(
        id=paper_id,
        title="Retrieval Augmented Generation",
        authors=["Ada Lovelace"],
        year=2026,
        venue="NeurIPS",
        page_count=12,
        preprocess=PREPROCESS,
    )


@pytest.mark.parametrize(("paper_id", "expected"), [(1, "0001"), (42, "0042"), (9999, "9999"), (12345, "12345")])
def test_paper_meta_dir_name_is_zero_padded_without_truncation(paper_id: int, expected: str) -> None:
    assert make_paper_meta(paper_id).dir_name == expected


def test_paper_meta_rejects_id_below_one() -> None:
    with pytest.raises(ValidationError):
        make_paper_meta(0)


def test_parse_page_records_every_marker_in_line_order() -> None:
    page = parse_page(2, "intro\n<!-- figure: 3 -->\nabout figure 3\n<!-- equation: 12 -->\nabout equation 12\n")
    assert [(marker.kind, marker.number, marker.line) for marker in page.markers] == [
        (MarkerKind.FIGURE, 3, 2),
        (MarkerKind.EQUATION, 12, 4),
    ]


@pytest.mark.parametrize(
    "line",
    [
        "<!-- figure: x -->",
        "<!-- figure: 1 --> and more text",
        "text before <!-- figure: 1 -->",
        "<!--figure: 1-->",
        "<!-- table: 1 -->",
    ],
)
def test_parse_page_ignores_a_line_that_is_not_exactly_a_marker(line: str) -> None:
    assert parse_page(1, f"before\n{line}\nafter\n").markers == ()


def test_parse_page_returns_no_markers_for_plain_text() -> None:
    assert parse_page(1, "plain body text\nsecond line\n").markers == ()


def test_parse_page_keeps_the_text_unchanged() -> None:
    text = "before\n<!-- figure: 1 -->\nafter\n"
    assert parse_page(7, text).text == text


def test_parse_page_accepts_a_blank_page() -> None:
    page = parse_page(1, "")
    assert page.text == ""
    assert page.markers == ()
