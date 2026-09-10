import json
import shutil
from pathlib import Path

import pytest

from rkgk.loader import (
    LoaderError,
    MarkerKind,
    build_page_file_name,
    load_index,
    load_paper,
    parse_page,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A writable copy of the fixture, so a test can break one artifact without touching the others."""
    copy = tmp_path / "data"
    shutil.copytree(FIXTURE_DIR, copy)
    return copy


def write_index(data_dir: Path, text: str) -> None:
    (data_dir / "papers" / "index.json").write_text(text, encoding="utf-8")


def test_index_lists_every_entry_in_file_order() -> None:
    entries = load_index(FIXTURE_DIR)
    assert [(entry.id, entry.title) for entry in entries] == [
        (1, "Retrieval-Augmented Generation for Conference Paper Search"),
        (2, "Knowledge Graphs as Retrieval Backbones"),
    ]


def test_index_entry_may_have_no_artifacts_yet() -> None:
    assert load_index(FIXTURE_DIR)[1].id == 2
    with pytest.raises(LoaderError):
        load_paper(FIXTURE_DIR, 2)


def test_index_rejects_duplicate_ids(data_dir: Path) -> None:
    write_index(data_dir, json.dumps({"papers": [{"id": 1, "title": "A"}, {"id": 1, "title": "B"}]}))
    with pytest.raises(LoaderError, match="duplicate paper id 1"):
        load_index(data_dir)


def test_index_rejects_invalid_json(data_dir: Path) -> None:
    write_index(data_dir, "{not json")
    with pytest.raises(LoaderError, match="index.json"):
        load_index(data_dir)


def test_index_rejects_entry_without_title(data_dir: Path) -> None:
    write_index(data_dir, json.dumps({"papers": [{"id": 1}]}))
    with pytest.raises(LoaderError):
        load_index(data_dir)


def test_loaded_paper_has_meta_and_pages_in_ascending_order() -> None:
    paper = load_paper(FIXTURE_DIR, 1)
    assert paper.meta.id == 1
    assert paper.meta.page_count == 3
    assert [page.number for page in paper.pages] == [1, 2, 3]


def test_loaded_paper_records_figure_marker_with_its_line() -> None:
    page = load_paper(FIXTURE_DIR, 1).pages[1]
    assert [(marker.kind, marker.number, marker.line) for marker in page.markers] == [(MarkerKind.FIGURE, 1, 5)]


def test_loaded_paper_records_equation_marker_with_its_line() -> None:
    page = load_paper(FIXTURE_DIR, 1).pages[2]
    assert [(marker.kind, marker.number, marker.line) for marker in page.markers] == [(MarkerKind.EQUATION, 1, 4)]


def test_loaded_page_text_keeps_the_marker_lines() -> None:
    pages = load_paper(FIXTURE_DIR, 1).pages
    assert "<!-- figure: 1 -->" in pages[1].text
    assert "<!-- equation: 1 -->" in pages[2].text


def test_first_page_has_no_markers() -> None:
    assert load_paper(FIXTURE_DIR, 1).pages[0].markers == ()


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


def test_missing_paper_directory_is_reported(data_dir: Path) -> None:
    shutil.rmtree(data_dir / "papers" / "0001")
    with pytest.raises(LoaderError, match="paper 1"):
        load_paper(data_dir, 1)


def test_missing_paper_json_is_reported(data_dir: Path) -> None:
    (data_dir / "papers" / "0001" / "paper.json").unlink()
    with pytest.raises(LoaderError, match="paper.json"):
        load_paper(data_dir, 1)


def test_paper_json_with_a_different_id_is_reported(data_dir: Path) -> None:
    path = data_dir / "papers" / "0001" / "paper.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["id"] = 7
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LoaderError, match="does not match"):
        load_paper(data_dir, 1)


def test_missing_pages_directory_is_reported(data_dir: Path) -> None:
    shutil.rmtree(data_dir / "papers" / "0001" / "pages")
    with pytest.raises(LoaderError, match="pages"):
        load_paper(data_dir, 1)


def test_missing_page_file_is_reported_by_name(data_dir: Path) -> None:
    (data_dir / "papers" / "0001" / "pages" / "002.md").unlink()
    with pytest.raises(LoaderError, match="002.md"):
        load_paper(data_dir, 1)


def test_page_file_beyond_page_count_is_reported_by_name(data_dir: Path) -> None:
    (data_dir / "papers" / "0001" / "pages" / "004.md").write_text("extra page\n", encoding="utf-8")
    with pytest.raises(LoaderError, match="004.md"):
        load_paper(data_dir, 1)


def test_stray_file_in_pages_is_reported_by_name(data_dir: Path) -> None:
    (data_dir / "papers" / "0001" / "pages" / "notes.txt").write_text("scratch\n", encoding="utf-8")
    with pytest.raises(LoaderError, match="notes.txt"):
        load_paper(data_dir, 1)


def test_page_count_larger_than_the_page_files_is_reported(data_dir: Path) -> None:
    path = data_dir / "papers" / "0001" / "paper.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["page_count"] = 4
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LoaderError, match="004.md"):
        load_paper(data_dir, 1)


@pytest.mark.parametrize(("number", "expected"), [(1, "001.md"), (12, "012.md"), (999, "999.md"), (1000, "1000.md")])
def test_page_file_name_is_zero_padded_without_truncation(number: int, expected: str) -> None:
    assert build_page_file_name(number) == expected
