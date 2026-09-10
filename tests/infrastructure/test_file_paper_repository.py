import json
import shutil
from pathlib import Path

import pytest

from rkgk.domain.paper import MarkerKind
from rkgk.domain.repositories import PaperRepositoryError
from rkgk.infrastructure.file_paper_repository import FilePaperRepository, build_page_file_name

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A writable copy of the fixture, so a test can break one artifact without touching the others."""
    copy = tmp_path / "data"
    shutil.copytree(FIXTURE_DIR, copy)
    return copy


def write_index(data_dir: Path, text: str) -> None:
    (data_dir / "papers" / "index.json").write_text(text, encoding="utf-8")


def test_index_lists_every_entry_in_file_order() -> None:
    entries = FilePaperRepository(FIXTURE_DIR).find_index()
    assert [(entry.id, entry.title) for entry in entries] == [
        (1, "Retrieval-Augmented Generation for Conference Paper Search"),
        (2, "Knowledge Graphs as Retrieval Backbones"),
    ]


def test_index_entry_may_have_no_artifacts_yet() -> None:
    repository = FilePaperRepository(FIXTURE_DIR)
    assert repository.find_index()[1].id == 2
    with pytest.raises(PaperRepositoryError):
        repository.find(2)


def test_index_rejects_duplicate_ids(data_dir: Path) -> None:
    write_index(data_dir, json.dumps({"papers": [{"id": 1, "title": "A"}, {"id": 1, "title": "B"}]}))
    with pytest.raises(PaperRepositoryError, match="duplicate paper id 1"):
        FilePaperRepository(data_dir).find_index()


def test_index_rejects_invalid_json(data_dir: Path) -> None:
    write_index(data_dir, "{not json")
    with pytest.raises(PaperRepositoryError, match="index.json"):
        FilePaperRepository(data_dir).find_index()


def test_index_rejects_entry_without_title(data_dir: Path) -> None:
    write_index(data_dir, json.dumps({"papers": [{"id": 1}]}))
    with pytest.raises(PaperRepositoryError):
        FilePaperRepository(data_dir).find_index()


def test_loaded_paper_has_meta_and_pages_in_ascending_order() -> None:
    paper = FilePaperRepository(FIXTURE_DIR).find(1)
    assert paper.meta.id == 1
    assert paper.meta.page_count == 3
    assert [page.number for page in paper.pages] == [1, 2, 3]


def test_loaded_paper_records_figure_marker_with_its_line() -> None:
    page = FilePaperRepository(FIXTURE_DIR).find(1).pages[1]
    assert [(marker.kind, marker.number, marker.line) for marker in page.markers] == [(MarkerKind.FIGURE, 1, 5)]


def test_loaded_paper_records_equation_marker_with_its_line() -> None:
    page = FilePaperRepository(FIXTURE_DIR).find(1).pages[2]
    assert [(marker.kind, marker.number, marker.line) for marker in page.markers] == [(MarkerKind.EQUATION, 1, 4)]


def test_loaded_page_text_keeps_the_marker_lines() -> None:
    pages = FilePaperRepository(FIXTURE_DIR).find(1).pages
    assert "<!-- figure: 1 -->" in pages[1].text
    assert "<!-- equation: 1 -->" in pages[2].text


def test_first_page_has_no_markers() -> None:
    assert FilePaperRepository(FIXTURE_DIR).find(1).pages[0].markers == ()


def test_missing_paper_directory_is_reported(data_dir: Path) -> None:
    shutil.rmtree(data_dir / "papers" / "0001")
    with pytest.raises(PaperRepositoryError, match="paper 1"):
        FilePaperRepository(data_dir).find(1)


def test_missing_paper_json_is_reported(data_dir: Path) -> None:
    (data_dir / "papers" / "0001" / "paper.json").unlink()
    with pytest.raises(PaperRepositoryError, match="paper.json"):
        FilePaperRepository(data_dir).find(1)


def test_paper_json_with_a_different_id_is_reported(data_dir: Path) -> None:
    path = data_dir / "papers" / "0001" / "paper.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["id"] = 7
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PaperRepositoryError, match="does not match"):
        FilePaperRepository(data_dir).find(1)


def test_missing_pages_directory_is_reported(data_dir: Path) -> None:
    shutil.rmtree(data_dir / "papers" / "0001" / "pages")
    with pytest.raises(PaperRepositoryError, match="pages"):
        FilePaperRepository(data_dir).find(1)


def test_missing_page_file_is_reported_by_name(data_dir: Path) -> None:
    (data_dir / "papers" / "0001" / "pages" / "002.md").unlink()
    with pytest.raises(PaperRepositoryError, match="002.md"):
        FilePaperRepository(data_dir).find(1)


def test_page_file_beyond_page_count_is_reported_by_name(data_dir: Path) -> None:
    (data_dir / "papers" / "0001" / "pages" / "004.md").write_text("extra page\n", encoding="utf-8")
    with pytest.raises(PaperRepositoryError, match="004.md"):
        FilePaperRepository(data_dir).find(1)


def test_stray_file_in_pages_is_reported_by_name(data_dir: Path) -> None:
    (data_dir / "papers" / "0001" / "pages" / "notes.txt").write_text("scratch\n", encoding="utf-8")
    with pytest.raises(PaperRepositoryError, match="notes.txt"):
        FilePaperRepository(data_dir).find(1)


def test_page_count_larger_than_the_page_files_is_reported(data_dir: Path) -> None:
    path = data_dir / "papers" / "0001" / "paper.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["page_count"] = 4
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PaperRepositoryError, match="004.md"):
        FilePaperRepository(data_dir).find(1)


@pytest.mark.parametrize(("number", "expected"), [(1, "001.md"), (12, "012.md"), (999, "999.md"), (1000, "1000.md")])
def test_page_file_name_is_zero_padded_without_truncation(number: int, expected: str) -> None:
    assert build_page_file_name(number) == expected
