import argparse
import copy
import json
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from rkgk.cli import _COMMANDS, main, register_command


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    assert version("rkgk") in capsys.readouterr().out


def test_no_subcommand_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out


def test_registered_command_is_callable() -> None:
    before = list(_COMMANDS)

    def _register(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("name")
        subparser.set_defaults(func=lambda args: 0 if args.name == "ok" else 1)

    register_command("dummy", "a dummy command for testing")(_register)
    try:
        assert main(["dummy", "ok"]) == 0
        assert main(["dummy", "nope"]) == 1
    finally:
        _COMMANDS[:] = before


FIXTURE_DIR = Path(__file__).parent / "fixtures"

VALID_EXTRACTION: dict[str, Any] = {
    "schema_version": 1,
    "paper_id": 1,
    "summary_ja": "この論文は会議論文検索のための検索拡張生成パイプラインを提案する。",
    "concepts": [
        {"local_id": "c1", "name": "Retrieval-Augmented Generation", "type": "method"},
        {"local_id": "c2", "name": "Page-Aligned Chunking", "type": "method"},
    ],
    "paper_concepts": [
        {
            "concept_id": "c1",
            "relation": "proposes",
            "evidence": [{"page": 1, "quote": "retrieval-augmented generation pipeline"}],
        },
        {
            "concept_id": "c2",
            "relation": "uses",
            "evidence": [{"page": 2, "quote": "splits every paper into page-aligned chunks"}],
        },
    ],
    "concept_relations": [
        {
            "source_id": "c2",
            "target_id": "c1",
            "relation": "part_of",
            "evidence": [{"page": 2, "quote": "The pipeline splits every paper into page-aligned chunks"}],
        }
    ],
}


def write_extraction(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "extraction.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def read_output(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


def test_schema_extraction_prints_the_schema_of_the_result(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["schema", "extraction"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert "summary_ja" in schema["properties"]


def test_extract_validate_accepts_an_extraction_backed_by_the_paper(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_extraction(tmp_path, VALID_EXTRACTION)
    assert main(["extract", "validate", "1", str(path), "--data-dir", str(FIXTURE_DIR)]) == 0
    assert read_output(capsys) == {
        "status": "ok",
        "paper_id": 1,
        "concepts": 2,
        "paper_concepts": 2,
        "concept_relations": 1,
    }


def test_extract_validate_reports_a_quote_that_is_not_in_the_paper(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = copy.deepcopy(VALID_EXTRACTION)
    payload["paper_concepts"][0]["evidence"][0]["quote"] = "a sentence the paper never wrote"
    path = write_extraction(tmp_path, payload)
    assert main(["extract", "validate", "1", str(path), "--data-dir", str(FIXTURE_DIR)]) == 1
    output = read_output(capsys)
    assert output["status"] == "invalid"
    assert output["issues"] == [
        {"path": "paper_concepts[0].evidence[0].quote", "message": "is not found in the text of page 1"}
    ]


def test_extract_validate_reports_a_payload_that_does_not_fit_the_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = copy.deepcopy(VALID_EXTRACTION)
    del payload["concepts"]
    path = write_extraction(tmp_path, payload)
    assert main(["extract", "validate", "1", str(path), "--data-dir", str(FIXTURE_DIR)]) == 1
    output = read_output(capsys)
    assert output["status"] == "invalid"
    assert "concepts" in [issue["path"] for issue in output["issues"]]


def test_extract_validate_reports_an_unknown_paper_as_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = copy.deepcopy(VALID_EXTRACTION)
    payload["paper_id"] = 9
    path = write_extraction(tmp_path, payload)
    assert main(["extract", "validate", "9", str(path), "--data-dir", str(FIXTURE_DIR)]) == 2
    output = read_output(capsys)
    assert output["status"] == "error"
    assert "paper 9" in output["message"]


def test_extract_validate_reports_a_file_that_is_not_json_as_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "extraction.json"
    path.write_text("{not json", encoding="utf-8")
    assert main(["extract", "validate", "1", str(path), "--data-dir", str(FIXTURE_DIR)]) == 2
    assert read_output(capsys)["status"] == "error"


def test_extract_validate_reports_a_missing_file_as_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "absent.json"
    assert main(["extract", "validate", "1", str(path), "--data-dir", str(FIXTURE_DIR)]) == 2
    assert read_output(capsys)["status"] == "error"


def test_extract_save_writes_the_extraction_next_to_the_paper(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "data"
    shutil.copytree(FIXTURE_DIR, data_dir)
    path = write_extraction(tmp_path, VALID_EXTRACTION)
    assert main(["extract", "save", "1", str(path), "--data-dir", str(data_dir)]) == 0
    written = data_dir / "papers" / "0001" / "extraction.json"
    assert read_output(capsys)["path"] == str(written)
    assert json.loads(written.read_text(encoding="utf-8"))["paper_id"] == 1


def test_extract_save_writes_nothing_when_the_extraction_is_invalid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "data"
    shutil.copytree(FIXTURE_DIR, data_dir)
    payload = copy.deepcopy(VALID_EXTRACTION)
    payload["paper_concepts"][0]["evidence"][0]["page"] = 9
    path = write_extraction(tmp_path, payload)
    assert main(["extract", "save", "1", str(path), "--data-dir", str(data_dir)]) == 1
    assert read_output(capsys)["status"] == "invalid"
    assert not (data_dir / "papers" / "0001" / "extraction.json").exists()
