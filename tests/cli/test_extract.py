import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from rkgk.cli import extract, main
from rkgk.domain.models.paper_extraction import ExtractorError

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"

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


def copy_fixture(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    shutil.copytree(FIXTURE_DIR, data_dir)
    return data_dir


def test_schema_prints_the_schema_of_an_extraction(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["extract", "schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert "summary_ja" in schema["properties"]


def test_schema_needs_no_data_directory(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["extract", "schema"]) == 0
    assert capsys.readouterr().out.startswith("{\n")


def test_an_extract_command_without_an_action_is_rejected() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["extract"])
    assert caught.value.code == 2


def test_validate_accepts_an_extraction_backed_by_the_paper(
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


def test_validate_reports_a_quote_that_is_not_in_the_paper(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    payload = copy.deepcopy(VALID_EXTRACTION)
    payload["paper_concepts"][0]["evidence"][0]["quote"] = "a sentence the paper never wrote"
    path = write_extraction(tmp_path, payload)
    assert main(["extract", "validate", "1", str(path), "--data-dir", str(FIXTURE_DIR)]) == 1
    output = read_output(capsys)
    assert output["status"] == "invalid"
    assert output["issues"] == [
        {"path": "paper_concepts[0].evidence[0].quote", "message": "is not found in the text of page 1"}
    ]


def test_validate_reports_a_payload_that_does_not_fit_the_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = copy.deepcopy(VALID_EXTRACTION)
    del payload["concepts"]
    path = write_extraction(tmp_path, payload)
    assert main(["extract", "validate", "1", str(path), "--data-dir", str(FIXTURE_DIR)]) == 1
    output = read_output(capsys)
    assert output["status"] == "invalid"
    assert "concepts" in [issue["path"] for issue in output["issues"]]


def test_validate_reports_an_unknown_paper_as_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    payload = copy.deepcopy(VALID_EXTRACTION)
    payload["paper_id"] = 9
    path = write_extraction(tmp_path, payload)
    assert main(["extract", "validate", "9", str(path), "--data-dir", str(FIXTURE_DIR)]) == 2
    output = read_output(capsys)
    assert output["status"] == "error"
    assert "paper 9" in output["message"]


def test_validate_reports_a_file_that_is_not_json_as_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "extraction.json"
    path.write_text("{not json", encoding="utf-8")
    assert main(["extract", "validate", "1", str(path), "--data-dir", str(FIXTURE_DIR)]) == 2
    assert read_output(capsys)["status"] == "error"


def test_validate_reports_a_missing_file_as_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "absent.json"
    assert main(["extract", "validate", "1", str(path), "--data-dir", str(FIXTURE_DIR)]) == 2
    assert read_output(capsys)["status"] == "error"


def test_save_writes_the_extraction_next_to_the_paper(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data_dir = copy_fixture(tmp_path)
    path = write_extraction(tmp_path, VALID_EXTRACTION)
    assert main(["extract", "save", "1", str(path), "--data-dir", str(data_dir)]) == 0
    written = data_dir / "papers" / "0001" / "extraction.json"
    assert read_output(capsys)["path"] == str(written)
    assert json.loads(written.read_text(encoding="utf-8"))["paper_id"] == 1


def test_save_writes_nothing_when_the_extraction_is_invalid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = copy_fixture(tmp_path)
    payload = copy.deepcopy(VALID_EXTRACTION)
    payload["paper_concepts"][0]["evidence"][0]["page"] = 9
    path = write_extraction(tmp_path, payload)
    assert main(["extract", "save", "1", str(path), "--data-dir", str(data_dir)]) == 1
    assert read_output(capsys)["status"] == "invalid"
    assert not (data_dir / "papers" / "0001" / "extraction.json").exists()


class FakeAgent:
    """Stands in for the Claude CLI so the command tests never spawn an agent."""

    def __init__(self, *payloads: object) -> None:
        self._payloads = list(payloads)

    def answer(self, prompt: str, schema: dict[str, object]) -> object:
        return self._payloads.pop(0)


def install_agent(monkeypatch: pytest.MonkeyPatch, *payloads: object) -> None:
    monkeypatch.setattr(extract, "ClaudeExtractor", lambda model=None: FakeAgent(*payloads))


def install_failing_agent(monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    class FailingAgent:
        def answer(self, prompt: str, schema: dict[str, object]) -> object:
            raise ExtractorError(message)

    monkeypatch.setattr(extract, "ClaudeExtractor", lambda model=None: FailingAgent())


def build_run_payload(quote: str) -> dict[str, Any]:
    payload = copy.deepcopy(VALID_EXTRACTION)
    payload["paper_concepts"][0]["evidence"][0]["quote"] = quote
    return payload


def test_run_extracts_the_paper_and_writes_the_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = copy_fixture(tmp_path)
    install_agent(monkeypatch, VALID_EXTRACTION)
    assert main(["extract", "run", "1", "--data-dir", str(data_dir)]) == 0
    written = data_dir / "papers" / "0001" / "extraction.json"
    output = read_output(capsys)
    assert output["status"] == "ok"
    assert output["attempts"] == 1
    assert output["concepts"] == 2
    assert output["path"] == str(written)
    assert json.loads(written.read_text(encoding="utf-8"))["paper_id"] == 1


def test_run_counts_the_attempt_the_agent_needed_to_correct_itself(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = copy_fixture(tmp_path)
    install_agent(monkeypatch, build_run_payload("a sentence the paper never wrote"), VALID_EXTRACTION)
    assert main(["extract", "run", "1", "--data-dir", str(data_dir)]) == 0
    assert read_output(capsys)["attempts"] == 2


def test_run_reports_the_issues_when_the_agent_keeps_failing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = copy_fixture(tmp_path)
    rejected = build_run_payload("a sentence the paper never wrote")
    install_agent(monkeypatch, rejected, rejected)
    assert main(["extract", "run", "1", "--data-dir", str(data_dir), "--max-attempts", "2"]) == 1
    output = read_output(capsys)
    assert output["status"] == "invalid"
    assert output["attempts"] == 2
    assert output["issues"][0]["path"] == "paper_concepts[0].evidence[0].quote"
    assert not (data_dir / "papers" / "0001" / "extraction.json").exists()


def test_run_reports_an_agent_that_cannot_be_started(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = copy_fixture(tmp_path)
    install_failing_agent(monkeypatch, "claude was not found, so no extraction can run")
    assert main(["extract", "run", "1", "--data-dir", str(data_dir)]) == 2
    output = read_output(capsys)
    assert output["status"] == "error"
    assert "claude was not found" in output["message"]


def test_run_reports_an_unknown_paper(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = copy_fixture(tmp_path)
    install_agent(monkeypatch, VALID_EXTRACTION)
    assert main(["extract", "run", "9", "--data-dir", str(data_dir)]) == 2
    assert read_output(capsys)["status"] == "error"


def test_run_passes_the_chosen_model_to_the_extractor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = copy_fixture(tmp_path)
    seen: list[str | None] = []

    def _build(model: str | None = None) -> FakeAgent:
        seen.append(model)
        return FakeAgent(VALID_EXTRACTION)

    monkeypatch.setattr(extract, "ClaudeExtractor", _build)
    assert main(["extract", "run", "1", "--data-dir", str(data_dir), "--model", "claude-opus-4"]) == 0
    assert seen == ["claude-opus-4"]
