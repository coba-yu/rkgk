import copy
import json
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from rkgk.cli import extract
from rkgk.cli.extract import main
from rkgk.domain.agents import StructuredOutputAgentError

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


def read_output(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


def copy_fixture(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    shutil.copytree(FIXTURE_DIR, data_dir)
    return data_dir


class FakeAgent:
    """Stands in for the Claude CLI so the command tests never spawn an agent."""

    def __init__(self, *payloads: object) -> None:
        self._payloads = list(payloads)

    def answer(self, prompt: str, schema: dict[str, object]) -> object:
        return self._payloads.pop(0)


def install_agent(monkeypatch: pytest.MonkeyPatch, *payloads: object) -> None:
    monkeypatch.setattr(extract, "ClaudeCodeAgent", lambda model=None: FakeAgent(*payloads))


def install_failing_agent(monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    class FailingAgent:
        def answer(self, prompt: str, schema: dict[str, object]) -> object:
            raise StructuredOutputAgentError(message)

    monkeypatch.setattr(extract, "ClaudeCodeAgent", lambda model=None: FailingAgent())


def build_run_payload(quote: str) -> dict[str, Any]:
    payload = copy.deepcopy(VALID_EXTRACTION)
    payload["paper_concepts"][0]["evidence"][0]["quote"] = quote
    return payload


def test_run_extracts_the_paper_and_writes_the_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = copy_fixture(tmp_path)
    install_agent(monkeypatch, VALID_EXTRACTION)
    assert main(["1", "--data-dir", str(data_dir)]) == 0
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
    assert main(["1", "--data-dir", str(data_dir)]) == 0
    assert read_output(capsys)["attempts"] == 2


def test_run_reports_the_issues_when_the_agent_keeps_failing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = copy_fixture(tmp_path)
    rejected = build_run_payload("a sentence the paper never wrote")
    install_agent(monkeypatch, rejected, rejected)
    assert main(["1", "--data-dir", str(data_dir), "--max-attempts", "2"]) == 1
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
    assert main(["1", "--data-dir", str(data_dir)]) == 2
    output = read_output(capsys)
    assert output["status"] == "error"
    assert "claude was not found" in output["message"]


def test_run_reports_an_unknown_paper(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = copy_fixture(tmp_path)
    install_agent(monkeypatch, VALID_EXTRACTION)
    assert main(["9", "--data-dir", str(data_dir)]) == 2
    assert read_output(capsys)["status"] == "error"


def test_run_passes_the_chosen_model_to_the_agent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = copy_fixture(tmp_path)
    seen: list[str | None] = []

    def _build(model: str | None = None) -> FakeAgent:
        seen.append(model)
        return FakeAgent(VALID_EXTRACTION)

    monkeypatch.setattr(extract, "ClaudeCodeAgent", _build)
    assert main(["1", "--data-dir", str(data_dir), "--model", "claude-opus-4"]) == 0
    assert seen == ["claude-opus-4"]


def test_version_is_printed(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert version("rkgk") in capsys.readouterr().out


def test_no_arguments_reports_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main([])
    assert caught.value.code == 2


def test_help_mentions_the_paper_id(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    assert "paper_id" in capsys.readouterr().out
