import json
import subprocess
from pathlib import Path

import pytest

from rkgk.domain.models.paper_extraction import ExtractorError
from rkgk.infrastructure.claude_extractor import ClaudeExtractor

SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

SUCCESS = {"type": "result", "subtype": "success", "is_error": False, "structured_output": {"answer": "hello"}}


def write_fake_claude(tmp_path: Path, stdout: str, stderr: str = "", exit_code: int = 0) -> str:
    """A stand-in for the CLI that echoes canned output, so the tests never call the real agent."""
    stdout_path = tmp_path / "stdout.txt"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path = tmp_path / "stderr.txt"
    stderr_path.write_text(stderr, encoding="utf-8")
    path = tmp_path / "fake-claude"
    path.write_text(
        f'#!/bin/sh\ncat > "$0.stdin"\ncat "{stdout_path}"\ncat "{stderr_path}" >&2\nexit {exit_code}\n',
        encoding="utf-8",
    )
    path.chmod(0o755)
    return str(path)


def test_the_structured_output_is_returned_as_the_payload(tmp_path: Path) -> None:
    command = write_fake_claude(tmp_path, json.dumps(SUCCESS))
    assert ClaudeExtractor(command=command).answer("prompt", SCHEMA) == {"answer": "hello"}


def test_the_prompt_is_passed_on_standard_input(tmp_path: Path) -> None:
    command = write_fake_claude(tmp_path, json.dumps(SUCCESS))
    ClaudeExtractor(command=command).answer("a prompt", SCHEMA)
    assert Path(command + ".stdin").read_text(encoding="utf-8") == "a prompt"


def test_the_command_asks_for_json_output_against_the_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def _fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(SUCCESS), "")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    ClaudeExtractor().answer("prompt", SCHEMA)
    assert seen[0][:6] == ["claude", "-p", "--output-format", "json", "--json-schema", json.dumps(SCHEMA)]
    assert "--allowedTools" in seen[0]
    assert "--model" not in seen[0]


def test_a_model_is_passed_on_when_it_is_given(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def _fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(SUCCESS), "")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    ClaudeExtractor(model="claude-opus-4").answer("prompt", SCHEMA)
    assert seen[0][-2:] == ["--model", "claude-opus-4"]


def test_a_failing_command_is_reported_with_its_exit_code_and_stderr(tmp_path: Path) -> None:
    payload = {"type": "result", "subtype": "error_max_turns", "is_error": True, "structured_output": None}
    command = write_fake_claude(tmp_path, json.dumps(payload), stderr="ran out of turns", exit_code=1)
    with pytest.raises(ExtractorError, match="error_max_turns"):
        ClaudeExtractor(command=command).answer("prompt", SCHEMA)


def test_an_error_flag_without_a_failing_exit_code_is_still_a_failure(tmp_path: Path) -> None:
    payload = {"type": "result", "subtype": "success", "is_error": True, "structured_output": {"answer": "hello"}}
    command = write_fake_claude(tmp_path, json.dumps(payload))
    with pytest.raises(ExtractorError, match="reported a failure"):
        ClaudeExtractor(command=command).answer("prompt", SCHEMA)


def test_an_answer_without_structured_output_is_reported(tmp_path: Path) -> None:
    payload = {"type": "result", "subtype": "success", "is_error": False, "result": "here you go"}
    command = write_fake_claude(tmp_path, json.dumps(payload))
    with pytest.raises(ExtractorError, match="no structured output"):
        ClaudeExtractor(command=command).answer("prompt", SCHEMA)


def test_output_that_is_not_json_is_reported_with_an_excerpt(tmp_path: Path) -> None:
    command = write_fake_claude(tmp_path, "not json at all")
    with pytest.raises(ExtractorError, match="not JSON: not json at all"):
        ClaudeExtractor(command=command).answer("prompt", SCHEMA)


def test_output_that_is_a_json_array_is_reported(tmp_path: Path) -> None:
    command = write_fake_claude(tmp_path, "[1, 2]")
    with pytest.raises(ExtractorError, match="instead of an object"):
        ClaudeExtractor(command=command).answer("prompt", SCHEMA)


def test_a_missing_command_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ExtractorError, match="was not found"):
        ClaudeExtractor(command=str(tmp_path / "absent")).answer("prompt", SCHEMA)


def test_a_command_that_never_answers_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, 1.5)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    with pytest.raises(ExtractorError, match="did not answer within 1.5 seconds"):
        ClaudeExtractor(timeout_seconds=1.5).answer("prompt", SCHEMA)


@pytest.mark.slow
def test_the_real_claude_cli_answers_with_the_requested_structure() -> None:
    payload = ClaudeExtractor().answer("Answer with the word hello.", SCHEMA)
    assert isinstance(payload, dict)
    assert isinstance(payload["answer"], str)
