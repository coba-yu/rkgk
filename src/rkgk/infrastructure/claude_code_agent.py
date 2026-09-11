"""Structured-output agent backed by the Claude Code CLI running headless.

`claude -p --output-format json --json-schema <schema>` reads the prompt from stdin and prints one JSON object
on stdout whose keys are, as observed with version 2.1.259:
`type`, `subtype`, `is_error`, `result`, `structured_output`, `session_id`, `num_turns`, `stop_reason`,
`terminal_reason`, `usage`, `modelUsage`, `total_cost_usd`, `duration_ms`, `duration_api_ms`, `uuid`,
`permission_denials`, `queued_turn_count`, `api_error_status`, `fast_mode_state`, `fast_mode_disabled_reason`,
`subagent_stats`, `time_to_request_ms`, `ttft_ms`, `ttft_stream_ms`.
A successful run exits 0 with `type: "result"`, `subtype: "success"`, `is_error: false`, and the payload that
follows the schema under `structured_output`.
A run that hits the turn limit exits 1 with `is_error: true`, `subtype: "error_max_turns"` and
`structured_output: null`, so the structured output is the only field worth reading.
"""

import json
import subprocess

from rkgk.domain.agents import StructuredOutputAgentError

# The structured answer costs a turn of its own on top of the assistant turn, so one turn can never succeed.
MAX_TURNS = "2"

_EXCERPT_LIMIT = 500


def _excerpt(text: str) -> str:
    stripped = text.strip()
    if len(stripped) <= _EXCERPT_LIMIT:
        return stripped
    return stripped[:_EXCERPT_LIMIT] + "..."


class ClaudeCodeAgent:
    def __init__(self, model: str | None = None, timeout_seconds: float = 600.0, command: str = "claude") -> None:
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._command = command

    def answer(self, prompt: str, schema: dict[str, object]) -> object:
        completed = self._run(prompt, schema)
        payload = self._parse(completed)
        if completed.returncode != 0 or payload.get("is_error") or payload.get("subtype") != "success":
            raise StructuredOutputAgentError(
                f"{self._command} reported a failure: exit code {completed.returncode}, "
                f"subtype {payload.get('subtype')!r}{self._stderr_note(completed.stderr)}"
            )
        structured_output = payload.get("structured_output")
        if structured_output is None:
            raise StructuredOutputAgentError(
                f"{self._command} returned no structured output{self._stderr_note(completed.stderr)}"
            )
        return structured_output

    def _build_command(self, schema: dict[str, object]) -> list[str]:
        command = [
            self._command,
            "-p",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema),
            # The paper text travels inside the prompt, so the model needs no tool; disabling them all also keeps
            # instructions hidden in a paper from reaching the file system through Read, Bash or an MCP server.
            "--tools",
            "",
            "--strict-mcp-config",
            "--max-turns",
            MAX_TURNS,
        ]
        if self._model is not None:
            command += ["--model", self._model]
        return command

    def _run(self, prompt: str, schema: dict[str, object]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                self._build_command(schema),
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self._timeout_seconds,
            )
        except FileNotFoundError as error:
            raise StructuredOutputAgentError(f"{self._command} was not found, so no extraction can run") from error
        except subprocess.TimeoutExpired as error:
            raise StructuredOutputAgentError(
                f"{self._command} did not answer within {self._timeout_seconds} seconds"
            ) from error

    def _parse(self, completed: subprocess.CompletedProcess[str]) -> dict[str, object]:
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise StructuredOutputAgentError(
                f"{self._command} exited with code {completed.returncode} and printed output that is not JSON: "
                f"{_excerpt(completed.stdout)}{self._stderr_note(completed.stderr)}"
            ) from error
        if not isinstance(payload, dict):
            raise StructuredOutputAgentError(
                f"{self._command} printed a JSON {type(payload).__name__} instead of an object: "
                f"{_excerpt(completed.stdout)}"
            )
        return payload

    def _stderr_note(self, stderr: str) -> str:
        return f"; stderr: {_excerpt(stderr)}" if stderr.strip() else ""
