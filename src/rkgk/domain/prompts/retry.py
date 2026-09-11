"""Builds the section both prompts append when the agent's previous answer was rejected.

The wording stays in Python rather than in a .md because it is two sentences wrapped around generated content;
a file per sentence would say less about the section than this module does.
"""

import json
from typing import Protocol


class PromptIssue(Protocol):
    """One complaint about a rejected answer, as the validation services report it.

    Read-only properties because the issue models are frozen.
    """

    @property
    def path(self) -> str: ...

    @property
    def message(self) -> str: ...


def build_retry_section(previous: object, issues: tuple[PromptIssue, ...]) -> list[str]:
    """Write the rejected JSON and every issue found in it, as the lines that follow the prompt's data.

    The agent gets its own answer back so it corrects that instead of starting over and losing the parts that
    were already right.
    """
    return [
        "",
        "# Previous attempt",
        "",
        "この JSON は却下された。",
        "",
        "```json",
        json.dumps(previous, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Issues",
        "",
        *(f"- {issue.path}: {issue.message}" for issue in issues),
        "",
        "上記のすべての問題を修正した、完全な JSON オブジェクトを返す。",
    ]
