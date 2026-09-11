"""Ports for the agents the domain drives.

The domain says what an agent must answer, not which CLI or API produces the answer, so an implementation may
be the Claude Code CLI, an HTTP API, or a canned fake in a test.
"""

from typing import Protocol


class StructuredOutputAgent(Protocol):
    """An agent that answers a prompt with JSON that follows the given schema."""

    def answer(self, prompt: str, schema: dict[str, object]) -> object: ...


class StructuredOutputAgentError(Exception):
    """Raised when the agent could not be run or answered with something other than the requested JSON."""
