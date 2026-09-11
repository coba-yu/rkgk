"""Base model and the field conventions shared by every entity in the domain.

Every model is frozen and forbids unknown fields so that a schema change surfaces as a validation error
instead of silently dropping data that a later pipeline stage expects.
Rendering the location of such a validation error lives here too, because every agent-facing step reports its
issues by the same path and an agent must read the same path in its own JSON.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

SLUG_PATTERN = r"^[a-z0-9]+(-[a-z0-9]+)*$"

Slug = Annotated[str, Field(pattern=SLUG_PATTERN)]


class Entity(BaseModel):
    """Sequences are tuples because frozen=True only blocks attribute assignment, not list mutation."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def format_error_path(location: tuple[int | str, ...]) -> str:
    """Render a pydantic error location the way an agent reads its own JSON: `concepts[2].merged_from[0]`."""
    parts: list[str] = []
    for item in location:
        if isinstance(item, int):
            parts.append(f"[{item}]")
        else:
            parts.append(f".{item}" if parts else item)
    return "".join(parts) or "<root>"
