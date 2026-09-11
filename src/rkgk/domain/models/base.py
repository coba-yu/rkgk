"""Base model shared by every entity in the domain.

Every model is frozen and forbids unknown fields so that a schema change surfaces as a validation error
instead of silently dropping data that a later pipeline stage expects.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

SLUG_PATTERN = r"^[a-z0-9]+(-[a-z0-9]+)*$"

Slug = Annotated[str, Field(pattern=SLUG_PATTERN)]


class Entity(BaseModel):
    """Sequences are tuples because frozen=True only blocks attribute assignment, not list mutation."""

    model_config = ConfigDict(frozen=True, extra="forbid")
