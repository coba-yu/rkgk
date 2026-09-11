"""Renders a validation error location the way every agent-facing check reports its issues.

This module reads no model; it only formats the location tuples pydantic and the domain checks produce.
"""


def format_error_path(location: tuple[int | str, ...]) -> str:
    """Render a pydantic error location the way an agent reads its own JSON: `concepts[2].merged_from[0]`."""
    parts: list[str] = []
    for item in location:
        if isinstance(item, int):
            parts.append(f"[{item}]")
        else:
            parts.append(f".{item}" if parts else item)
    return "".join(parts) or "<root>"
