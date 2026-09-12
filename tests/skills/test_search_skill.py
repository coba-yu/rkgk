"""What keeps the search Agent Skill from drifting away from the `search` command.

The skill is prose an agent reads, not code the parser checks, so nothing stops an option, a field name or an
exit code from changing in `rkgk.cli.search` while `SKILL.md` keeps describing the old one.
These tests read the parser and the result models as the source of truth and check the skill document
against them.
"""

import argparse
import re
import shlex
from pathlib import Path

from rkgk.cli._shared import EXIT_ERROR, EXIT_INVALID, EXIT_OK
from rkgk.cli.search import _build_parser
from rkgk.domain.models.graph import ChunkEvidence, ConceptEdge, PaperConceptEdge
from rkgk.domain.models.search import ConceptHop, EmbeddedItemHit, PaperCandidate, SearchResult, TraversalPath

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / ".agents" / "skills" / "rkgk-search-papers"
SKILL_PATH = SKILL_DIR / "SKILL.md"
CLAUDE_SKILL_LINK = REPO_ROOT / ".claude" / "skills" / "rkgk-search-papers"

# The skill also documents `uv run --extra ...` and `aws s3 sync --exclude ... --delete`, so these long options belong
# to those commands and must not be checked against `_build_parser()` like an option of `search`.
FOREIGN_LONG_OPTIONS = frozenset({"--extra", "--exclude", "--delete"})

LONG_OPTION_PATTERN = re.compile(r"--[a-z][a-z0-9-]*")
BACKTICK_SPAN_PATTERN = re.compile(r"`([^`]+)`")

FIELD_OWNERS = (
    SearchResult,
    PaperCandidate,
    EmbeddedItemHit,
    TraversalPath,
    ConceptHop,
    PaperConceptEdge,
    ConceptEdge,
    ChunkEvidence,
)


def read_skill() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def parse_frontmatter(text: str) -> dict[str, str]:
    """Read the `key: value` lines between the leading `---` markers, the same block a skill loader reads."""
    lines = text.splitlines()
    assert lines[0] == "---", "SKILL.md must open with a frontmatter block"
    closing = lines[1:].index("---") + 1
    fields: dict[str, str] = {}
    for line in lines[1:closing]:
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def collect_long_options(text: str) -> set[str]:
    """Pull every `--long-option` token out of the whole document, prose and tables included."""
    return set(LONG_OPTION_PATTERN.findall(text))


def strip_fenced_code_blocks(text: str) -> str:
    """Drop every ``` ... ``` block, so a triple backtick fence cannot desync single-backtick pair matching.

    A fence is itself three backticks, which throws off the parity a `` `...` `` regex relies on for everything
    that follows it in the file; the prose outside fences is what documents the field names anyway.
    """
    kept: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            kept.append(line)
    return "\n".join(kept)


def collect_backtick_spans(text: str) -> list[str]:
    """Pull the contents of every backtick pair outside fenced code blocks, to check a field name is one of them."""
    return BACKTICK_SPAN_PATTERN.findall(strip_fenced_code_blocks(text))


def collect_command_lines(text: str) -> list[str]:
    """Pull the argument text of every documented `search` invocation out of the fenced code blocks.

    A fence can hold a shell session or a made-up path, so only the lines that actually invoke the command are
    worth parsing; everything past "search " is what a caller would type after the command name.
    """
    prefixes = ("uv run --extra embedding search ", "uv run search ")
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            continue
        for prefix in prefixes:
            if line.startswith(prefix):
                lines.append(line[len(prefix) :])
                break
    return lines


def collect_option_strings(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    # _option_string_actions is private, but argparse keeps no public way to list every registered option
    # string; parsing --help text back into option names would be a worse way to get the same mapping.
    return parser._option_string_actions


def test_every_long_option_in_the_skill_is_an_option_of_the_parser() -> None:
    parser = _build_parser()
    option_strings = collect_option_strings(parser)
    documented = collect_long_options(read_skill()) - FOREIGN_LONG_OPTIONS
    for option in documented:
        assert option in option_strings, f"{option} is documented but not an option of the search parser"


def test_every_parser_option_except_help_and_version_is_mentioned_in_the_skill() -> None:
    parser = _build_parser()
    option_strings = collect_option_strings(parser)
    text = read_skill()
    long_options = {option for option in option_strings if option.startswith("--")}
    for option in long_options - {"--help", "--version"}:
        assert option in text, f"{option} is an option of the search parser but missing from the skill"


def test_every_documented_search_invocation_parses() -> None:
    parser = _build_parser()
    command_lines = collect_command_lines(read_skill())
    assert command_lines, "the skill must document at least one search invocation to check"
    for rest in command_lines:
        # parse_args exits the process on a usage error, so a plain call here is enough to prove the line is valid.
        parser.parse_args(shlex.split(rest))


def test_the_skill_documents_the_same_status_words_and_exit_codes_as_the_code() -> None:
    text = read_skill()
    for word in ("ok", "invalid", "error", "embedding_model_mismatch"):
        assert f"`{word}`" in text, f"{word!r} is missing from the skill"
    for code, word in ((EXIT_OK, "ok"), (EXIT_INVALID, "invalid"), (EXIT_ERROR, "error")):
        row_start = f"| {code} | `{word}` |"
        assert row_start in text, f"the skill's status table row for {word!r} does not start with {row_start!r}"


def test_every_result_model_field_is_mentioned_in_the_skill_in_backticks() -> None:
    text = read_skill()
    spans = collect_backtick_spans(text)
    for model in FIELD_OWNERS:
        for field_name in model.model_fields:
            assert any(field_name in span for span in spans), (
                f"{model.__name__}.{field_name} is not wrapped in backticks anywhere in the skill"
            )


def test_the_skill_frontmatter_name_matches_the_directory() -> None:
    frontmatter = parse_frontmatter(read_skill())
    # A loader keys the skill by the frontmatter name, so a directory renamed without its name would leave two
    # names for one skill; the command name is not part of this, because the skill is named after what it does.
    assert frontmatter["name"] == SKILL_DIR.name


def test_the_claude_skills_link_resolves_to_the_agents_skills_directory() -> None:
    assert CLAUDE_SKILL_LINK.is_symlink()
    assert CLAUDE_SKILL_LINK.resolve() == SKILL_DIR.resolve()
