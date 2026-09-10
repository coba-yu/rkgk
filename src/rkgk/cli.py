"""Command-line entry point for rkgk.

Subcommands register themselves via `register_command` so that later PRs can
add functionality without editing `main`.
Every subcommand prints one JSON object on stdout and reports the outcome as an exit code:
0 for success, 1 for input the agent must fix, 2 for a failure of the environment or of the stored artifacts.
"""

import argparse
import json
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

from rkgk.domain.extraction import (
    ExtractionIssue,
    ExtractionResult,
    ExtractionValidationError,
    build_extraction_schema,
)
from rkgk.domain.repositories import ExtractionRepositoryError, PaperRepositoryError
from rkgk.infrastructure.file_extraction_repository import FileExtractionRepository
from rkgk.infrastructure.file_paper_repository import FilePaperRepository
from rkgk.usecase.save_extraction import SaveExtractionUseCase
from rkgk.usecase.validate_extraction import ValidateExtractionUseCase

Registrar = Callable[[argparse.ArgumentParser], None]

EXIT_INVALID = 1
EXIT_ERROR = 2

DEFAULT_DATA_DIR = Path("data")

# Each entry is (name, help, register_fn). register_fn(subparser) must add
# whatever arguments the command needs and call
# subparser.set_defaults(func=...) to wire up its handler.
_COMMANDS: list[tuple[str, str, Registrar]] = []


def register_command(name: str, help: str) -> Callable[[Registrar], Registrar]:
    """Decorator that registers a subcommand's argument setup function."""

    def decorator(register_fn: Registrar) -> Registrar:
        _COMMANDS.append((name, help, register_fn))
        return register_fn

    return decorator


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


@register_command("schema", "print the JSON Schema of an artifact an agent writes")
def _register_schema(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("artifact", choices=("extraction",))
    subparser.set_defaults(func=_run_schema)


def _run_schema(_args: argparse.Namespace) -> int:
    print(json.dumps(build_extraction_schema(), indent=2, ensure_ascii=False))
    return 0


@register_command("extract", "check or store the extraction JSON an agent wrote for one paper")
def _register_extract(subparser: argparse.ArgumentParser) -> None:
    actions = subparser.add_subparsers(dest="action", required=True)
    for name, help_text, handler in (
        ("validate", "check an extraction JSON against the paper text", _run_extract_validate),
        ("save", "check an extraction JSON and write it next to the paper", _run_extract_save),
    ):
        action = actions.add_parser(name, help=help_text)
        action.add_argument("paper_id", type=int)
        action.add_argument("json_path", type=Path)
        action.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
        action.set_defaults(func=handler)


def _run_extract_validate(args: argparse.Namespace) -> int:
    use_case = ValidateExtractionUseCase(FilePaperRepository(args.data_dir))
    return _run_extraction(args, use_case.execute, {})


def _run_extract_save(args: argparse.Namespace) -> int:
    extraction_repository = FileExtractionRepository(args.data_dir)
    use_case = SaveExtractionUseCase(FilePaperRepository(args.data_dir), extraction_repository)
    return _run_extraction(args, use_case.execute, {"path": str(extraction_repository.path_for(args.paper_id))})


def _run_extraction(
    args: argparse.Namespace, execute: Callable[[int, object], ExtractionResult], extra: dict[str, object]
) -> int:
    try:
        payload = json.loads(args.json_path.read_text(encoding="utf-8"))
        result = execute(args.paper_id, payload)
    except ExtractionValidationError as error:
        _print_json({"status": "invalid", "issues": [_render_issue(issue) for issue in error.issues]})
        return EXIT_INVALID
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        PaperRepositoryError,
        ExtractionRepositoryError,
    ) as error:
        _print_json({"status": "error", "message": str(error)})
        return EXIT_ERROR
    _print_json({"status": "ok", **_render_result(result), **extra})
    return 0


def _render_issue(issue: ExtractionIssue) -> dict[str, str]:
    return {"path": issue.path, "message": issue.message}


def _render_result(result: ExtractionResult) -> dict[str, object]:
    return {
        "paper_id": result.paper_id,
        "concepts": len(result.concepts),
        "paper_concepts": len(result.paper_concepts),
        "concept_relations": len(result.concept_relations),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rkgk")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('rkgk')}",
    )
    subparsers = parser.add_subparsers(dest="command")
    for name, help_text, register_fn in _COMMANDS:
        subparser = subparsers.add_parser(name, help=help_text)
        register_fn(subparser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
