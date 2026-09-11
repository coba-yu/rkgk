"""The `extract` command: everything that happens to the extraction JSON of one paper.

`run` drives the whole extraction with Claude and stores the result, while `schema`, `validate` and `save`
expose the single steps for a payload that was produced by hand or by another tool.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from rkgk.cli._output import EXIT_ERROR, EXIT_INVALID, EXIT_OK, print_json
from rkgk.domain.agents import StructuredOutputAgentError
from rkgk.domain.models.paper_extraction import (
    ExtractionValidationError,
    PaperExtraction,
    PaperExtractionIssue,
    build_extraction_schema,
)
from rkgk.domain.repositories.extraction import ExtractionRepositoryError
from rkgk.domain.repositories.paper import PaperRepositoryError
from rkgk.infrastructure.claude_extractor import ClaudeExtractor
from rkgk.infrastructure.file_extraction_repository import FileExtractionRepository
from rkgk.infrastructure.file_paper_repository import FilePaperRepository
from rkgk.usecase.extract_paper import ExtractPaperUseCase
from rkgk.usecase.save_extraction import SaveExtractionUseCase
from rkgk.usecase.validate_extraction import ValidateExtractionUseCase

NAME = "extract"
HELP = "describe, check, or store the extraction JSON an agent wrote for one paper"

DEFAULT_DATA_DIR = Path("data")
DEFAULT_MAX_ATTEMPTS = 3


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(NAME, help=HELP)
    actions = parser.add_subparsers(dest="action", required=True)
    schema = actions.add_parser("schema", help="print the JSON Schema an extraction JSON must follow")
    schema.set_defaults(func=_run_schema)
    run = actions.add_parser("run", help="extract one paper with Claude and write the result next to it")
    run.add_argument("paper_id", type=int)
    run.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    run.add_argument("--model", default=None, help="model passed to the Claude CLI; its default is used when unset")
    run.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    run.set_defaults(func=_run_run)
    # validate and save read the same two arguments, so they are declared together to keep them in step.
    for name, help_text, handler in (
        ("validate", "check an extraction JSON against the paper text", _run_validate),
        ("save", "check an extraction JSON and write it next to the paper", _run_save),
    ):
        action = actions.add_parser(name, help=help_text)
        action.add_argument("paper_id", type=int)
        action.add_argument("json_path", type=Path)
        action.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
        action.set_defaults(func=handler)


def _run_schema(_args: argparse.Namespace) -> int:
    print(json.dumps(build_extraction_schema(), indent=2, ensure_ascii=False))
    return EXIT_OK


def _run_run(args: argparse.Namespace) -> int:
    extraction_repository = FileExtractionRepository(args.data_dir)
    use_case = ExtractPaperUseCase(
        FilePaperRepository(args.data_dir),
        ClaudeExtractor(model=args.model),
        extraction_repository,
        max_attempts=args.max_attempts,
    )
    try:
        outcome = use_case.execute(args.paper_id)
    except ExtractionValidationError as error:
        print_json(
            {
                "status": "invalid",
                "attempts": args.max_attempts,
                "issues": [_render_issue(issue) for issue in error.issues],
            }
        )
        return EXIT_INVALID
    except (StructuredOutputAgentError, PaperRepositoryError, ExtractionRepositoryError) as error:
        print_json({"status": "error", "message": str(error)})
        return EXIT_ERROR
    payload: dict[str, object] = {
        "status": "ok",
        "paper_id": outcome.extraction.paper_id,
        "attempts": outcome.attempts,
    }
    payload.update(_render_result(outcome.extraction))
    payload["path"] = str(extraction_repository.path_for(args.paper_id))
    print_json(payload)
    return EXIT_OK


def _run_validate(args: argparse.Namespace) -> int:
    use_case = ValidateExtractionUseCase(FilePaperRepository(args.data_dir))
    return _run(args, use_case.execute, {})


def _run_save(args: argparse.Namespace) -> int:
    extraction_repository = FileExtractionRepository(args.data_dir)
    use_case = SaveExtractionUseCase(FilePaperRepository(args.data_dir), extraction_repository)
    return _run(args, use_case.execute, {"path": str(extraction_repository.path_for(args.paper_id))})


def _run(args: argparse.Namespace, execute: Callable[[int, object], PaperExtraction], extra: dict[str, object]) -> int:
    try:
        payload = json.loads(args.json_path.read_text(encoding="utf-8"))
        result = execute(args.paper_id, payload)
    except ExtractionValidationError as error:
        print_json({"status": "invalid", "issues": [_render_issue(issue) for issue in error.issues]})
        return EXIT_INVALID
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        PaperRepositoryError,
        ExtractionRepositoryError,
    ) as error:
        print_json({"status": "error", "message": str(error)})
        return EXIT_ERROR
    print_json({"status": "ok", **_render_result(result), **extra})
    return EXIT_OK


def _render_issue(issue: PaperExtractionIssue) -> dict[str, str]:
    return {"path": issue.path, "message": issue.message}


def _render_result(result: PaperExtraction) -> dict[str, object]:
    return {
        "paper_id": result.paper_id,
        "concepts": len(result.concepts),
        "paper_concepts": len(result.paper_concepts),
        "concept_relations": len(result.concept_relations),
    }
