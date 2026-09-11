"""The `extract` command, installed as the console script of the same name.

`main` extracts one paper with Claude, validates the result against the paper text, retries on failure, and
saves the accepted extraction next to the paper.
The command prints one JSON object on stdout and reports the outcome as an exit code, so an agent can branch on
the code and read the details from the same output without parsing prose.
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from rkgk.domain.agents import StructuredOutputAgentError
from rkgk.domain.models.paper_extraction import PaperExtraction, PaperExtractionIssue, PaperExtractionValidationError
from rkgk.domain.repositories.paper import PaperRepositoryError
from rkgk.domain.repositories.paper_extraction import PaperExtractionRepositoryError
from rkgk.infrastructure.claude_code_agent import ClaudeCodeAgent
from rkgk.infrastructure.file_paper_extraction_repository import FilePaperExtractionRepository
from rkgk.infrastructure.file_paper_repository import FilePaperRepository
from rkgk.usecase.extract_paper import ExtractPaperUseCase

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_ERROR = 2

NAME = "extract"
HELP = "extract one paper with Claude and write the result next to it"

DEFAULT_DATA_DIR = Path("data")
DEFAULT_MAX_ATTEMPTS = 3


def print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def _run(args: argparse.Namespace) -> int:
    extraction_repository = FilePaperExtractionRepository(args.data_dir)
    use_case = ExtractPaperUseCase(
        FilePaperRepository(args.data_dir),
        ClaudeCodeAgent(model=args.model),
        extraction_repository,
        max_attempts=args.max_attempts,
    )
    try:
        outcome = use_case.execute(args.paper_id)
    except PaperExtractionValidationError as error:
        print_json(
            {
                "status": "invalid",
                "attempts": args.max_attempts,
                "issues": [_render_issue(issue) for issue in error.issues],
            }
        )
        return EXIT_INVALID
    except (StructuredOutputAgentError, PaperRepositoryError, PaperExtractionRepositoryError) as error:
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


def _render_issue(issue: PaperExtractionIssue) -> dict[str, str]:
    return {"path": issue.path, "message": issue.message}


def _render_result(result: PaperExtraction) -> dict[str, object]:
    return {
        "paper_id": result.paper_id,
        "concepts": len(result.concepts),
        "paper_concepts": len(result.paper_concepts),
        "concept_relations": len(result.concept_relations),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=NAME, description=HELP)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('rkgk')}",
    )
    parser.add_argument("paper_id", type=int)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", default=None, help="model passed to the Claude CLI; its default is used when unset")
    parser.add_argument("--max-attempts", type=_positive_int, default=DEFAULT_MAX_ATTEMPTS)
    return parser


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _run(args)
