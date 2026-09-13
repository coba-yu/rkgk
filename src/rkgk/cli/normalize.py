"""The `normalize` command, installed as the console script of the same name.

`main` embeds the concepts of every extracted paper to find the ones that sit close together, asks Claude once
to bundle them into merge candidates, asks it once per bundle to merge that bundle, joins the answers in
Python, and finally asks it once more to relate the merged concepts from general knowledge; the result is saved
as one normalization under the data directory.
Each stage is validated and retried on its own, so `attempts` reports a count per stage and a rejection reports
the `stage` it comes from.
The command takes no paper id because normalization is one cross-paper step: it reads the whole index and fails
when any paper of it has not been extracted yet.
The embedder is loaded here, so the command runs under `uv run --extra embedding`.
"""

import argparse
from importlib.metadata import version
from pathlib import Path

from rkgk.cli._shared import EXIT_ERROR, EXIT_INVALID, EXIT_OK, print_json
from rkgk.domain.agents import StructuredOutputAgentError
from rkgk.domain.embedders import EmbedderError
from rkgk.domain.models.concept_normalization import (
    ConceptNormalization,
    ConceptNormalizationIssue,
    ConceptNormalizationValidationError,
    MissingPaperExtractionsError,
)
from rkgk.domain.repositories.concept_normalization import ConceptNormalizationRepositoryError
from rkgk.domain.repositories.paper import PaperRepositoryError
from rkgk.domain.repositories.paper_extraction import PaperExtractionRepositoryError
from rkgk.domain.services.concept_candidates import DEFAULT_NEIGHBORS
from rkgk.infrastructure.claude_code_agent import ClaudeCodeAgent
from rkgk.infrastructure.file_concept_normalization_repository import FileConceptNormalizationRepository
from rkgk.infrastructure.file_paper_extraction_repository import FilePaperExtractionRepository
from rkgk.infrastructure.file_paper_repository import FilePaperRepository
from rkgk.infrastructure.qwen3_embedder import DEFAULT_MODEL_NAME, Qwen3Embedder
from rkgk.usecase.normalize_concepts import DEFAULT_CONCURRENCY, NormalizeConceptsUseCase

NAME = "normalize"
HELP = (
    "group the concepts of every extracted paper by their embeddings, merge each group with Claude, "
    "then relate the merged concepts from general knowledge"
)

DEFAULT_DATA_DIR = Path("data")
DEFAULT_MAX_ATTEMPTS = 3


def _run(args: argparse.Namespace) -> int:
    normalization_repository = FileConceptNormalizationRepository(args.data_dir)
    use_case = NormalizeConceptsUseCase(
        FilePaperRepository(args.data_dir),
        FilePaperExtractionRepository(args.data_dir),
        ClaudeCodeAgent(model=args.model),
        Qwen3Embedder(model_name=args.embedding_model),
        normalization_repository,
        max_attempts=args.max_attempts,
        neighbors=args.neighbors,
        concurrency=args.concurrency,
    )
    try:
        result = use_case.execute()
    except ConceptNormalizationValidationError as error:
        print_json(
            {
                "status": "invalid",
                "stage": error.stage,
                "attempts": args.max_attempts,
                "issues": [_render_issue(issue) for issue in error.issues],
            }
        )
        return EXIT_INVALID
    except (
        MissingPaperExtractionsError,
        StructuredOutputAgentError,
        PaperRepositoryError,
        PaperExtractionRepositoryError,
        ConceptNormalizationRepositoryError,
        EmbedderError,
        # An index without papers is a state of the data directory, not a bug, so it is reported like the rest.
        ValueError,
    ) as error:
        print_json({"status": "error", "message": str(error)})
        return EXIT_ERROR
    payload: dict[str, object] = {"status": "ok", "papers": len(result.paper_ids)}
    payload.update(_render_result(result.normalization))
    payload["groups"] = result.groups
    payload["attempts"] = {
        "grouping": result.grouping_attempts,
        "merge_calls": result.merge_calls,
        "merge_rounds": result.merge_rounds,
        "relations": result.relation_attempts,
    }
    payload["paths"] = [str(path) for path in normalization_repository.paths()]
    print_json(payload)
    return EXIT_OK


def _render_issue(issue: ConceptNormalizationIssue) -> dict[str, str]:
    return {"path": issue.path, "message": issue.message}


def _render_result(normalization: ConceptNormalization) -> dict[str, object]:
    return {
        "concepts": len(normalization.concepts),
        "merged_concepts": sum(1 for concept in normalization.concepts if len(concept.merged_from) > 1),
        "concept_relations": len(normalization.concept_relations),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=NAME, description=HELP)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('rkgk')}",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", default=None, help="model passed to the Claude CLI; its default is used when unset")
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_MODEL_NAME,
        help="model the qwen3 embedder loads",
    )
    parser.add_argument(
        "--neighbors",
        type=_positive_int,
        default=DEFAULT_NEIGHBORS,
        help="how many close concepts each concept is paired with before the grouping stage sees them",
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=DEFAULT_CONCURRENCY,
        help="how many groups the merge stage asks Claude about at once",
    )
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
