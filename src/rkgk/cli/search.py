"""The `search` command, installed as the console script of the same name.

This is the entry point the Codex and Claude Code Agent Skills call: a theme in Japanese goes in, and the papers
to read next come out with the hits and the graph paths that chose them.
The whole answer is one JSON object on stdout, so the skill reads the candidates and their evidence from the same
output it branches on, without a second call and without parsing prose.
"""

import argparse
from importlib.metadata import version
from pathlib import Path

from rkgk.cli._shared import EXIT_ERROR, EXIT_INVALID, EXIT_OK, print_json
from rkgk.domain.embedders import Embedder, EmbedderError
from rkgk.domain.models.manifest import EmbeddingModelMismatchError
from rkgk.domain.models.search import SearchConfig
from rkgk.domain.repositories.index import IndexRepository, IndexRepositoryError
from rkgk.domain.repositories.paper import PaperRepositoryError
from rkgk.domain.repositories.paper_extraction import PaperExtractionRepositoryError
from rkgk.infrastructure.fake_embedder import FakeEmbedder
from rkgk.infrastructure.file_index_repository import FileIndexRepository
from rkgk.infrastructure.file_paper_extraction_repository import FilePaperExtractionRepository
from rkgk.infrastructure.file_paper_repository import FilePaperRepository
from rkgk.infrastructure.qwen3_embedder import Qwen3Embedder
from rkgk.usecase.search_papers import SearchPapersUseCase

NAME = "search"
HELP = "find the papers to read next for one or more themes, with the hits and the graph paths that chose them"

DEFAULT_DATA_DIR = Path("data")
QWEN3_EMBEDDER = "qwen3"
FAKE_EMBEDDER = "fake"

# The defaults of the command are the defaults of the configuration itself, so the two cannot drift apart.
DEFAULT_CONFIG = SearchConfig()


def _build_embedder(args: argparse.Namespace, index_repository: IndexRepository) -> Embedder:
    if args.embedder == FAKE_EMBEDDER:
        return FakeEmbedder()
    # Falling back to the manifest keeps the queries on the model the vectors were made with, which is the only
    # model they can be compared against; the use case reports the mismatch if the caller names another one.
    model_name = args.embedding_model or index_repository.find_manifest().embedding_model
    return Qwen3Embedder(model_name=model_name)


def _build_config(args: argparse.Namespace) -> SearchConfig:
    return SearchConfig(
        top_k=args.top_k,
        max_hops=args.max_hops,
        generic_concept_threshold=args.generic_concept_threshold,
        max_graph_candidates=args.max_graph_candidates,
    )


def _run(args: argparse.Namespace) -> int:
    index_repository = FileIndexRepository(args.data_dir)
    try:
        use_case = SearchPapersUseCase(
            FilePaperRepository(args.data_dir),
            FilePaperExtractionRepository(args.data_dir),
            index_repository,
            _build_embedder(args, index_repository),
        )
        result = use_case.execute(args.queries, _build_config(args))
    except EmbeddingModelMismatchError as error:
        print_json(
            {
                "status": "invalid",
                "reason": "embedding_model_mismatch",
                "index_model": error.index_model,
                "embedder_model": error.embedder_model,
                "message": str(error),
            }
        )
        return EXIT_INVALID
    except (
        PaperRepositoryError,
        PaperExtractionRepositoryError,
        IndexRepositoryError,
        EmbedderError,
        # A rejected query list and a candidate that cannot be assembled are states of the input and of the data
        # directory, not bugs, so they are reported like a missing artifact instead of crashing the command.
        ValueError,
    ) as error:
        print_json({"status": "error", "message": str(error)})
        return EXIT_ERROR
    print_json({"status": "ok", **result.model_dump(mode="json")})
    return EXIT_OK


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=NAME, description=HELP)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('rkgk')}",
    )
    parser.add_argument(
        "queries",
        nargs="+",
        metavar="QUERY",
        help="theme to search for, in Japanese; every query is ranked on its own and the hits are unioned",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--embedder",
        choices=[QWEN3_EMBEDDER, FAKE_EMBEDDER],
        default=QWEN3_EMBEDDER,
        help="fake derives vectors from a hash of the text, which checks the wiring without loading a model",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=(
            "model the qwen3 embedder loads; defaults to the model the index was built with, "
            "the embedding_model of its manifest; ignored by the fake embedder"
        ),
    )
    parser.add_argument("--top-k", type=_positive_int, default=DEFAULT_CONFIG.top_k)
    parser.add_argument("--max-hops", type=_non_negative_int, default=DEFAULT_CONFIG.max_hops)
    parser.add_argument("--max-graph-candidates", type=_non_negative_int, default=DEFAULT_CONFIG.max_graph_candidates)
    parser.add_argument("--generic-concept-threshold", type=_share, default=DEFAULT_CONFIG.generic_concept_threshold)
    return parser


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be at least 0, got {value}")
    return value


def _share(text: str) -> float:
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0.0 and 1.0, got {value}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _run(args)
