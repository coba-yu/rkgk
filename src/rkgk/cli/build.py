"""The `build` command, installed as the console script of the same name.

`main` turns the stored papers, extractions and normalization into the index search reads: the chunks, the
embedding table, the knowledge graph, and the manifest that records with which settings they were made.
No agent runs here, so the command reports a stale or hand-edited artifact as invalid rather than retrying:
extracting or normalizing again is what fixes it.
"""

import argparse
from importlib.metadata import version
from pathlib import Path

from rkgk.cli._shared import EXIT_ERROR, EXIT_INVALID, EXIT_OK, print_json
from rkgk.domain.embedders import Embedder, EmbedderError
from rkgk.domain.models.concept_normalization import (
    ConceptNormalizationIssue,
    ConceptNormalizationValidationError,
    MissingPaperExtractionsError,
)
from rkgk.domain.models.graph import EvidenceResolutionError
from rkgk.domain.models.manifest import IndexBuildRun
from rkgk.domain.models.paper_extraction import PaperExtractionIssue, PaperExtractionMismatchError
from rkgk.domain.repositories.concept_normalization import ConceptNormalizationRepositoryError
from rkgk.domain.repositories.index import IndexRepositoryError
from rkgk.domain.repositories.paper import PaperRepositoryError
from rkgk.domain.repositories.paper_extraction import PaperExtractionRepositoryError
from rkgk.domain.services.chunking import DEFAULT_MAX_TOKENS
from rkgk.infrastructure.fake_embedder import FakeEmbedder
from rkgk.infrastructure.file_concept_normalization_repository import FileConceptNormalizationRepository
from rkgk.infrastructure.file_index_repository import FileIndexRepository
from rkgk.infrastructure.file_paper_extraction_repository import FilePaperExtractionRepository
from rkgk.infrastructure.file_paper_repository import FilePaperRepository
from rkgk.infrastructure.qwen3_embedder import DEFAULT_MODEL_NAME, Qwen3Embedder
from rkgk.infrastructure.whitespace_tokenizer import WhitespaceTokenizer
from rkgk.usecase.build_index import BuildIndexUseCase

NAME = "build"
HELP = "build the index search reads from the stored papers, extractions and normalization"

DEFAULT_DATA_DIR = Path("data")
QWEN3_EMBEDDER = "qwen3"
FAKE_EMBEDDER = "fake"


def _build_embedder(args: argparse.Namespace) -> Embedder:
    if args.embedder == FAKE_EMBEDDER:
        return FakeEmbedder()
    return Qwen3Embedder(model_name=args.embedding_model)


def _run(args: argparse.Namespace) -> int:
    index_repository = FileIndexRepository(args.data_dir)
    use_case = BuildIndexUseCase(
        FilePaperRepository(args.data_dir),
        FilePaperExtractionRepository(args.data_dir),
        FileConceptNormalizationRepository(args.data_dir),
        # The whitespace tokenizer is the only Tokenizer implemented so far, so the chunk budget counts words.
        WhitespaceTokenizer(),
        _build_embedder(args),
        index_repository,
        max_tokens=args.max_tokens,
    )
    try:
        outcome = use_case.execute()
    except PaperExtractionMismatchError as error:
        print_json(
            {
                "status": "invalid",
                "reason": "extraction_mismatch",
                "papers": [
                    {"paper_id": paper_id, "issues": [_render_extraction_issue(issue) for issue in issues]}
                    for paper_id, issues in error.issues_by_paper.items()
                ],
            }
        )
        return EXIT_INVALID
    except ConceptNormalizationValidationError as error:
        print_json(
            {
                "status": "invalid",
                "reason": "normalization_mismatch",
                "issues": [_render_normalization_issue(issue) for issue in error.issues],
            }
        )
        return EXIT_INVALID
    except EvidenceResolutionError as error:
        print_json(
            {
                "status": "invalid",
                "reason": "unresolved_evidence",
                "unresolved": [
                    {"paper_id": item.paper_id, "page": item.page, "quote": item.quote} for item in error.unresolved
                ],
            }
        )
        return EXIT_INVALID
    except (
        MissingPaperExtractionsError,
        PaperRepositoryError,
        PaperExtractionRepositoryError,
        ConceptNormalizationRepositoryError,
        IndexRepositoryError,
        EmbedderError,
        # An index without papers is a state of the data directory, not a bug, so it is reported like the rest.
        ValueError,
    ) as error:
        print_json({"status": "error", "message": str(error)})
        return EXIT_ERROR
    payload: dict[str, object] = {"status": "ok"}
    payload.update(_render_result(outcome))
    payload["paths"] = [str(path) for path in index_repository.paths()]
    print_json(payload)
    return EXIT_OK


def _render_extraction_issue(issue: PaperExtractionIssue) -> dict[str, str]:
    return {"path": issue.path, "message": issue.message}


def _render_normalization_issue(issue: ConceptNormalizationIssue) -> dict[str, str]:
    return {"path": issue.path, "message": issue.message}


def _render_result(outcome: IndexBuildRun) -> dict[str, object]:
    return {
        "papers": len(outcome.manifest.paper_ids),
        "chunks": len(outcome.chunks),
        "embedded_items": len(outcome.embeddings.items),
        "embedding_model": outcome.manifest.embedding_model,
        "embedding_dimension": outcome.manifest.embedding_dimension,
        "concepts": len(outcome.graph.concepts),
        "paper_concepts": len(outcome.graph.paper_concepts),
        "concept_relations": len(outcome.graph.concept_relations),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=NAME, description=HELP)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('rkgk')}",
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
        default=DEFAULT_MODEL_NAME,
        help="model the qwen3 embedder loads; ignored by the fake embedder",
    )
    parser.add_argument("--max-tokens", type=_positive_int, default=DEFAULT_MAX_TOKENS)
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
