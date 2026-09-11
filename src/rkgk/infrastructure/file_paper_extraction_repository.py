"""Extraction repository backed by files next to the paper they describe.

The artifact lives at `papers/NNNN/extraction.json`, so the extraction of a paper travels with its pages
and can be inspected, diffed, or deleted by hand.
"""

import json
from pathlib import Path

from pydantic import ValidationError

from rkgk.domain.models.paper import build_paper_dir_name
from rkgk.domain.models.paper_extraction import PaperExtraction
from rkgk.domain.repositories.paper_extraction import (
    PaperExtractionArtifactInvalidError,
    PaperExtractionArtifactUnreadableError,
    PaperExtractionNotFoundError,
    PaperExtractionRepositoryError,
)
from rkgk.infrastructure.file_paper_repository import PAPERS_DIR_NAME

EXTRACTION_FILE_NAME = "extraction.json"


def _fail(
    kind: type[PaperExtractionRepositoryError], path: Path, problem: str, paper_id: int
) -> PaperExtractionRepositoryError:
    return kind(f"paper {paper_id}: {path}: {problem}", location=str(path), paper_id=paper_id)


class FilePaperExtractionRepository:
    def __init__(self, data_dir: Path) -> None:
        self._papers_dir = data_dir / PAPERS_DIR_NAME

    def path_for(self, paper_id: int) -> Path:
        return self._papers_dir / build_paper_dir_name(paper_id) / EXTRACTION_FILE_NAME

    def save(self, result: PaperExtraction) -> None:
        path = self.path_for(result.paper_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
        except OSError as error:
            # A failed write is an environment problem, the same kind as a failed read, so it uses the same class.
            raise _fail(
                PaperExtractionArtifactUnreadableError,
                path,
                f"cannot be written: {error.strerror or error}",
                result.paper_id,
            ) from error

    def find(self, paper_id: int) -> PaperExtraction:
        path = self.path_for(paper_id)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError as error:
            raise _fail(PaperExtractionNotFoundError, path, "not found", paper_id) from error
        except OSError as error:
            raise _fail(
                PaperExtractionArtifactUnreadableError, path, f"cannot be read: {error.strerror or error}", paper_id
            ) from error
        except UnicodeDecodeError as error:
            raise _fail(PaperExtractionArtifactInvalidError, path, f"is not valid UTF-8: {error}", paper_id) from error
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise _fail(PaperExtractionArtifactInvalidError, path, f"is not valid JSON: {error}", paper_id) from error
        try:
            extraction = PaperExtraction.model_validate(payload)
        except ValidationError as error:
            raise _fail(
                PaperExtractionArtifactInvalidError, path, f"is not a valid PaperExtraction: {error}", paper_id
            ) from error
        if extraction.paper_id != paper_id:
            raise _fail(
                PaperExtractionArtifactInvalidError,
                path,
                f"declares paper_id {extraction.paper_id}, which does not match the requested id {paper_id}",
                paper_id,
            )
        return extraction
