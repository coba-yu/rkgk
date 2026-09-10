"""Paper repository backed by the artifacts produced by the preprocessing project.

The layout under the data directory is `papers/index.json` plus one directory per paper that holds `paper.json`
and a `pages/` directory with one Markdown file per page.
Every problem found while reading is raised as `PaperRepositoryError` so that a caller never sees a half-built paper.
"""

import json
from pathlib import Path

from pydantic import ValidationError

from rkgk.domain.paper import Page, Paper, PaperIndexEntry, PaperMeta, build_paper_dir_name, parse_page
from rkgk.domain.repositories import (
    PaperArtifactInvalidError,
    PaperArtifactUnreadableError,
    PaperNotFoundError,
    PaperRepositoryError,
)

PAPERS_DIR_NAME = "papers"
INDEX_FILE_NAME = "index.json"
META_FILE_NAME = "paper.json"
PAGES_DIR_NAME = "pages"


def build_page_file_name(number: int) -> str:
    return f"{number:03d}.md"


def _fail(
    kind: type[PaperRepositoryError], path: Path, problem: str, paper_id: int | None = None
) -> PaperRepositoryError:
    subject = "" if paper_id is None else f"paper {paper_id}: "
    return kind(f"{subject}{path}: {problem}", location=str(path), paper_id=paper_id)


def _read_text(path: Path, paper_id: int | None = None) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise _fail(PaperNotFoundError, path, "not found", paper_id) from error
    except OSError as error:
        raise _fail(
            PaperArtifactUnreadableError, path, f"cannot be read: {error.strerror or error}", paper_id
        ) from error
    except UnicodeDecodeError as error:
        raise _fail(PaperArtifactInvalidError, path, f"is not valid UTF-8: {error}", paper_id) from error


def _read_json(path: Path, paper_id: int | None = None) -> object:
    raw = _read_text(path, paper_id)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise _fail(PaperArtifactInvalidError, path, f"is not valid JSON: {error}", paper_id) from error


class FilePaperRepository:
    def __init__(self, data_dir: Path) -> None:
        self._papers_dir = data_dir / PAPERS_DIR_NAME

    def find_index(self) -> tuple[PaperIndexEntry, ...]:
        path = self._papers_dir / INDEX_FILE_NAME
        payload = _read_json(path)
        if not isinstance(payload, dict) or not isinstance(payload.get("papers"), list):
            raise _fail(PaperArtifactInvalidError, path, "must be an object with a 'papers' array")
        entries: list[PaperIndexEntry] = []
        seen: set[int] = set()
        for position, item in enumerate(payload["papers"]):
            try:
                entry = PaperIndexEntry.model_validate(item)
            except ValidationError as error:
                raise _fail(
                    PaperArtifactInvalidError, path, f"entry at position {position} is invalid: {error}"
                ) from error
            if entry.id in seen:
                raise _fail(PaperArtifactInvalidError, path, f"contains duplicate paper id {entry.id}")
            seen.add(entry.id)
            entries.append(entry)
        return tuple(entries)

    def find(self, paper_id: int) -> Paper:
        paper_dir = self._papers_dir / build_paper_dir_name(paper_id)
        if not paper_dir.is_dir():
            raise _fail(PaperNotFoundError, paper_dir, "paper directory not found", paper_id)
        meta = self._read_meta(paper_dir, paper_id)
        return Paper(meta=meta, pages=self._read_pages(paper_dir, meta))

    def _read_meta(self, paper_dir: Path, paper_id: int) -> PaperMeta:
        path = paper_dir / META_FILE_NAME
        if not path.is_file():
            raise _fail(PaperNotFoundError, path, f"{META_FILE_NAME} not found", paper_id)
        payload = _read_json(path, paper_id)
        try:
            meta = PaperMeta.model_validate(payload)
        except ValidationError as error:
            raise _fail(PaperArtifactInvalidError, path, f"is not a valid PaperMeta: {error}", paper_id) from error
        if meta.id != paper_id:
            raise _fail(
                PaperArtifactInvalidError,
                path,
                f"declares id {meta.id}, which does not match the requested id {paper_id}",
                paper_id,
            )
        return meta

    def _read_pages(self, paper_dir: Path, meta: PaperMeta) -> tuple[Page, ...]:
        pages_dir = paper_dir / PAGES_DIR_NAME
        if not pages_dir.is_dir():
            raise _fail(PaperArtifactInvalidError, pages_dir, f"{PAGES_DIR_NAME} directory not found", meta.id)
        expected = {build_page_file_name(number) for number in range(1, meta.page_count + 1)}
        try:
            found = {entry.name for entry in pages_dir.iterdir()}
        except OSError as error:
            raise _fail(
                PaperArtifactUnreadableError, pages_dir, f"cannot be listed: {error.strerror or error}", meta.id
            ) from error
        missing = sorted(expected - found)
        if missing:
            raise _fail(
                PaperArtifactInvalidError,
                pages_dir,
                f"is missing page files for page_count {meta.page_count}: {', '.join(missing)}",
                meta.id,
            )
        unexpected = sorted(found - expected)
        if unexpected:
            raise _fail(
                PaperArtifactInvalidError,
                pages_dir,
                f"contains files that are not pages 1..{meta.page_count}: {', '.join(unexpected)}",
                meta.id,
            )
        pages: list[Page] = []
        for number in range(1, meta.page_count + 1):
            path = pages_dir / build_page_file_name(number)
            text = _read_text(path, meta.id)
            try:
                pages.append(parse_page(number, text))
            except ValidationError as error:
                raise _fail(PaperArtifactInvalidError, path, f"has an invalid marker: {error}", meta.id) from error
        return tuple(pages)
