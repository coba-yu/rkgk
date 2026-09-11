"""Concept normalization repository backed by two files under the data directory.

The nodes live in `normalization/concepts.json` and the general-knowledge edges in
`normalization/concept_relations.json`, so the shared vocabulary can be reviewed by hand without scrolling past
the edges, and a diff of one of them stays readable.
Both files carry the schema version of the normalization they were written from, and `find` reassembles one
`ConceptNormalization` from them, so a pair that does not belong together is reported instead of being merged.
"""

import json
from pathlib import Path

from pydantic import ValidationError

from rkgk.domain.models.concept_normalization import ConceptNormalization
from rkgk.domain.repositories.concept_normalization import (
    ConceptNormalizationArtifactInvalidError,
    ConceptNormalizationArtifactUnreadableError,
    ConceptNormalizationNotFoundError,
    ConceptNormalizationRepositoryError,
)

NORMALIZATION_DIR_NAME = "normalization"
CONCEPTS_FILE_NAME = "concepts.json"
CONCEPT_RELATIONS_FILE_NAME = "concept_relations.json"


def _fail(
    kind: type[ConceptNormalizationRepositoryError], path: Path, problem: str
) -> ConceptNormalizationRepositoryError:
    return kind(f"{path}: {problem}", location=str(path))


class FileConceptNormalizationRepository:
    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir / NORMALIZATION_DIR_NAME

    @property
    def concepts_path(self) -> Path:
        return self._dir / CONCEPTS_FILE_NAME

    @property
    def concept_relations_path(self) -> Path:
        return self._dir / CONCEPT_RELATIONS_FILE_NAME

    def paths(self) -> tuple[Path, ...]:
        return (self.concepts_path, self.concept_relations_path)

    def save(self, normalization: ConceptNormalization) -> None:
        payload = normalization.model_dump(mode="json")
        version = payload["schema_version"]
        self._write(self.concepts_path, {"schema_version": version, "concepts": payload["concepts"]})
        self._write(
            self.concept_relations_path,
            {"schema_version": version, "concept_relations": payload["concept_relations"]},
        )

    def find(self) -> ConceptNormalization:
        concepts_document = self._read(self.concepts_path)
        relations_document = self._read(self.concept_relations_path)
        version = concepts_document.get("schema_version")
        if version != relations_document.get("schema_version"):
            raise _fail(
                ConceptNormalizationArtifactInvalidError,
                self._dir,
                f"{CONCEPTS_FILE_NAME} declares schema_version {version!r} while "
                f"{CONCEPT_RELATIONS_FILE_NAME} declares {relations_document.get('schema_version')!r}",
            )
        try:
            return ConceptNormalization.model_validate(
                {
                    "schema_version": version,
                    "concepts": concepts_document.get("concepts"),
                    "concept_relations": relations_document.get("concept_relations"),
                }
            )
        except ValidationError as error:
            raise _fail(
                ConceptNormalizationArtifactInvalidError,
                self._dir,
                f"is not a valid ConceptNormalization: {error}",
            ) from error

    def _write(self, path: Path, document: dict[str, object]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError as error:
            # A failed write is an environment problem, the same kind as a failed read, so it uses the same class.
            raise _fail(
                ConceptNormalizationArtifactUnreadableError, path, f"cannot be written: {error.strerror or error}"
            ) from error

    def _read(self, path: Path) -> dict[str, object]:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError as error:
            raise _fail(ConceptNormalizationNotFoundError, path, "not found") from error
        except OSError as error:
            raise _fail(
                ConceptNormalizationArtifactUnreadableError, path, f"cannot be read: {error.strerror or error}"
            ) from error
        except UnicodeDecodeError as error:
            raise _fail(ConceptNormalizationArtifactInvalidError, path, f"is not valid UTF-8: {error}") from error
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise _fail(ConceptNormalizationArtifactInvalidError, path, f"is not valid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise _fail(
                ConceptNormalizationArtifactInvalidError,
                path,
                f"is a JSON {type(payload).__name__} instead of an object",
            )
        return payload
