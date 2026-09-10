import json
from pathlib import Path

import pytest

from rkgk.domain.models.extraction import (
    ExtractedConcept,
    ExtractedEvidence,
    ExtractedPaperConceptEdge,
    ExtractionResult,
)
from rkgk.domain.models.vocabulary import ConceptType, PaperConceptRelation
from rkgk.domain.repositories.extraction import (
    ExtractionArtifactInvalidError,
    ExtractionNotFoundError,
    ExtractionRepositoryError,
)
from rkgk.infrastructure.file_extraction_repository import FileExtractionRepository

RESULT = ExtractionResult(
    schema_version=1,
    paper_id=1,
    summary_ja="この論文は検索拡張生成のパイプラインを提案する。",
    concepts=(ExtractedConcept(local_id="c1", name="Retrieval-Augmented Generation", type=ConceptType.METHOD),),
    paper_concepts=(
        ExtractedPaperConceptEdge(
            concept_id="c1",
            relation=PaperConceptRelation.PROPOSES,
            evidence=(ExtractedEvidence(page=1, quote="retrieval-augmented generation"),),
        ),
    ),
)


def test_a_saved_extraction_is_read_back_unchanged(tmp_path: Path) -> None:
    repository = FileExtractionRepository(tmp_path)
    repository.save(RESULT)
    assert repository.find(1) == RESULT


def test_the_file_is_written_next_to_the_paper_as_readable_json(tmp_path: Path) -> None:
    FileExtractionRepository(tmp_path).save(RESULT)
    path = tmp_path / "papers" / "0001" / "extraction.json"
    text = path.read_text(encoding="utf-8")
    assert json.loads(text)["paper_id"] == 1
    assert text.endswith("}\n")
    assert "\n  " in text


def test_the_summary_is_stored_as_japanese_characters(tmp_path: Path) -> None:
    FileExtractionRepository(tmp_path).save(RESULT)
    text = (tmp_path / "papers" / "0001" / "extraction.json").read_text(encoding="utf-8")
    assert "検索拡張生成" in text


def test_saving_twice_replaces_the_previous_extraction(tmp_path: Path) -> None:
    repository = FileExtractionRepository(tmp_path)
    repository.save(RESULT)
    repository.save(RESULT.model_copy(update={"summary_ja": "二回目の要約。"}))
    assert repository.find(1).summary_ja == "二回目の要約。"


def test_a_paper_without_an_extraction_is_reported_with_its_id(tmp_path: Path) -> None:
    with pytest.raises(ExtractionNotFoundError, match="paper 2") as caught:
        FileExtractionRepository(tmp_path).find(2)
    assert caught.value.paper_id == 2


def test_a_corrupted_file_is_reported_as_invalid(tmp_path: Path) -> None:
    FileExtractionRepository(tmp_path).save(RESULT)
    (tmp_path / "papers" / "0001" / "extraction.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ExtractionArtifactInvalidError, match="is not valid JSON"):
        FileExtractionRepository(tmp_path).find(1)


def test_a_file_that_is_not_an_extraction_is_reported_as_invalid(tmp_path: Path) -> None:
    FileExtractionRepository(tmp_path).save(RESULT)
    (tmp_path / "papers" / "0001" / "extraction.json").write_text('{"paper_id": 1}', encoding="utf-8")
    with pytest.raises(ExtractionArtifactInvalidError, match="is not a valid ExtractionResult"):
        FileExtractionRepository(tmp_path).find(1)


def test_a_non_utf8_file_is_reported_as_invalid(tmp_path: Path) -> None:
    FileExtractionRepository(tmp_path).save(RESULT)
    (tmp_path / "papers" / "0001" / "extraction.json").write_bytes(b"\xff\xfe")
    with pytest.raises(ExtractionArtifactInvalidError, match="is not valid UTF-8"):
        FileExtractionRepository(tmp_path).find(1)


def test_errors_carry_the_location_as_an_attribute(tmp_path: Path) -> None:
    with pytest.raises(ExtractionRepositoryError) as caught:
        FileExtractionRepository(tmp_path).find(1)
    assert caught.value.location == str(tmp_path / "papers" / "0001" / "extraction.json")
