import json
from pathlib import Path

import pytest

from rkgk.domain.models.concept_normalization import (
    ConceptNormalization,
    GeneralKnowledgeEdge,
    LocalConceptRef,
    NormalizedConcept,
)
from rkgk.domain.models.vocabulary import ConceptRelationType, ConceptType
from rkgk.domain.repositories.concept_normalization import (
    ConceptNormalizationArtifactInvalidError,
    ConceptNormalizationNotFoundError,
    ConceptNormalizationRepositoryError,
)
from rkgk.infrastructure.file_concept_normalization_repository import FileConceptNormalizationRepository

NORMALIZATION = ConceptNormalization(
    schema_version=1,
    concepts=(
        NormalizedConcept(
            id="retrieval-augmented-generation",
            canonical_name="Retrieval-Augmented Generation",
            type=ConceptType.METHOD,
            aliases=("RAG", "検索拡張生成"),
            merged_from=(LocalConceptRef(paper_id=1, local_id="c1"), LocalConceptRef(paper_id=2, local_id="c1")),
        ),
        NormalizedConcept(
            id="knowledge-graph",
            canonical_name="Knowledge Graph",
            type=ConceptType.METHOD,
            merged_from=(LocalConceptRef(paper_id=2, local_id="c2"),),
        ),
    ),
    concept_relations=(
        GeneralKnowledgeEdge(
            source_id="knowledge-graph",
            target_id="retrieval-augmented-generation",
            relation=ConceptRelationType.USED_FOR,
            rationale="A knowledge graph supplies the structure a retrieval-augmented pipeline walks.",
        ),
    ),
)


def test_a_saved_normalization_is_read_back_unchanged(tmp_path: Path) -> None:
    repository = FileConceptNormalizationRepository(tmp_path)
    repository.save(NORMALIZATION)
    assert repository.find() == NORMALIZATION


def test_the_concepts_and_the_relations_are_written_to_two_readable_files(tmp_path: Path) -> None:
    repository = FileConceptNormalizationRepository(tmp_path)
    repository.save(NORMALIZATION)
    assert repository.paths() == (
        tmp_path / "normalization" / "concepts.json",
        tmp_path / "normalization" / "concept_relations.json",
    )
    concepts = json.loads(repository.concepts_path.read_text(encoding="utf-8"))
    relations = json.loads(repository.concept_relations_path.read_text(encoding="utf-8"))
    assert [concept["id"] for concept in concepts["concepts"]] == [
        "retrieval-augmented-generation",
        "knowledge-graph",
    ]
    assert "concepts" not in relations
    assert relations["concept_relations"][0]["relation"] == "used_for"
    assert concepts["schema_version"] == relations["schema_version"] == 1


def test_the_aliases_are_stored_as_japanese_characters(tmp_path: Path) -> None:
    repository = FileConceptNormalizationRepository(tmp_path)
    repository.save(NORMALIZATION)
    assert "検索拡張生成" in repository.concepts_path.read_text(encoding="utf-8")


def test_saving_twice_replaces_the_previous_normalization(tmp_path: Path) -> None:
    repository = FileConceptNormalizationRepository(tmp_path)
    repository.save(NORMALIZATION)
    repository.save(NORMALIZATION.model_copy(update={"concept_relations": ()}))
    assert repository.find().concept_relations == ()


def test_a_data_directory_without_a_normalization_is_reported_as_not_found(tmp_path: Path) -> None:
    with pytest.raises(ConceptNormalizationNotFoundError, match="not found"):
        FileConceptNormalizationRepository(tmp_path).find()


def test_a_normalization_whose_relations_file_is_gone_is_reported_as_not_found(tmp_path: Path) -> None:
    repository = FileConceptNormalizationRepository(tmp_path)
    repository.save(NORMALIZATION)
    repository.concept_relations_path.unlink()
    with pytest.raises(ConceptNormalizationNotFoundError, match="concept_relations.json"):
        repository.find()


def test_a_normalization_whose_concepts_file_is_gone_is_reported_as_not_found(tmp_path: Path) -> None:
    repository = FileConceptNormalizationRepository(tmp_path)
    repository.save(NORMALIZATION)
    repository.concepts_path.unlink()
    with pytest.raises(ConceptNormalizationNotFoundError, match="concepts.json"):
        repository.find()


def test_a_corrupted_file_is_reported_as_invalid(tmp_path: Path) -> None:
    repository = FileConceptNormalizationRepository(tmp_path)
    repository.save(NORMALIZATION)
    repository.concepts_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConceptNormalizationArtifactInvalidError, match="is not valid JSON"):
        repository.find()


def test_a_file_that_is_not_a_normalization_is_reported_as_invalid(tmp_path: Path) -> None:
    repository = FileConceptNormalizationRepository(tmp_path)
    repository.save(NORMALIZATION)
    repository.concepts_path.write_text('{"schema_version": 1, "concepts": [{"id": "rag"}]}', encoding="utf-8")
    with pytest.raises(ConceptNormalizationArtifactInvalidError, match="is not a valid ConceptNormalization"):
        repository.find()


def test_two_files_written_from_different_schema_versions_are_reported_as_invalid(tmp_path: Path) -> None:
    repository = FileConceptNormalizationRepository(tmp_path)
    repository.save(NORMALIZATION)
    document = json.loads(repository.concept_relations_path.read_text(encoding="utf-8"))
    document["schema_version"] = 2
    repository.concept_relations_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConceptNormalizationArtifactInvalidError, match="declares schema_version 1 while"):
        repository.find()


def test_a_non_utf8_file_is_reported_as_invalid(tmp_path: Path) -> None:
    repository = FileConceptNormalizationRepository(tmp_path)
    repository.save(NORMALIZATION)
    repository.concept_relations_path.write_bytes(b"\xff\xfe")
    with pytest.raises(ConceptNormalizationArtifactInvalidError, match="is not valid UTF-8"):
        repository.find()


def test_a_file_holding_a_json_array_is_reported_as_invalid(tmp_path: Path) -> None:
    repository = FileConceptNormalizationRepository(tmp_path)
    repository.save(NORMALIZATION)
    repository.concepts_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ConceptNormalizationArtifactInvalidError, match="is a JSON list instead of an object"):
        repository.find()


def test_errors_carry_the_location_as_an_attribute(tmp_path: Path) -> None:
    repository = FileConceptNormalizationRepository(tmp_path)
    with pytest.raises(ConceptNormalizationRepositoryError) as caught:
        repository.find()
    assert caught.value.location == str(repository.concepts_path)
