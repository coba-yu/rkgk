import json

import pytest
from pydantic import ValidationError

from rkgk.domain.models.paper_extraction import (
    EXTRACTION_SCHEMA_VERSION,
    LOCAL_CONCEPT_ID_PATTERN,
    ExtractedConcept,
    ExtractedConceptEdge,
    ExtractedPaperConceptEdge,
    PageEvidence,
    PaperExtraction,
    PaperExtractionIssue,
    PaperExtractionMismatchError,
    build_paper_extraction_schema,
)
from rkgk.domain.models.vocabulary import ConceptRelationType, ConceptType, PaperConceptRelation

CONCEPTS = (
    ExtractedConcept(local_id="c1", name="Retrieval-Augmented Generation", type=ConceptType.METHOD),
    ExtractedConcept(local_id="c2", name="Page-Aligned Chunking", type=ConceptType.METHOD),
)


def evidence(page: int = 1, quote: str = "retrieval-augmented") -> tuple[PageEvidence, ...]:
    return (PageEvidence(page=page, quote=quote),)


def build_result(**overrides: object) -> PaperExtraction:
    payload: dict[str, object] = {
        "schema_version": 1,
        "paper_id": 1,
        "summary_ja": "この論文は検索拡張生成のパイプラインを提案する。",
        "concepts": CONCEPTS,
        "paper_concepts": (
            ExtractedPaperConceptEdge(concept_id="c1", relation=PaperConceptRelation.PROPOSES, evidence=evidence()),
        ),
    }
    payload.update(overrides)
    return PaperExtraction.model_validate(payload)


def test_a_result_declares_the_current_schema_version() -> None:
    assert build_result().schema_version == EXTRACTION_SCHEMA_VERSION


def test_another_schema_version_is_rejected() -> None:
    with pytest.raises(ValidationError):
        build_result(schema_version=2)


def test_duplicate_local_ids_are_rejected() -> None:
    duplicate = ExtractedConcept(local_id="c1", name="Dense Retrieval", type=ConceptType.METHOD)
    with pytest.raises(ValidationError, match="'c1' more than once"):
        build_result(concepts=(CONCEPTS[0], duplicate))


@pytest.mark.parametrize("local_id", ["c0", "c01", "concept-1", "1", ""])
def test_local_ids_outside_the_pattern_are_rejected(local_id: str) -> None:
    with pytest.raises(ValidationError):
        ExtractedConcept(local_id=local_id, name="Dense Retrieval", type=ConceptType.METHOD)


def test_paper_concept_edge_naming_an_undeclared_concept_is_rejected_with_that_id() -> None:
    edge = ExtractedPaperConceptEdge(concept_id="c9", relation=PaperConceptRelation.USES, evidence=evidence())
    with pytest.raises(ValidationError, match="undeclared concept 'c9'"):
        build_result(paper_concepts=(edge,))


def test_concept_relation_naming_an_undeclared_concept_is_rejected_with_that_id() -> None:
    edge = ExtractedConceptEdge(
        source_id="c1", target_id="c7", relation=ConceptRelationType.USED_FOR, evidence=evidence()
    )
    with pytest.raises(ValidationError, match="undeclared concept 'c7'"):
        build_result(concept_relations=(edge,))


def test_the_same_concept_and_relation_twice_is_rejected() -> None:
    edge = ExtractedPaperConceptEdge(concept_id="c1", relation=PaperConceptRelation.PROPOSES, evidence=evidence())
    with pytest.raises(ValidationError, match="repeats 'c1' with relation 'proposes'"):
        build_result(paper_concepts=(edge, edge))


def test_the_same_concept_pair_and_relation_twice_is_rejected() -> None:
    edge = ExtractedConceptEdge(
        source_id="c1", target_id="c2", relation=ConceptRelationType.RELATED_TO, evidence=evidence()
    )
    with pytest.raises(ValidationError, match="repeats 'c1' -> 'c2' with relation 'related_to'"):
        build_result(concept_relations=(edge, edge))


def test_the_same_concept_pair_with_another_relation_is_accepted() -> None:
    result = build_result(
        concept_relations=(
            ExtractedConceptEdge(
                source_id="c1", target_id="c2", relation=ConceptRelationType.RELATED_TO, evidence=evidence()
            ),
            ExtractedConceptEdge(
                source_id="c1", target_id="c2", relation=ConceptRelationType.PART_OF, evidence=evidence()
            ),
        )
    )
    assert len(result.concept_relations) == 2


def test_a_relation_from_a_concept_to_itself_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        ExtractedConceptEdge(source_id="c1", target_id="c1", relation=ConceptRelationType.IS_A, evidence=evidence())


def test_an_edge_without_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedPaperConceptEdge(concept_id="c1", relation=PaperConceptRelation.USES, evidence=())


@pytest.mark.parametrize("quote", ["", "   \n "])
def test_a_blank_quote_is_rejected(quote: str) -> None:
    with pytest.raises(ValidationError):
        PageEvidence(page=1, quote=quote)


def test_a_blank_summary_is_rejected() -> None:
    with pytest.raises(ValidationError):
        build_result(summary_ja="  ")


def test_a_result_without_concepts_is_rejected() -> None:
    with pytest.raises(ValidationError):
        build_result(concepts=())


def test_the_schema_describes_the_summary_and_the_local_id_pattern() -> None:
    schema = build_paper_extraction_schema()
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert "summary_ja" in properties
    assert LOCAL_CONCEPT_ID_PATTERN in json.dumps(schema)


def test_the_schema_is_json_serializable() -> None:
    assert json.loads(json.dumps(build_paper_extraction_schema())) == build_paper_extraction_schema()


def test_a_mismatch_error_names_the_paper_and_the_field_of_every_issue() -> None:
    error = PaperExtractionMismatchError(
        {
            1: (
                PaperExtractionIssue(
                    path="paper_concepts[0].evidence[0].quote", message="is not found in the text of page 1"
                ),
            ),
            3: (PaperExtractionIssue(path="paper_id", message="is 3, but the paper being checked is 4"),),
        }
    )
    assert str(error) == (
        "paper 1: paper_concepts[0].evidence[0].quote: is not found in the text of page 1; "
        "paper 3: paper_id: is 3, but the paper being checked is 4"
    )


def test_a_mismatch_error_keeps_the_issues_of_each_paper_as_an_attribute() -> None:
    issues = (PaperExtractionIssue(path="paper_id", message="is 3, but the paper being checked is 4"),)
    error = PaperExtractionMismatchError({3: issues})
    assert error.issues_by_paper == {3: issues}
