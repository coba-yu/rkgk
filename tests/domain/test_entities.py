from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from rkgk.domain.entities import (
    Chunk,
    Concept,
    ConceptRelation,
    Evidence,
    PaperConcept,
    PaperMeta,
    PaperPreprocessInfo,
)
from rkgk.domain.vocabulary import ConceptRelationType, ConceptType, Origin, PaperConceptRelation

PREPROCESS = PaperPreprocessInfo(tool="pymupdf", version="1.24.0", processed_at=datetime(2026, 1, 1, tzinfo=UTC))
EVIDENCE = Evidence(page=3, quote="we propose a retrieval augmented generation pipeline")


def make_paper_meta(paper_id: int) -> PaperMeta:
    return PaperMeta(
        id=paper_id,
        title="Retrieval Augmented Generation",
        authors=["Ada Lovelace"],
        year=2026,
        venue="NeurIPS",
        page_count=12,
        preprocess=PREPROCESS,
    )


@pytest.mark.parametrize(("paper_id", "expected"), [(1, "0001"), (42, "0042"), (9999, "9999"), (12345, "12345")])
def test_paper_meta_dir_name_is_zero_padded_without_truncation(paper_id: int, expected: str) -> None:
    assert make_paper_meta(paper_id).dir_name == expected


def test_paper_meta_rejects_id_below_one() -> None:
    with pytest.raises(ValidationError):
        make_paper_meta(0)


def test_evidence_rejects_blank_quote() -> None:
    with pytest.raises(ValidationError):
        Evidence(page=1, quote="   \n ")


def test_evidence_accepts_optional_chunk_id() -> None:
    assert Evidence(page=1, quote="a quote", chunk_id="1:0").chunk_id == "1:0"


@pytest.mark.parametrize("concept_id", ["Graph-RAG", "graph-rag-", "-graph-rag", "graph rag", "graph--rag", ""])
def test_concept_rejects_invalid_slug(concept_id: str) -> None:
    with pytest.raises(ValidationError):
        Concept(id=concept_id, canonical_name="Graph RAG", type=ConceptType.METHOD)


def test_concept_accepts_slug_and_defaults_to_no_aliases() -> None:
    concept = Concept(id="graph-rag2", canonical_name="Graph RAG", type=ConceptType.METHOD)
    assert concept.aliases == []
    assert concept.description == ""


def test_paper_concept_requires_at_least_one_evidence() -> None:
    with pytest.raises(ValidationError):
        PaperConcept(paper_id=1, concept_id="graph-rag", relation=PaperConceptRelation.PROPOSES, evidence=[])


def test_paper_concept_accepts_evidence() -> None:
    paper_concept = PaperConcept(
        paper_id=1,
        concept_id="graph-rag",
        relation=PaperConceptRelation.PROPOSES,
        evidence=[EVIDENCE],
    )
    assert paper_concept.evidence == [EVIDENCE]


def test_concept_relation_rejects_self_relation() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        ConceptRelation(
            source_id="graph-rag",
            target_id="graph-rag",
            relation=ConceptRelationType.IS_A,
            origin=Origin.PAPER,
            paper_id=1,
            evidence=[EVIDENCE],
        )


def test_concept_relation_from_paper_requires_paper_id() -> None:
    with pytest.raises(ValidationError, match="paper_id is required when origin is paper"):
        ConceptRelation(
            source_id="graph-rag",
            target_id="rag",
            relation=ConceptRelationType.IS_A,
            origin=Origin.PAPER,
            evidence=[EVIDENCE],
        )


def test_concept_relation_from_paper_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence must not be empty when origin is paper"):
        ConceptRelation(
            source_id="graph-rag",
            target_id="rag",
            relation=ConceptRelationType.IS_A,
            origin=Origin.PAPER,
            paper_id=1,
        )


def test_concept_relation_from_general_knowledge_rejects_paper_id() -> None:
    with pytest.raises(ValidationError, match="paper_id must be None when origin is general_knowledge"):
        ConceptRelation(
            source_id="graph-rag",
            target_id="rag",
            relation=ConceptRelationType.IS_A,
            origin=Origin.GENERAL_KNOWLEDGE,
            paper_id=1,
        )


def test_concept_relation_from_general_knowledge_rejects_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence must be empty when origin is general_knowledge"):
        ConceptRelation(
            source_id="graph-rag",
            target_id="rag",
            relation=ConceptRelationType.IS_A,
            origin=Origin.GENERAL_KNOWLEDGE,
            evidence=[EVIDENCE],
        )


def test_concept_relation_accepts_paper_origin_with_paper_id_and_evidence() -> None:
    relation = ConceptRelation(
        source_id="graph-rag",
        target_id="rag",
        relation=ConceptRelationType.IS_A,
        origin=Origin.PAPER,
        paper_id=1,
        evidence=[EVIDENCE],
    )
    assert relation.paper_id == 1


def test_concept_relation_accepts_general_knowledge_without_paper_id_or_evidence() -> None:
    relation = ConceptRelation(
        source_id="graph-rag",
        target_id="rag",
        relation=ConceptRelationType.IS_A,
        origin=Origin.GENERAL_KNOWLEDGE,
    )
    assert relation.paper_id is None
    assert relation.evidence == []


def test_chunk_make_builds_the_id() -> None:
    chunk = Chunk.create(paper_id=7, index_in_paper=3, page_start=2, page_end=3, text="body text")
    assert chunk.id == "7:3"


def test_chunk_rejects_page_end_before_page_start() -> None:
    with pytest.raises(ValidationError, match="page_end"):
        Chunk(id="1:0", paper_id=1, index_in_paper=0, page_start=5, page_end=4, text="body text")


def test_chunk_rejects_mismatched_id() -> None:
    with pytest.raises(ValidationError, match="id must be"):
        Chunk(id="2:0", paper_id=1, index_in_paper=0, page_start=1, page_end=1, text="body text")


def test_chunk_rejects_blank_text() -> None:
    with pytest.raises(ValidationError):
        Chunk(id="1:0", paper_id=1, index_in_paper=0, page_start=1, page_end=1, text="  ")


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Evidence(page=1, quote="a quote", pages=2)  # ty: ignore[unknown-argument]


def test_models_are_frozen() -> None:
    evidence = Evidence(page=1, quote="a quote")
    with pytest.raises(ValidationError):
        evidence.page = 2  # ty: ignore[invalid-assignment]
