import pytest
from pydantic import ValidationError

from rkgk.domain.models.graph import Concept, ConceptEdge, Evidence, PaperConceptEdge
from rkgk.domain.models.vocabulary import ConceptRelationType, ConceptType, Origin, PaperConceptRelation

EVIDENCE = Evidence(page=3, quote="we propose a retrieval augmented generation pipeline")


def test_evidence_rejects_blank_quote() -> None:
    with pytest.raises(ValidationError):
        Evidence(page=1, quote="   \n ")


def test_evidence_accepts_optional_chunk_id() -> None:
    assert Evidence(page=1, quote="a quote", chunk_id="1:0").chunk_id == "1:0"


@pytest.mark.parametrize("concept_id", ["Graph-RAG", "graph-rag-", "-graph-rag", "graph rag", "graph--rag", ""])
def test_concept_rejects_invalid_slug(concept_id: str) -> None:
    with pytest.raises(ValidationError):
        Concept(id=concept_id, canonical_name="Graph RAG", type=ConceptType.METHOD, paper_count=1)


def test_concept_accepts_slug_and_defaults_to_no_aliases() -> None:
    concept = Concept(id="graph-rag2", canonical_name="Graph RAG", type=ConceptType.METHOD, paper_count=1)
    assert concept.aliases == ()
    assert concept.description == ""


def test_paper_concept_requires_at_least_one_evidence() -> None:
    with pytest.raises(ValidationError):
        PaperConceptEdge(paper_id=1, concept_id="graph-rag", relation=PaperConceptRelation.PROPOSES, evidence=[])


def test_paper_concept_accepts_evidence() -> None:
    paper_concept = PaperConceptEdge(
        paper_id=1,
        concept_id="graph-rag",
        relation=PaperConceptRelation.PROPOSES,
        evidence=[EVIDENCE],
    )
    assert paper_concept.evidence == (EVIDENCE,)


def test_concept_relation_rejects_self_relation() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        ConceptEdge(
            source_id="graph-rag",
            target_id="graph-rag",
            relation=ConceptRelationType.IS_A,
            origin=Origin.PAPER,
            paper_id=1,
            evidence=[EVIDENCE],
        )


def test_concept_relation_from_paper_requires_paper_id() -> None:
    with pytest.raises(ValidationError, match="paper_id is required when origin is paper"):
        ConceptEdge(
            source_id="graph-rag",
            target_id="rag",
            relation=ConceptRelationType.IS_A,
            origin=Origin.PAPER,
            evidence=[EVIDENCE],
        )


def test_concept_relation_from_paper_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence must not be empty when origin is paper"):
        ConceptEdge(
            source_id="graph-rag",
            target_id="rag",
            relation=ConceptRelationType.IS_A,
            origin=Origin.PAPER,
            paper_id=1,
        )


def test_concept_relation_from_general_knowledge_rejects_paper_id() -> None:
    with pytest.raises(ValidationError, match="paper_id must be None when origin is general_knowledge"):
        ConceptEdge(
            source_id="graph-rag",
            target_id="rag",
            relation=ConceptRelationType.IS_A,
            origin=Origin.GENERAL_KNOWLEDGE,
            paper_id=1,
        )


def test_concept_relation_from_general_knowledge_rejects_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence must be empty when origin is general_knowledge"):
        ConceptEdge(
            source_id="graph-rag",
            target_id="rag",
            relation=ConceptRelationType.IS_A,
            origin=Origin.GENERAL_KNOWLEDGE,
            evidence=[EVIDENCE],
        )


def test_concept_relation_accepts_paper_origin_with_paper_id_and_evidence() -> None:
    relation = ConceptEdge(
        source_id="graph-rag",
        target_id="rag",
        relation=ConceptRelationType.IS_A,
        origin=Origin.PAPER,
        paper_id=1,
        evidence=[EVIDENCE],
    )
    assert relation.paper_id == 1


def test_concept_relation_accepts_general_knowledge_without_paper_id_or_evidence() -> None:
    relation = ConceptEdge(
        source_id="graph-rag",
        target_id="rag",
        relation=ConceptRelationType.IS_A,
        origin=Origin.GENERAL_KNOWLEDGE,
        rationale="Graph RAG は RAG の検索段階をグラフ探索に置き換えたものである。",
    )
    assert relation.paper_id is None
    assert relation.evidence == ()


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Evidence(page=1, quote="a quote", pages=2)  # ty: ignore[unknown-argument]


def test_models_are_frozen() -> None:
    evidence = Evidence(page=1, quote="a quote")
    with pytest.raises(ValidationError):
        evidence.page = 2  # ty: ignore[invalid-assignment]


def test_concept_edge_rejects_non_positive_paper_id() -> None:
    with pytest.raises(ValidationError):
        ConceptEdge(
            source_id="graph-rag",
            target_id="retrieval-augmented-generation",
            relation=ConceptRelationType.IS_A,
            origin=Origin.PAPER,
            paper_id=0,
            evidence=[EVIDENCE],
        )


def test_evidence_sequences_are_immutable_after_creation() -> None:
    edge = PaperConceptEdge(
        paper_id=1, concept_id="graph-rag", relation=PaperConceptRelation.PROPOSES, evidence=[EVIDENCE]
    )
    assert isinstance(edge.evidence, tuple)
    assert not hasattr(edge.evidence, "append")
