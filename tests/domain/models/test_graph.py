import pytest
from pydantic import ValidationError

from rkgk.domain.models.graph import Concept, ConceptEdge, Evidence, KnowledgeGraph, PaperConceptEdge
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


GRAPH_RAG = Concept(id="graph-rag", canonical_name="Graph RAG", type=ConceptType.METHOD, paper_count=1)
RAG = Concept(id="rag", canonical_name="RAG", type=ConceptType.METHOD, paper_count=1)

PAPER_EDGE = PaperConceptEdge(
    paper_id=1, concept_id="graph-rag", relation=PaperConceptRelation.PROPOSES, evidence=[EVIDENCE]
)
PAPER_RELATION = ConceptEdge(
    source_id="graph-rag",
    target_id="rag",
    relation=ConceptRelationType.IS_A,
    origin=Origin.PAPER,
    paper_id=1,
    evidence=[EVIDENCE],
)
GENERAL_RELATION = ConceptEdge(
    source_id="graph-rag",
    target_id="rag",
    relation=ConceptRelationType.IS_A,
    origin=Origin.GENERAL_KNOWLEDGE,
    rationale="Graph RAG は RAG の検索段階をグラフ探索に置き換えたものである。",
)


def build_graph(
    paper_ids: tuple[int, ...] = (1,),
    concepts: tuple[Concept, ...] = (GRAPH_RAG, RAG),
    paper_concepts: tuple[PaperConceptEdge, ...] = (),
    concept_relations: tuple[ConceptEdge, ...] = (),
) -> KnowledgeGraph:
    return KnowledgeGraph(
        paper_ids=paper_ids,
        concepts=concepts,
        paper_concepts=paper_concepts,
        concept_relations=concept_relations,
    )


def test_concept_relation_from_paper_rejects_a_rationale() -> None:
    with pytest.raises(ValidationError, match="rationale must be empty when origin is paper"):
        ConceptEdge(
            source_id="graph-rag",
            target_id="rag",
            relation=ConceptRelationType.IS_A,
            origin=Origin.PAPER,
            paper_id=1,
            evidence=[EVIDENCE],
            rationale="Graph RAG は RAG の一種である。",
        )


def test_concept_relation_from_general_knowledge_rejects_a_blank_rationale() -> None:
    with pytest.raises(ValidationError, match="rationale must contain non-whitespace characters"):
        ConceptEdge(
            source_id="graph-rag",
            target_id="rag",
            relation=ConceptRelationType.IS_A,
            origin=Origin.GENERAL_KNOWLEDGE,
            rationale="  \n ",
        )


def test_knowledge_graph_rejects_repeated_paper_ids() -> None:
    with pytest.raises(ValidationError, match="paper_ids must not repeat"):
        build_graph(paper_ids=(1, 1))


def test_knowledge_graph_rejects_a_concept_declared_twice() -> None:
    with pytest.raises(ValidationError, match="concepts declares 'graph-rag' more than once"):
        build_graph(concepts=(GRAPH_RAG, GRAPH_RAG))


def test_knowledge_graph_rejects_a_concept_counting_more_papers_than_the_graph_has() -> None:
    crowded = Concept(id="rag", canonical_name="RAG", type=ConceptType.METHOD, paper_count=2)

    with pytest.raises(ValidationError, match="concept 'rag' counts 2 papers but the graph has 1"):
        build_graph(concepts=(GRAPH_RAG, crowded))


def test_knowledge_graph_rejects_a_paper_edge_of_an_undeclared_paper() -> None:
    edge = PaperConceptEdge(
        paper_id=2, concept_id="graph-rag", relation=PaperConceptRelation.PROPOSES, evidence=[EVIDENCE]
    )

    with pytest.raises(ValidationError, match="paper_concepts refers to the undeclared paper 2"):
        build_graph(paper_concepts=(edge,))


def test_knowledge_graph_rejects_a_paper_edge_of_an_undeclared_concept() -> None:
    edge = PaperConceptEdge(
        paper_id=1, concept_id="graph-of-thought", relation=PaperConceptRelation.PROPOSES, evidence=[EVIDENCE]
    )

    with pytest.raises(ValidationError, match="paper_concepts refers to the undeclared concept 'graph-of-thought'"):
        build_graph(paper_concepts=(edge,))


def test_knowledge_graph_rejects_the_same_paper_concept_relation_twice() -> None:
    with pytest.raises(ValidationError, match="paper_concepts repeats paper 1 -> 'graph-rag'"):
        build_graph(paper_concepts=(PAPER_EDGE, PAPER_EDGE))


def test_knowledge_graph_rejects_a_concept_relation_of_an_undeclared_concept() -> None:
    edge = ConceptEdge(
        source_id="graph-rag",
        target_id="graph-of-thought",
        relation=ConceptRelationType.IS_A,
        origin=Origin.PAPER,
        paper_id=1,
        evidence=[EVIDENCE],
    )

    with pytest.raises(ValidationError, match="concept_relations refers to the undeclared concept"):
        build_graph(concept_relations=(edge,))


def test_knowledge_graph_rejects_the_same_concept_relation_of_the_same_paper_twice() -> None:
    with pytest.raises(ValidationError, match="concept_relations repeats 'graph-rag' -> 'rag'"):
        build_graph(concept_relations=(PAPER_RELATION, PAPER_RELATION))


def test_knowledge_graph_accepts_the_same_relation_from_a_paper_and_from_general_knowledge() -> None:
    graph = build_graph(concept_relations=(PAPER_RELATION, GENERAL_RELATION))

    assert [edge.origin for edge in graph.concept_relations] == [Origin.PAPER, Origin.GENERAL_KNOWLEDGE]


def test_knowledge_graph_accepts_the_edges_of_its_declared_nodes() -> None:
    graph = build_graph(paper_concepts=(PAPER_EDGE,), concept_relations=(PAPER_RELATION,))

    assert graph.paper_concepts == (PAPER_EDGE,)
    assert graph.concept_relations == (PAPER_RELATION,)


def test_document_frequency_is_the_paper_count_over_the_papers_of_the_graph() -> None:
    shared = Concept(id="rag", canonical_name="RAG", type=ConceptType.METHOD, paper_count=2)
    graph = build_graph(paper_ids=(1, 2, 3, 4), concepts=(GRAPH_RAG, shared))

    assert graph.document_frequency("rag") == 0.5
    assert graph.document_frequency("graph-rag") == 0.25
