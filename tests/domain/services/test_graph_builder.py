import pytest

from rkgk.domain.models.concept_normalization import (
    ConceptNormalization,
    GeneralKnowledgeEdge,
    LocalConceptRef,
    NormalizedConcept,
)
from rkgk.domain.models.graph import Evidence, KnowledgeGraph
from rkgk.domain.models.paper_extraction import (
    ExtractedConcept,
    ExtractedConceptEdge,
    ExtractedEvidence,
    ExtractedPaperConceptEdge,
    PaperExtraction,
)
from rkgk.domain.models.vocabulary import ConceptRelationType, ConceptType, Origin, PaperConceptRelation
from rkgk.domain.services.graph_builder import build_knowledge_graph, paper_node_id, to_networkx

P1_PROPOSES = ExtractedEvidence(page=1, quote="we propose Graph RAG")
P1_USES = ExtractedEvidence(page=2, quote="Graph RAG runs on the index we build")
P1_ADDRESSES = ExtractedEvidence(page=1, quote="answers keep hallucinating")
P1_RELATION = ExtractedEvidence(page=3, quote="Graph RAG is meant to ground the answer")
P2_USES = ExtractedEvidence(page=1, quote="we use GraphRAG as our retriever")
P2_RELATION = ExtractedEvidence(page=2, quote="GraphRAG is a retrieval augmented generation system")

EXTRACTION_1 = PaperExtraction(
    schema_version=1,
    paper_id=1,
    summary_ja="この論文はグラフを使う検索拡張生成を提案する。",
    concepts=(
        ExtractedConcept(local_id="c1", name="Graph RAG", type=ConceptType.METHOD),
        ExtractedConcept(local_id="c2", name="Hallucination", type=ConceptType.PROBLEM),
    ),
    paper_concepts=(
        ExtractedPaperConceptEdge(concept_id="c1", relation=PaperConceptRelation.PROPOSES, evidence=(P1_PROPOSES,)),
        ExtractedPaperConceptEdge(concept_id="c1", relation=PaperConceptRelation.USES, evidence=(P1_USES,)),
        ExtractedPaperConceptEdge(concept_id="c2", relation=PaperConceptRelation.ADDRESSES, evidence=(P1_ADDRESSES,)),
    ),
    concept_relations=(
        ExtractedConceptEdge(
            source_id="c1", target_id="c2", relation=ConceptRelationType.USED_FOR, evidence=(P1_RELATION,)
        ),
    ),
)

EXTRACTION_2 = PaperExtraction(
    schema_version=1,
    paper_id=2,
    summary_ja="この論文は GraphRAG を検索器として使う。",
    concepts=(
        ExtractedConcept(local_id="c1", name="GraphRAG", type=ConceptType.METHOD),
        ExtractedConcept(local_id="c2", name="Retrieval-Augmented Generation", type=ConceptType.METHOD),
    ),
    paper_concepts=(
        ExtractedPaperConceptEdge(concept_id="c1", relation=PaperConceptRelation.USES, evidence=(P2_USES,)),
    ),
    concept_relations=(
        ExtractedConceptEdge(
            source_id="c1", target_id="c2", relation=ConceptRelationType.IS_A, evidence=(P2_RELATION,)
        ),
    ),
)

NORMALIZATION = ConceptNormalization(
    schema_version=1,
    concepts=(
        NormalizedConcept(
            id="graph-rag",
            canonical_name="Graph RAG",
            type=ConceptType.METHOD,
            aliases=("GraphRAG",),
            merged_from=(LocalConceptRef(paper_id=1, local_id="c1"), LocalConceptRef(paper_id=2, local_id="c1")),
        ),
        NormalizedConcept(
            id="hallucination",
            canonical_name="Hallucination",
            type=ConceptType.PROBLEM,
            merged_from=(LocalConceptRef(paper_id=1, local_id="c2"),),
        ),
        NormalizedConcept(
            id="retrieval-augmented-generation",
            canonical_name="Retrieval-Augmented Generation",
            type=ConceptType.METHOD,
            merged_from=(LocalConceptRef(paper_id=2, local_id="c2"),),
        ),
    ),
    concept_relations=(
        GeneralKnowledgeEdge(
            source_id="retrieval-augmented-generation",
            target_id="hallucination",
            relation=ConceptRelationType.USED_FOR,
            rationale="検索拡張生成は幻覚を減らすために使われる。",
        ),
    ),
)


def resolve(paper_id: int, *items: ExtractedEvidence) -> dict[ExtractedEvidence, Evidence]:
    """Stand in for resolve_extraction_evidence: give every quote of a paper a chunk id of its own."""
    return {
        item: Evidence(page=item.page, quote=item.quote, chunk_id=f"{paper_id}:{index}")
        for index, item in enumerate(items)
    }


RESOLVED = {
    1: resolve(1, P1_PROPOSES, P1_USES, P1_ADDRESSES, P1_RELATION),
    2: resolve(2, P2_USES, P2_RELATION),
}


def build() -> KnowledgeGraph:
    return build_knowledge_graph((EXTRACTION_1, EXTRACTION_2), NORMALIZATION, RESOLVED)


MERGED_FIRST = ExtractedEvidence(page=1, quote="we propose Graph RAG")
MERGED_SHARED = ExtractedEvidence(page=1, quote="the contribution of this paper")
MERGED_SECOND = ExtractedEvidence(page=2, quote="GraphRAG, as we call it")
MERGED_RELATION_FIRST = ExtractedEvidence(page=3, quote="Graph RAG grounds the answer")
MERGED_RELATION_SECOND = ExtractedEvidence(page=3, quote="GraphRAG grounds the answer")

MERGED_CONCEPTS = (
    ExtractedConcept(local_id="c1", name="Graph RAG", type=ConceptType.METHOD),
    ExtractedConcept(local_id="c2", name="GraphRAG", type=ConceptType.METHOD),
    ExtractedConcept(local_id="c3", name="Hallucination", type=ConceptType.PROBLEM),
)

MERGED_NORMALIZATION = ConceptNormalization(
    schema_version=1,
    concepts=(
        NormalizedConcept(
            id="graph-rag",
            canonical_name="Graph RAG",
            type=ConceptType.METHOD,
            merged_from=(LocalConceptRef(paper_id=3, local_id="c1"), LocalConceptRef(paper_id=3, local_id="c2")),
        ),
        NormalizedConcept(
            id="hallucination",
            canonical_name="Hallucination",
            type=ConceptType.PROBLEM,
            merged_from=(LocalConceptRef(paper_id=3, local_id="c3"),),
        ),
    ),
)


def build_merged_extraction(
    paper_concepts: tuple[ExtractedPaperConceptEdge, ...] = (),
    concept_relations: tuple[ExtractedConceptEdge, ...] = (),
) -> PaperExtraction:
    return PaperExtraction(
        schema_version=1,
        paper_id=3,
        summary_ja="この論文は同じ手法を二つの綴りで呼ぶ。",
        concepts=MERGED_CONCEPTS,
        paper_concepts=paper_concepts,
        concept_relations=concept_relations,
    )


def test_paper_node_id_prefixes_the_paper_id() -> None:
    assert paper_node_id(3) == "paper:3"


def test_local_concept_ids_are_replaced_by_the_slug_on_paper_edges() -> None:
    graph = build()

    assert [(edge.paper_id, edge.concept_id, edge.relation) for edge in graph.paper_concepts] == [
        (1, "graph-rag", PaperConceptRelation.PROPOSES),
        (1, "graph-rag", PaperConceptRelation.USES),
        (1, "hallucination", PaperConceptRelation.ADDRESSES),
        (2, "graph-rag", PaperConceptRelation.USES),
    ]


def test_local_concept_ids_are_replaced_by_the_slug_on_concept_relations() -> None:
    graph = build()

    assert [(edge.source_id, edge.target_id, edge.relation) for edge in graph.concept_relations] == [
        ("graph-rag", "hallucination", ConceptRelationType.USED_FOR),
        ("graph-rag", "retrieval-augmented-generation", ConceptRelationType.IS_A),
        ("retrieval-augmented-generation", "hallucination", ConceptRelationType.USED_FOR),
    ]


def test_paper_edge_evidence_carries_the_chunk_id_of_the_resolved_evidence() -> None:
    graph = build()

    assert graph.paper_concepts[0].evidence == (Evidence(page=1, quote="we propose Graph RAG", chunk_id="1:0"),)


def test_concept_relation_evidence_carries_the_chunk_id_of_the_resolved_evidence() -> None:
    graph = build()

    assert [item.chunk_id for item in graph.concept_relations[1].evidence] == ["2:1"]


def test_a_relation_of_a_paper_keeps_origin_paper_and_the_paper_it_came_from() -> None:
    edge = build().concept_relations[0]

    assert (edge.origin, edge.paper_id, edge.rationale) == (Origin.PAPER, 1, None)


def test_a_relation_of_general_knowledge_keeps_its_origin_and_rationale() -> None:
    edge = build().concept_relations[2]

    assert edge.origin == Origin.GENERAL_KNOWLEDGE
    assert edge.paper_id is None
    assert edge.evidence == ()
    assert edge.rationale == "検索拡張生成は幻覚を減らすために使われる。"


def test_paper_count_counts_the_distinct_papers_a_concept_was_merged_from() -> None:
    counts = {concept.id: concept.paper_count for concept in build().concepts}

    assert counts == {"graph-rag": 2, "hallucination": 1, "retrieval-augmented-generation": 1}


def test_paper_count_counts_one_paper_once_when_two_of_its_concepts_merged() -> None:
    graph = build_knowledge_graph((build_merged_extraction(),), MERGED_NORMALIZATION, {3: {}})

    assert graph.concepts[0].paper_count == 1


def test_document_frequency_is_the_share_of_the_papers_of_the_graph() -> None:
    graph = build()

    assert graph.document_frequency("graph-rag") == 1.0
    assert graph.document_frequency("hallucination") == 0.5


def test_paper_ids_follow_the_order_of_the_extractions() -> None:
    assert build_knowledge_graph((EXTRACTION_2, EXTRACTION_1), NORMALIZATION, RESOLVED).paper_ids == (2, 1)


def test_concepts_follow_the_order_of_the_normalization() -> None:
    assert [concept.id for concept in build().concepts] == [
        "graph-rag",
        "hallucination",
        "retrieval-augmented-generation",
    ]


def test_concepts_copy_the_name_type_aliases_and_description_of_the_normalization() -> None:
    concept = build().concepts[0]

    assert (concept.canonical_name, concept.type, concept.aliases, concept.description) == (
        "Graph RAG",
        ConceptType.METHOD,
        ("GraphRAG",),
        "",
    )


def test_paper_edges_of_merged_concepts_collapse_into_one_edge_with_the_union_of_evidence() -> None:
    extraction = build_merged_extraction(
        paper_concepts=(
            ExtractedPaperConceptEdge(
                concept_id="c1",
                relation=PaperConceptRelation.PROPOSES,
                evidence=(MERGED_FIRST, MERGED_SHARED),
            ),
            ExtractedPaperConceptEdge(
                concept_id="c2",
                relation=PaperConceptRelation.PROPOSES,
                evidence=(MERGED_SHARED, MERGED_SECOND),
            ),
        )
    )
    resolved = {3: resolve(3, MERGED_FIRST, MERGED_SHARED, MERGED_SECOND)}

    graph = build_knowledge_graph((extraction,), MERGED_NORMALIZATION, resolved)

    assert len(graph.paper_concepts) == 1
    assert [item.chunk_id for item in graph.paper_concepts[0].evidence] == ["3:0", "3:1", "3:2"]


def test_concept_relations_of_merged_concepts_collapse_into_one_edge_with_the_union_of_evidence() -> None:
    extraction = build_merged_extraction(
        concept_relations=(
            ExtractedConceptEdge(
                source_id="c1",
                target_id="c3",
                relation=ConceptRelationType.USED_FOR,
                evidence=(MERGED_RELATION_FIRST,),
            ),
            ExtractedConceptEdge(
                source_id="c2",
                target_id="c3",
                relation=ConceptRelationType.USED_FOR,
                evidence=(MERGED_RELATION_SECOND,),
            ),
        )
    )
    resolved = {3: resolve(3, MERGED_RELATION_FIRST, MERGED_RELATION_SECOND)}

    graph = build_knowledge_graph((extraction,), MERGED_NORMALIZATION, resolved)

    assert len(graph.concept_relations) == 1
    assert [item.chunk_id for item in graph.concept_relations[0].evidence] == ["3:0", "3:1"]


def test_a_concept_relation_whose_two_ends_merged_into_one_slug_is_dropped() -> None:
    extraction = build_merged_extraction(
        concept_relations=(
            ExtractedConceptEdge(
                source_id="c1",
                target_id="c2",
                relation=ConceptRelationType.RELATED_TO,
                evidence=(MERGED_RELATION_FIRST,),
            ),
        )
    )

    graph = build_knowledge_graph((extraction,), MERGED_NORMALIZATION, {3: resolve(3, MERGED_RELATION_FIRST)})

    assert graph.concept_relations == ()


def test_a_local_id_that_no_normalized_concept_merged_is_rejected() -> None:
    normalization = ConceptNormalization(
        schema_version=1,
        concepts=(
            NormalizedConcept(
                id="graph-rag",
                canonical_name="Graph RAG",
                type=ConceptType.METHOD,
                merged_from=(LocalConceptRef(paper_id=1, local_id="c1"),),
            ),
        ),
    )

    with pytest.raises(ValueError, match="paper 1 concept 'c2' is in no normalized concept"):
        build_knowledge_graph((EXTRACTION_1,), normalization, RESOLVED)


def test_evidence_missing_from_the_resolution_of_its_paper_is_rejected() -> None:
    resolved = {1: resolve(1, P1_PROPOSES, P1_USES, P1_ADDRESSES), 2: RESOLVED[2]}

    with pytest.raises(ValueError, match=r"paper 1 page 3: quote 'Graph RAG is meant to ground the answer'"):
        build_knowledge_graph((EXTRACTION_1, EXTRACTION_2), NORMALIZATION, resolved)


def test_a_paper_missing_from_the_resolution_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"paper 2 page 1: quote 'we use GraphRAG as our retriever'"):
        build_knowledge_graph((EXTRACTION_1, EXTRACTION_2), NORMALIZATION, {1: RESOLVED[1]})


def test_to_networkx_gives_every_paper_a_node_of_its_own() -> None:
    view = to_networkx(build())

    assert view.nodes["paper:1"] == {"kind": "paper", "paper_id": 1}
    assert view.nodes["paper:2"] == {"kind": "paper", "paper_id": 2}


def test_to_networkx_puts_the_concept_and_its_document_frequency_on_the_concept_node() -> None:
    view = to_networkx(build())

    assert view.nodes["hallucination"] == {
        "kind": "concept",
        "canonical_name": "Hallucination",
        "type": ConceptType.PROBLEM,
        "paper_count": 1,
        "document_frequency": 0.5,
    }


def test_to_networkx_keys_a_paper_edge_by_its_relation_and_keeps_the_evidence() -> None:
    view = to_networkx(build())

    assert view.get_edge_data("paper:1", "hallucination", "addresses") == {
        "relation": PaperConceptRelation.ADDRESSES,
        "evidence": (Evidence(page=1, quote="answers keep hallucinating", chunk_id="1:2"),),
    }


def test_to_networkx_keys_a_paper_origin_relation_by_relation_origin_and_paper() -> None:
    view = to_networkx(build())

    assert view.get_edge_data("graph-rag", "hallucination", "used_for:paper:1") == {
        "relation": ConceptRelationType.USED_FOR,
        "origin": Origin.PAPER,
        "paper_id": 1,
        "evidence": (Evidence(page=3, quote="Graph RAG is meant to ground the answer", chunk_id="1:3"),),
        "rationale": None,
    }


def test_to_networkx_keys_a_general_knowledge_relation_without_a_paper() -> None:
    view = to_networkx(build())

    assert view.get_edge_data("retrieval-augmented-generation", "hallucination", "used_for:general_knowledge") == {
        "relation": ConceptRelationType.USED_FOR,
        "origin": Origin.GENERAL_KNOWLEDGE,
        "paper_id": None,
        "evidence": (),
        "rationale": "検索拡張生成は幻覚を減らすために使われる。",
    }


def test_to_networkx_keeps_two_parallel_edges_when_a_paper_proposes_and_uses_one_concept() -> None:
    view = to_networkx(build())

    assert sorted(key for _, _, key in view.out_edges("paper:1", keys=True)) == [
        "addresses",
        "proposes",
        "uses",
    ]
    assert view.number_of_edges("paper:1", "graph-rag") == 2
