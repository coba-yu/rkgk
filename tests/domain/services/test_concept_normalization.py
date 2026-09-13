from rkgk.domain.models.concept_normalization import (
    ConceptMerge,
    ConceptNormalization,
    GeneralKnowledgeEdge,
    GeneralKnowledgeProposal,
    LocalConceptRef,
    NormalizedConcept,
)
from rkgk.domain.models.paper_extraction import ExtractedConcept, PaperExtraction
from rkgk.domain.models.vocabulary import ConceptRelationType, ConceptType
from rkgk.domain.services.concept_normalization import (
    check_merge_against_extractions,
    check_normalization_against_extractions,
    check_relations_against_concepts,
)

EXTRACTIONS = (
    PaperExtraction(
        schema_version=1,
        paper_id=1,
        summary_ja="この論文は検索拡張生成のパイプラインを提案する。",
        concepts=(
            ExtractedConcept(
                local_id="c1",
                name="Retrieval-Augmented Generation",
                type=ConceptType.METHOD,
                aliases=("RAG", "検索拡張生成"),
                description="Generation grounded in retrieved passages.",
            ),
            ExtractedConcept(local_id="c2", name="Page-Aligned Chunking", type=ConceptType.METHOD),
        ),
        paper_concepts=(),
    ),
    PaperExtraction(
        schema_version=1,
        paper_id=2,
        summary_ja="この論文は知識グラフを検索の骨格として使う。",
        concepts=(
            ExtractedConcept(local_id="c1", name="RAG", type=ConceptType.METHOD),
            ExtractedConcept(
                local_id="c2",
                name="Knowledge Graph",
                type=ConceptType.METHOD,
                aliases=("KG",),
                description="A graph of concepts and their relations.",
            ),
        ),
        paper_concepts=(),
    ),
)

RAG = NormalizedConcept(
    id="retrieval-augmented-generation",
    canonical_name="Retrieval-Augmented Generation",
    type=ConceptType.METHOD,
    aliases=("RAG", "検索拡張生成"),
    merged_from=(LocalConceptRef(paper_id=1, local_id="c1"), LocalConceptRef(paper_id=2, local_id="c1")),
)
CHUNKING = NormalizedConcept(
    id="page-aligned-chunking",
    canonical_name="Page-Aligned Chunking",
    type=ConceptType.METHOD,
    merged_from=(LocalConceptRef(paper_id=1, local_id="c2"),),
)
KNOWLEDGE_GRAPH = NormalizedConcept(
    id="knowledge-graph",
    canonical_name="Knowledge Graph",
    type=ConceptType.METHOD,
    aliases=("KG",),
    description="A graph of concepts and their relations.",
    merged_from=(LocalConceptRef(paper_id=2, local_id="c2"),),
)

PART_OF = GeneralKnowledgeEdge(
    source_id="page-aligned-chunking",
    target_id="retrieval-augmented-generation",
    relation=ConceptRelationType.PART_OF,
    rationale="Chunking is the indexing step of a retrieval-augmented generation pipeline.",
)


def build_normalization(**overrides: object) -> ConceptNormalization:
    payload: dict[str, object] = {
        "schema_version": 1,
        "concepts": (RAG, CHUNKING, KNOWLEDGE_GRAPH),
        "concept_relations": (PART_OF,),
    }
    payload.update(overrides)
    return ConceptNormalization.model_validate(payload)


def test_a_normalization_that_covers_every_extracted_concept_is_accepted() -> None:
    assert check_normalization_against_extractions(build_normalization(), EXTRACTIONS) == ()


def test_a_reference_to_a_concept_the_paper_never_extracted_is_reported_with_its_path() -> None:
    invented = KNOWLEDGE_GRAPH.model_copy(update={"merged_from": (LocalConceptRef(paper_id=2, local_id="c9"),)})
    issues = check_normalization_against_extractions(
        build_normalization(concepts=(RAG, CHUNKING, invented)), EXTRACTIONS
    )
    assert [issue.path for issue in issues] == ["concepts[2].merged_from[0]", "concepts"]
    assert issues[0].message == "paper 2 has no concept 'c9'"


def test_a_reference_to_a_paper_that_was_not_extracted_is_reported() -> None:
    invented = KNOWLEDGE_GRAPH.model_copy(update={"merged_from": (LocalConceptRef(paper_id=9, local_id="c1"),)})
    issues = check_normalization_against_extractions(
        build_normalization(concepts=(RAG, CHUNKING, invented)), EXTRACTIONS
    )
    assert issues[0].path == "concepts[2].merged_from[0]"
    assert issues[0].message == "there is no extraction for paper 9"


def test_an_extracted_concept_that_no_normalized_concept_covers_is_reported_with_its_paper_and_local_id() -> None:
    issues = check_normalization_against_extractions(
        build_normalization(concepts=(RAG, CHUNKING), concept_relations=()), EXTRACTIONS
    )
    assert [issue.path for issue in issues] == ["concepts"]
    assert issues[0].message == "paper 2 'c2' is in no merged_from"


def test_a_type_that_none_of_the_merged_concepts_has_is_reported_with_its_path() -> None:
    retyped = KNOWLEDGE_GRAPH.model_copy(update={"type": ConceptType.PROBLEM})
    issues = check_normalization_against_extractions(
        build_normalization(concepts=(RAG, CHUNKING, retyped)), EXTRACTIONS
    )
    assert [issue.path for issue in issues] == ["concepts[2].type"]
    assert issues[0].message == "is 'problem', which none of the merged concepts has; they are method"


def test_every_issue_is_reported_instead_of_only_the_first() -> None:
    broken = KNOWLEDGE_GRAPH.model_copy(
        update={
            "type": ConceptType.KEYWORD,
            "merged_from": (LocalConceptRef(paper_id=2, local_id="c2"), LocalConceptRef(paper_id=1, local_id="c7")),
        }
    )
    issues = check_normalization_against_extractions(
        build_normalization(concepts=(RAG, broken), concept_relations=()), EXTRACTIONS
    )
    assert [issue.path for issue in issues] == ["concepts[1].merged_from[1]", "concepts[1].type", "concepts"]
    assert issues[2].message == "paper 1 'c2' is in no merged_from"


def test_a_merge_that_covers_every_extracted_concept_is_accepted() -> None:
    merge = ConceptMerge(concepts=(RAG, CHUNKING, KNOWLEDGE_GRAPH))
    assert check_merge_against_extractions(merge, EXTRACTIONS) == ()


def test_a_merge_is_read_the_same_way_as_the_normalization_assembled_from_it() -> None:
    merge = ConceptMerge(concepts=(RAG, CHUNKING))
    assert check_merge_against_extractions(merge, EXTRACTIONS) == check_normalization_against_extractions(
        build_normalization(concepts=(RAG, CHUNKING), concept_relations=()), EXTRACTIONS
    )


def test_a_merge_that_invents_a_reference_is_reported_with_its_path() -> None:
    invented = KNOWLEDGE_GRAPH.model_copy(update={"merged_from": (LocalConceptRef(paper_id=2, local_id="c9"),)})
    issues = check_merge_against_extractions(ConceptMerge(concepts=(RAG, CHUNKING, invented)), EXTRACTIONS)
    assert [issue.path for issue in issues] == ["concepts[2].merged_from[0]", "concepts"]
    assert issues[0].message == "paper 2 has no concept 'c9'"


def test_relations_between_declared_concepts_are_accepted() -> None:
    proposal = GeneralKnowledgeProposal(concept_relations=(PART_OF,))
    assert check_relations_against_concepts(proposal, (RAG, CHUNKING, KNOWLEDGE_GRAPH)) == ()


def test_a_proposal_without_relations_is_accepted() -> None:
    assert check_relations_against_concepts(GeneralKnowledgeProposal(), (RAG, CHUNKING)) == ()


def test_a_relation_on_a_slug_no_concept_declares_is_reported_with_that_slug() -> None:
    edge = PART_OF.model_copy(update={"target_id": "dense-retrieval"})
    issues = check_relations_against_concepts(
        GeneralKnowledgeProposal(concept_relations=(edge,)), (RAG, CHUNKING, KNOWLEDGE_GRAPH)
    )
    assert [issue.path for issue in issues] == ["concept_relations[0].target_id"]
    assert issues[0].message == "is 'dense-retrieval', which is not one of the normalized concepts"


def test_both_ends_of_a_relation_are_reported_instead_of_only_the_first() -> None:
    edge = PART_OF.model_copy(update={"source_id": "dense-retrieval", "target_id": "sparse-retrieval"})
    issues = check_relations_against_concepts(
        GeneralKnowledgeProposal(concept_relations=(PART_OF, edge)), (RAG, CHUNKING, KNOWLEDGE_GRAPH)
    )
    assert [issue.path for issue in issues] == [
        "concept_relations[1].source_id",
        "concept_relations[1].target_id",
    ]
