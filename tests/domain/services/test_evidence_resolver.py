import pytest

from rkgk.domain.models.chunk import Chunk
from rkgk.domain.models.graph import Evidence, EvidenceResolutionError
from rkgk.domain.models.paper_extraction import (
    ExtractedConcept,
    ExtractedConceptEdge,
    ExtractedEvidence,
    ExtractedPaperConceptEdge,
    PaperExtraction,
)
from rkgk.domain.models.vocabulary import ConceptRelationType, ConceptType, PaperConceptRelation
from rkgk.domain.services.evidence_resolver import find_chunk, resolve_extraction_evidence

CHUNKS = (
    Chunk.create(
        paper_id=1, idx=0, page_start=1, page_end=1, text="We study a retrieval-augmented\ngeneration pipeline."
    ),
    Chunk.create(paper_id=1, idx=1, page_start=1, page_end=1, text="It helps a reader choose the next paper."),
    Chunk.create(paper_id=1, idx=2, page_start=2, page_end=2, text="The pipeline splits every paper into chunks."),
    Chunk.create(paper_id=1, idx=3, page_start=2, page_end=2, text="The pipeline splits every paper again."),
)

CONCEPTS = (
    ExtractedConcept(local_id="c1", name="Retrieval-Augmented Generation", type=ConceptType.METHOD),
    ExtractedConcept(local_id="c2", name="Chunking", type=ConceptType.METHOD),
)


def paper_edge(*evidence: ExtractedEvidence, concept_id: str = "c1") -> ExtractedPaperConceptEdge:
    return ExtractedPaperConceptEdge(concept_id=concept_id, relation=PaperConceptRelation.PROPOSES, evidence=evidence)


def build_extraction(
    *paper_concepts: ExtractedPaperConceptEdge, concept_relations: tuple[ExtractedConceptEdge, ...] = ()
) -> PaperExtraction:
    return PaperExtraction(
        schema_version=1,
        paper_id=1,
        summary_ja="この論文は検索拡張生成のパイプラインを提案する。",
        concepts=CONCEPTS,
        paper_concepts=paper_concepts,
        concept_relations=concept_relations,
    )


def test_find_chunk_returns_the_chunk_of_the_page_that_holds_the_quote() -> None:
    assert find_chunk(1, "choose the next paper", CHUNKS) == CHUNKS[1]


def test_find_chunk_matches_after_whitespace_normalization() -> None:
    assert find_chunk(1, "  retrieval-augmented   generation\tpipeline ", CHUNKS) == CHUNKS[0]


def test_find_chunk_returns_the_first_of_several_matching_chunks() -> None:
    assert find_chunk(2, "The pipeline splits every paper", CHUNKS) == CHUNKS[2]


def test_find_chunk_ignores_chunks_of_other_pages() -> None:
    assert find_chunk(2, "choose the next paper", CHUNKS) is None


def test_find_chunk_returns_none_for_a_quote_that_straddles_two_chunks() -> None:
    assert find_chunk(1, "generation pipeline. It helps", CHUNKS) is None


def test_resolved_evidence_keeps_page_and_quote_and_adds_the_chunk_id() -> None:
    item = ExtractedEvidence(page=1, quote="choose the next paper")

    resolved = resolve_extraction_evidence(build_extraction(paper_edge(item)), CHUNKS)

    assert resolved == {item: Evidence(page=1, quote="choose the next paper", chunk_id="1:1")}


def test_evidence_of_concept_relations_is_resolved_too() -> None:
    item = ExtractedEvidence(page=2, quote="splits every paper into chunks")
    relation = ExtractedConceptEdge(
        source_id="c1", target_id="c2", relation=ConceptRelationType.USED_FOR, evidence=(item,)
    )

    resolved = resolve_extraction_evidence(build_extraction(concept_relations=(relation,)), CHUNKS)

    assert resolved[item].chunk_id == "1:2"


def test_the_same_quote_on_two_edges_is_resolved_to_one_entry() -> None:
    item = ExtractedEvidence(page=1, quote="a reader")

    resolved = resolve_extraction_evidence(
        build_extraction(paper_edge(item), paper_edge(item, concept_id="c2")), CHUNKS
    )

    assert list(resolved) == [item]


def test_a_quote_that_straddles_two_chunks_is_reported_with_paper_page_and_quote() -> None:
    item = ExtractedEvidence(page=1, quote="generation pipeline. It helps")

    with pytest.raises(EvidenceResolutionError) as raised:
        resolve_extraction_evidence(build_extraction(paper_edge(item)), CHUNKS)

    assert [(u.paper_id, u.page, u.quote) for u in raised.value.unresolved] == [
        (1, 1, "generation pipeline. It helps")
    ]
    assert str(raised.value) == "paper 1 page 1: quote 'generation pipeline. It helps'"


def test_every_unresolved_quote_is_reported_once_instead_of_only_the_first() -> None:
    first = ExtractedEvidence(page=1, quote="generation pipeline. It helps")
    second = ExtractedEvidence(page=2, quote="not in the paper")
    found = ExtractedEvidence(page=2, quote="into chunks")

    with pytest.raises(EvidenceResolutionError) as raised:
        resolve_extraction_evidence(
            build_extraction(paper_edge(first, found, second), paper_edge(first, concept_id="c2")), CHUNKS
        )

    assert [u.quote for u in raised.value.unresolved] == [first.quote, second.quote]


def test_chunks_of_another_paper_are_rejected() -> None:
    item = ExtractedEvidence(page=1, quote="a reader")
    foreign = Chunk.create(paper_id=2, idx=0, page_start=1, page_end=1, text="a reader")

    with pytest.raises(ValueError, match=r"\['2:0'\] do not belong to paper 1"):
        resolve_extraction_evidence(build_extraction(paper_edge(item)), (*CHUNKS, foreign))
