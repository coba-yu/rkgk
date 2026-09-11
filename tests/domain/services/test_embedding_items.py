from rkgk.domain.models.chunk import Chunk
from rkgk.domain.models.concept_normalization import LocalConceptRef, NormalizedConcept
from rkgk.domain.models.embedding import EmbeddedItem, EmbeddedItemKind
from rkgk.domain.models.paper_extraction import (
    ExtractedConcept,
    ExtractedPaperConceptEdge,
    PageEvidence,
    PaperExtraction,
)
from rkgk.domain.models.vocabulary import ConceptType, PaperConceptRelation
from rkgk.domain.services.embedding_items import build_embedding_items, concept_embedding_text

CHUNKS = (
    Chunk.create(paper_id=1, idx=0, page_start=1, page_end=1, text="We study retrieval."),
    Chunk.create(paper_id=1, idx=1, page_start=2, page_end=2, text="It helps a reader."),
    Chunk.create(paper_id=2, idx=0, page_start=1, page_end=1, text="We build a graph."),
)


def build_extraction(paper_id: int, summary_ja: str) -> PaperExtraction:
    return PaperExtraction(
        schema_version=1,
        paper_id=paper_id,
        summary_ja=summary_ja,
        concepts=(ExtractedConcept(local_id="c1", name="Retrieval", type=ConceptType.METHOD),),
        paper_concepts=(
            ExtractedPaperConceptEdge(
                concept_id="c1",
                relation=PaperConceptRelation.PROPOSES,
                evidence=(PageEvidence(page=1, quote="We study"),),
            ),
        ),
    )


EXTRACTIONS = (build_extraction(1, "検索を研究する。"), build_extraction(2, "グラフを作る。"))

RAG = NormalizedConcept(
    id="retrieval-augmented-generation",
    canonical_name="Retrieval-Augmented Generation",
    type=ConceptType.METHOD,
    aliases=("RAG", "検索拡張生成"),
    description="Generation conditioned on retrieved passages.",
    merged_from=(LocalConceptRef(paper_id=1, local_id="c1"),),
)
GRAPH = NormalizedConcept(
    id="knowledge-graph",
    canonical_name="Knowledge Graph",
    type=ConceptType.METHOD,
    merged_from=(LocalConceptRef(paper_id=2, local_id="c1"),),
)


def test_items_come_out_as_chunks_then_summaries_then_concepts_in_the_given_order() -> None:
    items = build_embedding_items(CHUNKS, EXTRACTIONS, (RAG, GRAPH))
    assert [(item.kind, item.ref) for item in items] == [
        (EmbeddedItemKind.CHUNK, "1:0"),
        (EmbeddedItemKind.CHUNK, "1:1"),
        (EmbeddedItemKind.CHUNK, "2:0"),
        (EmbeddedItemKind.SUMMARY, "1"),
        (EmbeddedItemKind.SUMMARY, "2"),
        (EmbeddedItemKind.CONCEPT, "retrieval-augmented-generation"),
        (EmbeddedItemKind.CONCEPT, "knowledge-graph"),
    ]


def test_a_chunk_item_carries_the_chunk_text_and_its_paper() -> None:
    items = build_embedding_items(CHUNKS[:1], (), ())
    assert items == (EmbeddedItem(kind=EmbeddedItemKind.CHUNK, ref="1:0", paper_id=1, text="We study retrieval."),)


def test_a_summary_item_carries_the_japanese_summary_and_its_paper() -> None:
    items = build_embedding_items((), EXTRACTIONS[:1], ())
    assert items == (EmbeddedItem(kind=EmbeddedItemKind.SUMMARY, ref="1", paper_id=1, text="検索を研究する。"),)


def test_a_concept_item_carries_the_rendered_concept_and_no_paper() -> None:
    items = build_embedding_items((), (), (GRAPH,))
    assert items == (EmbeddedItem(kind=EmbeddedItemKind.CONCEPT, ref="knowledge-graph", text="Knowledge Graph"),)


def test_a_concept_text_puts_the_aliases_after_the_name_and_the_description_on_its_own_line() -> None:
    assert concept_embedding_text(RAG) == (
        "Retrieval-Augmented Generation (RAG, 検索拡張生成)\nGeneration conditioned on retrieved passages."
    )


def test_a_concept_text_without_aliases_or_description_is_just_the_name() -> None:
    assert concept_embedding_text(GRAPH) == "Knowledge Graph"


def test_nothing_to_embed_gives_no_items() -> None:
    assert build_embedding_items((), (), ()) == ()
