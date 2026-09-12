import numpy as np
import pytest
from pydantic import ValidationError

from rkgk.domain.models.chunk import Chunk
from rkgk.domain.models.embedding import EmbeddedItem, EmbeddedItemKind, EmbeddingTable
from rkgk.domain.models.graph import ChunkEvidence, Concept, KnowledgeGraph, PaperConceptEdge
from rkgk.domain.models.manifest import INDEX_SCHEMA_VERSION, IndexBuildRun, IndexManifest
from rkgk.domain.models.vocabulary import ConceptType, PaperConceptRelation

CHUNKS = (Chunk.create(paper_id=1, idx=0, page_start=1, page_end=1, text="We study retrieval."),)

GRAPH = KnowledgeGraph(
    paper_ids=(1,),
    concepts=(Concept(id="retrieval", canonical_name="Retrieval", type=ConceptType.METHOD, paper_count=1),),
    paper_concepts=(
        PaperConceptEdge(
            paper_id=1,
            concept_id="retrieval",
            relation=PaperConceptRelation.USES,
            evidence=(ChunkEvidence(page=1, quote="We study retrieval.", chunk_id="1:0"),),
        ),
    ),
    concept_relations=(),
)

TABLE = EmbeddingTable(
    items=(
        EmbeddedItem(kind=EmbeddedItemKind.CHUNK, ref="1:0", paper_id=1, text="We study retrieval."),
        EmbeddedItem(kind=EmbeddedItemKind.SUMMARY, ref="1", paper_id=1, text="検索を研究する。"),
        EmbeddedItem(kind=EmbeddedItemKind.CONCEPT, ref="retrieval", text="Retrieval"),
    ),
    vectors=np.zeros((3, 4), dtype=np.float32),
)


CHUNK_ITEM, SUMMARY_ITEM, CONCEPT_ITEM = TABLE.items


def build_table(*items: EmbeddedItem) -> EmbeddingTable:
    """Pair the given items with one zero vector each, in row order."""
    return EmbeddingTable(items=items, vectors=np.zeros((len(items), 4), dtype=np.float32))


def build_manifest(**overrides: object) -> IndexManifest:
    payload: dict[str, object] = {
        "schema_version": 1,
        "domain_model_version": 1,
        "embedding_model": "fake-4",
        "embedding_dimension": 4,
        "chunk_max_tokens": 512,
        "paper_ids": (1,),
    }
    payload.update(overrides)
    return IndexManifest.model_validate(payload)


def build_run(**overrides: object) -> IndexBuildRun:
    payload: dict[str, object] = {
        "manifest": build_manifest(),
        "chunks": CHUNKS,
        "embeddings": TABLE,
        "graph": GRAPH,
    }
    payload.update(overrides)
    return IndexBuildRun.model_validate(payload)


def test_a_manifest_declares_the_current_schema_version() -> None:
    assert build_manifest().schema_version == INDEX_SCHEMA_VERSION


def test_another_schema_version_is_rejected() -> None:
    with pytest.raises(ValidationError):
        build_manifest(schema_version=2)


def test_the_same_paper_listed_twice_is_rejected() -> None:
    with pytest.raises(ValidationError, match="declares 1 more than once"):
        build_manifest(paper_ids=(1, 2, 1))


def test_a_manifest_without_papers_is_rejected() -> None:
    with pytest.raises(ValidationError):
        build_manifest(paper_ids=())


def test_a_manifest_keeps_the_model_and_the_budget_of_the_build() -> None:
    manifest = build_manifest(embedding_model="Qwen/Qwen3-Embedding-0.6B", embedding_dimension=1024)
    assert manifest.embedding_model == "Qwen/Qwen3-Embedding-0.6B"
    assert manifest.embedding_dimension == 1024
    assert manifest.chunk_max_tokens == 512


def test_a_run_holds_the_manifest_and_the_artifacts_it_describes() -> None:
    run = build_run()
    assert run.manifest.paper_ids == run.graph.paper_ids
    assert run.chunks == CHUNKS
    assert run.embeddings == TABLE


def test_a_manifest_that_covers_other_papers_than_the_graph_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"manifest covers papers \[1, 2\]"):
        build_run(manifest=build_manifest(paper_ids=(1, 2)))


def test_a_manifest_that_declares_another_dimension_than_the_vectors_is_rejected() -> None:
    with pytest.raises(ValidationError, match="manifest declares dimension 8"):
        build_run(manifest=build_manifest(embedding_dimension=8))


def test_a_chunk_of_a_paper_the_manifest_does_not_cover_is_rejected() -> None:
    stray = Chunk.create(paper_id=2, idx=0, page_start=1, page_end=1, text="We rank documents.")
    with pytest.raises(ValidationError, match="chunks hold papers the manifest does not cover: '2'"):
        build_run(chunks=CHUNKS + (stray,))


def test_a_chunk_item_that_names_no_chunk_is_rejected() -> None:
    table = build_table(CHUNK_ITEM.model_copy(update={"ref": "1:1"}), SUMMARY_ITEM, CONCEPT_ITEM)
    with pytest.raises(ValidationError, match="chunk items point at chunks the index does not hold: '1:1'"):
        build_run(embeddings=table)


def test_a_chunk_without_an_item_is_rejected() -> None:
    later = Chunk.create(paper_id=1, idx=1, page_start=2, page_end=2, text="We rank documents.")
    with pytest.raises(ValidationError, match="chunks without a chunk item: '1:1'"):
        build_run(chunks=CHUNKS + (later,))


def test_a_chunk_item_whose_text_differs_from_its_chunk_is_rejected() -> None:
    table = build_table(CHUNK_ITEM.model_copy(update={"text": "We rank documents."}), SUMMARY_ITEM, CONCEPT_ITEM)
    with pytest.raises(ValidationError, match="embedded from another text than their chunk: '1:0'"):
        build_run(embeddings=table)


def test_a_summary_item_of_a_paper_outside_the_manifest_is_rejected() -> None:
    table = build_table(CHUNK_ITEM, SUMMARY_ITEM.model_copy(update={"ref": "2", "paper_id": 2}), CONCEPT_ITEM)
    with pytest.raises(ValidationError, match="summary items point at papers the index does not hold: '2'"):
        build_run(embeddings=table)


def test_a_paper_without_a_summary_item_is_rejected() -> None:
    table = build_table(CHUNK_ITEM, CONCEPT_ITEM)
    with pytest.raises(ValidationError, match="papers without a summary item: '1'"):
        build_run(embeddings=table)


def test_a_concept_item_that_names_no_concept_of_the_graph_is_rejected() -> None:
    table = build_table(CHUNK_ITEM, SUMMARY_ITEM, CONCEPT_ITEM.model_copy(update={"ref": "ranking"}))
    with pytest.raises(ValidationError, match="concept items point at concepts the index does not hold: 'ranking'"):
        build_run(embeddings=table)


def test_a_concept_without_an_item_is_rejected() -> None:
    table = build_table(CHUNK_ITEM, SUMMARY_ITEM)
    with pytest.raises(ValidationError, match="concepts without a concept item: 'retrieval'"):
        build_run(embeddings=table)


def test_a_run_whose_chunks_items_papers_and_concepts_all_match_is_accepted() -> None:
    chunks = CHUNKS + (Chunk.create(paper_id=2, idx=0, page_start=1, page_end=1, text="We rank documents."),)
    graph = KnowledgeGraph(
        paper_ids=(1, 2),
        concepts=GRAPH.concepts
        + (Concept(id="ranking", canonical_name="Ranking", type=ConceptType.METHOD, paper_count=1),),
        paper_concepts=GRAPH.paper_concepts,
        concept_relations=(),
    )
    table = build_table(
        CHUNK_ITEM,
        EmbeddedItem(kind=EmbeddedItemKind.CHUNK, ref="2:0", paper_id=2, text="We rank documents."),
        SUMMARY_ITEM,
        EmbeddedItem(kind=EmbeddedItemKind.SUMMARY, ref="2", paper_id=2, text="順位付けを研究する。"),
        CONCEPT_ITEM,
        EmbeddedItem(kind=EmbeddedItemKind.CONCEPT, ref="ranking", text="Ranking"),
    )
    run = build_run(manifest=build_manifest(paper_ids=(1, 2)), chunks=chunks, embeddings=table, graph=graph)
    assert run.chunks == chunks
    assert run.embeddings == table
    assert run.graph == graph
