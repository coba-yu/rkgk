import json
from pathlib import Path

import numpy as np
import pytest

from rkgk.domain import DOMAIN_MODEL_VERSION
from rkgk.domain.models.chunk import Chunk
from rkgk.domain.models.embedding import EmbeddedItem, EmbeddedItemKind, EmbeddingTable
from rkgk.domain.models.graph import ChunkEvidence, Concept, ConceptEdge, KnowledgeGraph, PaperConceptEdge
from rkgk.domain.models.manifest import IndexManifest
from rkgk.domain.models.vocabulary import ConceptRelationType, ConceptType, Origin, PaperConceptRelation
from rkgk.domain.repositories.index import (
    IndexArtifactInvalidError,
    IndexIncompatibleError,
    IndexNotFoundError,
    IndexRepositoryError,
)
from rkgk.infrastructure.file_index_repository import FileIndexRepository

TABLE = EmbeddingTable(
    items=(
        EmbeddedItem(kind=EmbeddedItemKind.CHUNK, ref="1:0", paper_id=1, text="We study retrieval."),
        EmbeddedItem(kind=EmbeddedItemKind.SUMMARY, ref="1", paper_id=1, text="検索を研究する。"),
        EmbeddedItem(kind=EmbeddedItemKind.CONCEPT, ref="retrieval", text="Retrieval (検索)"),
    ),
    vectors=np.arange(12, dtype=np.float32).reshape(3, 4),
)

CHUNKS = (
    Chunk.create(paper_id=1, idx=0, page_start=1, page_end=1, text="We study retrieval."),
    Chunk.create(paper_id=1, idx=1, page_start=2, page_end=2, text="検索の手法を述べる。"),
)

MANIFEST = IndexManifest(
    schema_version=1,
    domain_model_version=1,
    embedding_model="Qwen/Qwen3-Embedding-0.6B",
    embedding_dimension=4,
    chunk_max_tokens=512,
    paper_ids=(1, 2),
)

GRAPH = KnowledgeGraph(
    paper_ids=(1, 2),
    concepts=(
        Concept(
            id="retrieval",
            canonical_name="Retrieval",
            type=ConceptType.METHOD,
            aliases=("検索",),
            description="Retrieving documents relevant to a query.",
            paper_count=2,
        ),
        Concept(
            id="knowledge-graph",
            canonical_name="Knowledge Graph",
            type=ConceptType.PROBLEM,
            description="論文の概念同士のつながりを表すグラフ。",
            paper_count=1,
        ),
    ),
    paper_concepts=(
        PaperConceptEdge(
            paper_id=1,
            concept_id="retrieval",
            relation=PaperConceptRelation.USES,
            evidence=(ChunkEvidence(page=1, quote="We use retrieval.", chunk_id="1:0"),),
        ),
    ),
    concept_relations=(
        ConceptEdge(
            source_id="retrieval",
            target_id="knowledge-graph",
            relation=ConceptRelationType.USED_FOR,
            origin=Origin.PAPER,
            paper_id=1,
            evidence=(ChunkEvidence(page=1, quote="Retrieval draws on the knowledge graph.", chunk_id="1:0"),),
        ),
        ConceptEdge(
            source_id="knowledge-graph",
            target_id="retrieval",
            relation=ConceptRelationType.RELATED_TO,
            origin=Origin.GENERAL_KNOWLEDGE,
            rationale="知識グラフと検索は一般的によく組み合わせて使われる。",
        ),
    ),
)


# One row per chunk of CHUNKS, per paper of MANIFEST and per concept of GRAPH, the items IndexBuildRun expects.
CONSISTENT_TABLE = EmbeddingTable(
    items=(
        EmbeddedItem(kind=EmbeddedItemKind.CHUNK, ref="1:0", paper_id=1, text="We study retrieval."),
        EmbeddedItem(kind=EmbeddedItemKind.CHUNK, ref="1:1", paper_id=1, text="検索の手法を述べる。"),
        EmbeddedItem(kind=EmbeddedItemKind.SUMMARY, ref="1", paper_id=1, text="検索を研究する。"),
        EmbeddedItem(kind=EmbeddedItemKind.SUMMARY, ref="2", paper_id=2, text="知識グラフを研究する。"),
        EmbeddedItem(kind=EmbeddedItemKind.CONCEPT, ref="retrieval", text="Retrieval (検索)"),
        EmbeddedItem(kind=EmbeddedItemKind.CONCEPT, ref="knowledge-graph", text="Knowledge Graph"),
    ),
    vectors=np.arange(24, dtype=np.float32).reshape(6, 4),
)


def test_a_saved_table_is_read_back_unchanged(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_embeddings(TABLE)
    assert repository.find_embeddings() == TABLE


def test_the_vectors_keep_their_dtype_and_shape(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_embeddings(TABLE)
    vectors = repository.find_embeddings().vectors
    assert vectors.dtype == np.float32
    assert vectors.shape == (3, 4)


def test_the_items_are_written_one_per_line_in_row_order(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_embeddings(TABLE)
    assert repository.items_path == tmp_path / "index" / "items.jsonl"
    assert repository.embeddings_path == tmp_path / "index" / "embeddings.npy"
    lines = repository.items_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["ref"] for line in lines] == ["1:0", "1", "retrieval"]
    assert json.loads(lines[2]) == {
        "kind": "concept",
        "ref": "retrieval",
        "paper_id": None,
        "text": "Retrieval (検索)",
    }


def test_the_items_are_stored_as_japanese_characters(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_embeddings(TABLE)
    assert "検索を研究する。" in repository.items_path.read_text(encoding="utf-8")


def test_saving_twice_replaces_the_previous_table(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_embeddings(TABLE)
    smaller = EmbeddingTable(items=TABLE.items[:1], vectors=TABLE.vectors[:1])
    repository.save_embeddings(smaller)
    assert repository.find_embeddings() == smaller


def test_an_empty_table_survives_the_round_trip(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    empty = EmbeddingTable(items=(), vectors=np.empty((0, 4), dtype=np.float32))
    repository.save_embeddings(empty)
    assert repository.find_embeddings() == empty


def test_a_data_directory_without_an_index_is_reported_as_not_found(tmp_path: Path) -> None:
    with pytest.raises(IndexNotFoundError, match="items.jsonl: not found"):
        FileIndexRepository(tmp_path).find_embeddings()


def test_an_index_whose_vectors_are_gone_is_reported_as_not_found(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_embeddings(TABLE)
    repository.embeddings_path.unlink()
    with pytest.raises(IndexNotFoundError, match="embeddings.npy: not found"):
        repository.find_embeddings()


def test_a_corrupted_item_line_is_reported_with_its_line_number(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_embeddings(TABLE)
    lines = repository.items_path.read_text(encoding="utf-8").splitlines()
    lines[1] = '{"kind": "summary", "ref": "1"}'
    repository.items_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(IndexArtifactInvalidError, match="line 2 is not a valid EmbeddedItem"):
        repository.find_embeddings()


def test_a_non_utf8_items_file_is_reported_as_invalid(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_embeddings(TABLE)
    repository.items_path.write_bytes(b"\xff\xfe")
    with pytest.raises(IndexArtifactInvalidError, match="is not valid UTF-8"):
        repository.find_embeddings()


def test_a_vectors_file_that_is_not_npy_is_reported_as_invalid(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_embeddings(TABLE)
    repository.embeddings_path.write_bytes(b"not an npy file")
    with pytest.raises(IndexArtifactInvalidError, match="is not a valid npy file"):
        repository.find_embeddings()


def test_items_and_vectors_of_different_lengths_are_reported_as_invalid(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_embeddings(TABLE)
    with repository.embeddings_path.open("wb") as stream:
        np.save(stream, TABLE.vectors[:2])
    with pytest.raises(IndexArtifactInvalidError, match="vectors has 2 rows for 3 items"):
        repository.find_embeddings()


def test_errors_carry_the_location_as_an_attribute(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    with pytest.raises(IndexRepositoryError) as caught:
        repository.find_embeddings()
    assert caught.value.location == str(repository.items_path)


def test_a_saved_graph_is_read_back_unchanged(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_graph(GRAPH)
    assert repository.find_graph() == GRAPH


def test_the_graph_path_is_under_the_index_directory(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    assert repository.graph_path == tmp_path / "index" / "graph.json"


def test_the_graph_is_stored_as_japanese_characters_ending_with_a_newline(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_graph(GRAPH)
    text = repository.graph_path.read_text(encoding="utf-8")
    assert "知識グラフと検索は一般的によく組み合わせて使われる。" in text
    assert text.endswith("\n")


def test_a_data_directory_without_a_graph_is_reported_as_not_found(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    with pytest.raises(IndexNotFoundError, match="graph.json: not found") as caught:
        repository.find_graph()
    assert caught.value.location == str(repository.graph_path)


def test_non_json_graph_content_is_reported_as_invalid(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_graph(GRAPH)
    repository.graph_path.write_text("not json", encoding="utf-8")
    with pytest.raises(IndexArtifactInvalidError, match="is not a valid KnowledgeGraph"):
        repository.find_graph()


def test_a_graph_with_an_edge_to_an_undeclared_concept_is_reported_as_invalid(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    broken = GRAPH.model_dump(mode="json")
    broken["concepts"] = broken["concepts"][:1]
    repository.graph_path.parent.mkdir(parents=True, exist_ok=True)
    repository.graph_path.write_text(json.dumps(broken, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(IndexArtifactInvalidError, match="KnowledgeGraph"):
        repository.find_graph()


def test_saving_the_graph_twice_replaces_the_previous_graph(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_graph(GRAPH)
    smaller = KnowledgeGraph(
        paper_ids=GRAPH.paper_ids,
        concepts=GRAPH.concepts[:1],
        paper_concepts=GRAPH.paper_concepts,
        concept_relations=(),
    )
    repository.save_graph(smaller)
    assert repository.find_graph() == smaller


def test_the_graph_and_the_embedding_table_live_side_by_side(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_embeddings(TABLE)
    repository.save_graph(GRAPH)
    assert repository.find_embeddings() == TABLE
    assert repository.find_graph() == GRAPH


def test_saved_chunks_are_read_back_unchanged(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_chunks(CHUNKS)
    assert repository.find_chunks() == CHUNKS


def test_the_chunks_are_written_one_per_line_as_japanese_characters(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_chunks(CHUNKS)
    assert repository.chunks_path == tmp_path / "index" / "chunks.jsonl"
    lines = repository.chunks_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["1:0", "1:1"]
    assert "検索の手法を述べる。" in lines[1]


def test_a_data_directory_without_chunks_is_reported_as_not_found(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    with pytest.raises(IndexNotFoundError, match="chunks.jsonl: not found") as caught:
        repository.find_chunks()
    assert caught.value.location == str(repository.chunks_path)


def test_a_corrupted_chunk_line_is_reported_with_its_line_number(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_chunks(CHUNKS)
    lines = repository.chunks_path.read_text(encoding="utf-8").splitlines()
    lines[1] = '{"id": "1:1", "paper_id": 1}'
    repository.chunks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(IndexArtifactInvalidError, match="line 2 is not a valid Chunk"):
        repository.find_chunks()


def test_the_same_chunk_id_on_two_lines_is_reported_as_invalid(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_chunks((CHUNKS[0], CHUNKS[0]))
    with pytest.raises(IndexArtifactInvalidError, match="line 2 repeats the chunk '1:0'"):
        repository.find_chunks()


def test_saving_chunks_twice_replaces_the_previous_chunks(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_chunks(CHUNKS)
    repository.save_chunks(CHUNKS[:1])
    assert repository.find_chunks() == CHUNKS[:1]


def test_a_saved_manifest_is_read_back_unchanged(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_manifest(MANIFEST)
    assert repository.find_manifest() == MANIFEST


def test_the_manifest_is_stored_as_indented_json_ending_with_a_newline(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_manifest(MANIFEST)
    assert repository.manifest_path == tmp_path / "index" / "manifest.json"
    text = repository.manifest_path.read_text(encoding="utf-8")
    assert '\n  "embedding_model": "Qwen/Qwen3-Embedding-0.6B"' in text
    assert text.endswith("\n")


def test_a_data_directory_without_a_manifest_is_reported_as_not_found(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    with pytest.raises(IndexNotFoundError, match="manifest.json: not found") as caught:
        repository.find_manifest()
    assert caught.value.location == str(repository.manifest_path)


def store_manifest_with(repository: FileIndexRepository, **overrides: object) -> None:
    """Write a manifest whose stored JSON differs from MANIFEST, so a version this code rejects can be tested."""
    repository.save_manifest(MANIFEST)
    stored = json.loads(repository.manifest_path.read_text(encoding="utf-8"))
    stored.update(overrides)
    repository.manifest_path.write_text(json.dumps(stored), encoding="utf-8")


def test_a_manifest_of_another_schema_version_is_reported_as_incompatible(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    store_manifest_with(repository, schema_version=2)
    with pytest.raises(IndexIncompatibleError) as caught:
        repository.find_manifest()
    message = str(caught.value)
    assert "schema" in message
    assert "2" in message
    assert "1" in message


def test_a_manifest_of_another_domain_model_version_is_reported_as_incompatible(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    store_manifest_with(repository, domain_model_version=DOMAIN_MODEL_VERSION + 1)
    with pytest.raises(IndexIncompatibleError, match="domain model"):
        repository.find_manifest()


def test_an_incompatible_index_is_a_repository_error_carrying_its_location(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    store_manifest_with(repository, schema_version=2)
    with pytest.raises(IndexRepositoryError) as caught:
        repository.find_manifest()
    assert isinstance(caught.value, IndexIncompatibleError)
    assert caught.value.location == str(repository.manifest_path)


def test_a_manifest_that_is_not_json_is_reported_as_invalid(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    repository.save_manifest(MANIFEST)
    repository.manifest_path.write_text("not json", encoding="utf-8")
    with pytest.raises(IndexArtifactInvalidError, match="is not valid JSON"):
        repository.find_manifest()


def save_whole_index(repository: FileIndexRepository) -> None:
    repository.save_chunks(CHUNKS)
    repository.save_embeddings(CONSISTENT_TABLE)
    repository.save_graph(GRAPH)
    repository.save_manifest(MANIFEST)


def test_the_whole_index_is_read_back_as_the_run_that_built_it(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    save_whole_index(repository)
    run = repository.find_index()
    assert run.manifest == MANIFEST
    assert run.chunks == CHUNKS
    assert run.embeddings == CONSISTENT_TABLE
    assert run.graph == GRAPH


def test_reading_the_index_of_an_empty_data_directory_is_reported_as_not_found(tmp_path: Path) -> None:
    with pytest.raises(IndexNotFoundError, match="manifest.json: not found"):
        FileIndexRepository(tmp_path).find_index()


def test_an_index_whose_build_did_not_write_the_manifest_is_reported_as_not_found(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    save_whole_index(repository)
    repository.manifest_path.unlink()
    with pytest.raises(IndexNotFoundError, match="manifest.json: not found"):
        repository.find_index()


def test_a_manifest_disagreeing_with_the_saved_vectors_is_reported_as_invalid(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    save_whole_index(repository)
    repository.save_manifest(MANIFEST.model_copy(update={"embedding_dimension": 8}))
    with pytest.raises(IndexArtifactInvalidError, match="do not belong together") as caught:
        repository.find_index()
    assert caught.value.location == str(tmp_path / "index")


def test_chunks_rewritten_after_the_saved_items_are_reported_as_invalid(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    save_whole_index(repository)
    rebuilt = (CHUNKS[0], Chunk.create(paper_id=1, idx=2, page_start=2, page_end=2, text="順位付けの手法を述べる。"))
    repository.save_chunks(rebuilt)
    with pytest.raises(IndexArtifactInvalidError, match="do not belong together") as caught:
        repository.find_index()
    assert "1:1" in str(caught.value)


def test_an_incompatible_manifest_is_reported_before_the_missing_artifacts(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    store_manifest_with(repository, schema_version=2)
    with pytest.raises(IndexIncompatibleError, match="schema"):
        repository.find_index()


def test_the_paths_of_an_index_start_with_the_manifest(tmp_path: Path) -> None:
    repository = FileIndexRepository(tmp_path)
    assert repository.paths() == (
        tmp_path / "index" / "manifest.json",
        tmp_path / "index" / "chunks.jsonl",
        tmp_path / "index" / "items.jsonl",
        tmp_path / "index" / "embeddings.npy",
        tmp_path / "index" / "graph.json",
    )
