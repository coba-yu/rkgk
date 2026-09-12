import copy
import json
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from rkgk.cli import build
from rkgk.cli.build import main
from rkgk.infrastructure.fake_embedder import FakeEmbedder
from tests.cli.test_extract import VALID_EXTRACTION

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"

# The fixture index lists two papers but ships the artifacts of the first one only, so the second is written here.
PAPER_TWO_META: dict[str, Any] = {
    "id": 2,
    "title": "Knowledge Graphs as Retrieval Backbones",
    "authors": ["Grace Hopper"],
    "year": 2026,
    "venue": "NeurIPS",
    "page_count": 1,
    "preprocess": {"tool": "pymupdf", "version": "1.24.0", "processed_at": "2026-01-01T00:00:00Z"},
}

PAPER_TWO_PAGE = """# Knowledge Graphs as Retrieval Backbones

## Abstract

We use a knowledge graph as the retrieval backbone of a paper search system.
Each concept node links the papers that mention it.
"""

EXTRACTION_TWO: dict[str, Any] = {
    "schema_version": 1,
    "paper_id": 2,
    "summary_ja": "この論文は知識グラフを検索の骨格として使う。",
    "concepts": [{"local_id": "c1", "name": "Knowledge Graph", "type": "method"}],
    "paper_concepts": [
        {
            "concept_id": "c1",
            "relation": "uses",
            "evidence": [{"page": 1, "quote": "a knowledge graph as the retrieval backbone"}],
        }
    ],
}

NORMALIZED_CONCEPTS: list[dict[str, Any]] = [
    {
        "id": "retrieval-augmented-generation",
        "canonical_name": "Retrieval-Augmented Generation",
        "type": "method",
        "aliases": ["RAG"],
        "description": "",
        "merged_from": [{"paper_id": 1, "local_id": "c1"}],
    },
    {
        "id": "page-aligned-chunking",
        "canonical_name": "Page-Aligned Chunking",
        "type": "method",
        "aliases": [],
        "description": "",
        "merged_from": [{"paper_id": 1, "local_id": "c2"}],
    },
    {
        "id": "knowledge-graph",
        "canonical_name": "Knowledge Graph",
        "type": "method",
        "aliases": [],
        "description": "",
        "merged_from": [{"paper_id": 2, "local_id": "c1"}],
    },
]

NORMALIZED_RELATIONS: list[dict[str, Any]] = [
    {
        "source_id": "knowledge-graph",
        "target_id": "retrieval-augmented-generation",
        "relation": "used_for",
        "rationale": "A knowledge graph supplies the structure the pipeline walks.",
    }
]


def read_output(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


def copy_fixture(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    shutil.copytree(FIXTURE_DIR, data_dir)
    write_second_paper(data_dir)
    return data_dir


def write_second_paper(data_dir: Path) -> None:
    paper_dir = data_dir / "papers" / "0002"
    (paper_dir / "pages").mkdir(parents=True, exist_ok=True)
    (paper_dir / "paper.json").write_text(json.dumps(PAPER_TWO_META, ensure_ascii=False), encoding="utf-8")
    (paper_dir / "pages" / "001.md").write_text(PAPER_TWO_PAGE, encoding="utf-8")


def write_extractions(data_dir: Path, first: dict[str, Any] = VALID_EXTRACTION) -> None:
    for paper_id, extraction in ((1, first), (2, EXTRACTION_TWO)):
        path = data_dir / "papers" / f"{paper_id:04d}" / "extraction.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(extraction, ensure_ascii=False), encoding="utf-8")


def write_normalization(data_dir: Path) -> None:
    directory = data_dir / "normalization"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "concepts.json").write_text(
        json.dumps({"schema_version": 1, "concepts": NORMALIZED_CONCEPTS}, ensure_ascii=False), encoding="utf-8"
    )
    (directory / "concept_relations.json").write_text(
        json.dumps({"schema_version": 1, "concept_relations": NORMALIZED_RELATIONS}, ensure_ascii=False),
        encoding="utf-8",
    )


def build_broken_extraction() -> dict[str, Any]:
    payload = copy.deepcopy(VALID_EXTRACTION)
    payload["paper_concepts"][0]["evidence"][0]["quote"] = "a sentence the paper never wrote"
    return payload


def test_run_builds_the_index_and_writes_every_artifact(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data_dir = copy_fixture(tmp_path)
    write_extractions(data_dir)
    write_normalization(data_dir)
    assert main(["--data-dir", str(data_dir), "--embedder", "fake"]) == 0
    output = read_output(capsys)
    assert output["status"] == "ok"
    assert output["papers"] == 2
    assert output["chunks"] == 4
    assert output["embedded_items"] == 9
    assert output["embedding_model"] == "fake-8"
    assert output["embedding_dimension"] == 8
    assert output["concepts"] == 3
    assert output["paper_concepts"] == 3
    assert output["concept_relations"] == 2
    index_dir = data_dir / "index"
    assert output["paths"] == [
        str(index_dir / "manifest.json"),
        str(index_dir / "chunks.jsonl"),
        str(index_dir / "items.jsonl"),
        str(index_dir / "embeddings.npy"),
        str(index_dir / "graph.json"),
    ]
    assert all(Path(path).exists() for path in output["paths"])


def test_run_records_the_settings_of_the_build_in_the_manifest(tmp_path: Path) -> None:
    data_dir = copy_fixture(tmp_path)
    write_extractions(data_dir)
    write_normalization(data_dir)
    assert main(["--data-dir", str(data_dir), "--embedder", "fake", "--max-tokens", "256"]) == 0
    manifest = json.loads((data_dir / "index" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "schema_version": 1,
        "domain_model_version": 1,
        "embedding_model": "fake-8",
        "embedding_dimension": 8,
        "chunk_max_tokens": 256,
        "paper_ids": [1, 2],
    }


def test_run_reports_every_paper_that_has_not_been_extracted_yet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = copy_fixture(tmp_path)
    write_normalization(data_dir)
    assert main(["--data-dir", str(data_dir), "--embedder", "fake"]) == 2
    output = read_output(capsys)
    assert output["status"] == "error"
    assert "1, 2" in output["message"]
    assert not (data_dir / "index").exists()


def test_run_reports_an_extraction_whose_quote_is_no_longer_in_the_paper(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = copy_fixture(tmp_path)
    write_extractions(data_dir, first=build_broken_extraction())
    write_normalization(data_dir)
    assert main(["--data-dir", str(data_dir), "--embedder", "fake"]) == 1
    output = read_output(capsys)
    assert output["status"] == "invalid"
    assert output["reason"] == "extraction_mismatch"
    assert output["papers"] == [
        {
            "paper_id": 1,
            "issues": [
                {
                    "path": "paper_concepts[0].evidence[0].quote",
                    "message": "is not found in the text of page 1",
                }
            ],
        }
    ]
    assert not (data_dir / "index").exists()


def test_run_reports_a_data_directory_whose_concepts_are_not_normalized_yet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = copy_fixture(tmp_path)
    write_extractions(data_dir)
    assert main(["--data-dir", str(data_dir), "--embedder", "fake"]) == 2
    output = read_output(capsys)
    assert output["status"] == "error"
    assert "concepts.json" in output["message"]


def test_run_reports_quotes_that_no_single_chunk_holds(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data_dir = copy_fixture(tmp_path)
    write_extractions(data_dir)
    write_normalization(data_dir)
    assert main(["--data-dir", str(data_dir), "--embedder", "fake", "--max-tokens", "2"]) == 1
    output = read_output(capsys)
    assert output["reason"] == "unresolved_evidence"
    assert {item["paper_id"] for item in output["unresolved"]} == {1, 2}


def test_run_passes_the_chosen_model_to_the_qwen3_embedder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = copy_fixture(tmp_path)
    write_extractions(data_dir)
    write_normalization(data_dir)
    seen: list[str] = []

    def _build(model_name: str) -> FakeEmbedder:
        seen.append(model_name)
        return FakeEmbedder()

    monkeypatch.setattr(build, "Qwen3Embedder", _build)
    assert main(["--data-dir", str(data_dir), "--embedding-model", "Qwen/Qwen3-Embedding-4B"]) == 0
    assert seen == ["Qwen/Qwen3-Embedding-4B"]


def test_version_is_printed(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert version("rkgk") in capsys.readouterr().out


def test_help_mentions_the_embedder_to_choose(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    output = capsys.readouterr().out
    assert "--embedder" in output
    assert "--data-dir" in output


@pytest.mark.parametrize("max_tokens", ["0", "-1"])
def test_fewer_than_one_token_per_chunk_is_a_usage_error(
    max_tokens: str, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--data-dir", str(tmp_path), "--max-tokens", max_tokens])
    assert caught.value.code == 2
    assert "must be at least 1" in capsys.readouterr().err
