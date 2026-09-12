"""Walks the whole pipeline end to end over the hand-written fixture under tests/fixtures/pipeline.

The fixture mirrors a data directory after extract and normalize have run: three papers with their pages, one
extraction per paper, and one normalization over all of them.
The extract and normalize commands themselves are not run, because they call Claude; the half of each command
that does not need an agent, the validation, is run on the hand-written artifacts instead, so this test still
proves that what those commands would write is accepted by what comes after them.
The FakeEmbedder stands in for the real model, so what is checked here is the wiring of build and search and the
graph the traversal walks, never the quality of retrieval.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from rkgk.cli import build, search
from rkgk.domain.services.concept_normalization import check_normalization_against_extractions
from rkgk.infrastructure.file_concept_normalization_repository import FileConceptNormalizationRepository
from rkgk.infrastructure.file_paper_extraction_repository import FilePaperExtractionRepository
from rkgk.infrastructure.file_paper_repository import FilePaperRepository
from rkgk.usecase.validate_paper_extraction import ValidatePaperExtractionUseCase

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "pipeline"

PAPER_IDS = (1, 2, 3)

# The FakeEmbedder gives one vector to one text, so a query that repeats the summary of paper 1 verbatim scores
# 1.0 on the summary item of that paper and nothing else can beat it, which pins the only direct candidate.
QUERY = "密検索で見つけた論文から知識グラフをたどり、次に読む論文を推薦するグラフ誘導検索を提案する。"


def read_output(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


def copy_fixture(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    shutil.copytree(FIXTURE_DIR, data_dir)
    return data_dir


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_index(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    """Leave a data directory with the index the build command wrote, and read away what the build printed."""
    data_dir = copy_fixture(tmp_path)
    assert build.main(["--data-dir", str(data_dir), "--embedder", "fake"]) == 0
    output = read_output(capsys)
    assert output["status"] == "ok"
    assert all(Path(path).exists() for path in output["paths"])
    return data_dir


def find_candidate(candidates: list[dict[str, Any]], paper_id: int) -> dict[str, Any]:
    return next(candidate for candidate in candidates if candidate["paper_id"] == paper_id)


def collect_concept_ids(path: dict[str, Any]) -> set[str]:
    """Collect every concept a path stands on, so a concept traversal must not have followed can be looked for."""
    ids = {path["source_edge"]["concept_id"], path["target_edge"]["concept_id"]}
    for hop in path["hops"]:
        ids |= {hop["edge"]["source_id"], hop["edge"]["target_id"], hop["reached_concept_id"]}
    return ids


def test_every_hand_written_extraction_passes_the_validation_the_extract_command_applies(tmp_path: Path) -> None:
    data_dir = copy_fixture(tmp_path)
    use_case = ValidatePaperExtractionUseCase(FilePaperRepository(data_dir))
    for paper_id in PAPER_IDS:
        payload = read_json(data_dir / "papers" / f"{paper_id:04d}" / "extraction.json")
        extraction = use_case.execute(paper_id, payload)
        assert extraction.paper_id == paper_id


def test_the_hand_written_normalization_covers_every_extracted_concept(tmp_path: Path) -> None:
    data_dir = copy_fixture(tmp_path)
    normalization = FileConceptNormalizationRepository(data_dir).find()
    extraction_repository = FilePaperExtractionRepository(data_dir)
    extractions = tuple(extraction_repository.find(paper_id) for paper_id in PAPER_IDS)
    assert check_normalization_against_extractions(normalization, extractions) == ()


def test_build_and_search_return_the_graph_candidates_with_their_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = build_index(tmp_path, capsys)
    assert search.main(["--data-dir", str(data_dir), "--embedder", "fake", "--top-k", "1", QUERY]) == 0
    output = read_output(capsys)
    assert output["status"] == "ok"

    direct_candidates = output["direct_candidates"]
    assert [candidate["paper_id"] for candidate in direct_candidates] == [1]
    first_hit = direct_candidates[0]["hits"][0]
    assert first_hit["kind"] == "summary"
    assert first_hit["ref"] == "1"

    graph_candidates = output["graph_candidates"]
    # Both papers are reached by exactly one path, so their order is not what this test is about.
    assert {candidate["paper_id"] for candidate in graph_candidates} == {2, 3}

    to_knowledge_graph = find_candidate(graph_candidates, 2)["paths"]
    assert len(to_knowledge_graph) == 1
    path = to_knowledge_graph[0]
    assert path["source_edge"]["paper_id"] == 1
    assert path["source_edge"]["concept_id"] == "graph-guided-retrieval"
    assert path["source_edge"]["relation"] == "proposes"
    assert len(path["hops"]) == 1
    assert path["hops"][0]["edge"]["origin"] == "general_knowledge"
    assert path["hops"][0]["edge"]["source_id"] == "knowledge-graph"
    assert path["hops"][0]["edge"]["relation"] == "used_for"
    assert path["hops"][0]["reached_concept_id"] == "knowledge-graph"
    assert path["target_edge"]["paper_id"] == 2
    assert path["target_edge"]["relation"] == "proposes"
    assert path["target_edge"]["evidence"][0]["quote"] == "We propose a knowledge graph as the retrieval backbone"

    to_dense_retrieval = find_candidate(graph_candidates, 3)["paths"]
    assert len(to_dense_retrieval) == 1
    path = to_dense_retrieval[0]
    assert path["source_edge"]["paper_id"] == 1
    assert path["source_edge"]["concept_id"] == "graph-guided-retrieval"
    assert path["source_edge"]["relation"] == "proposes"
    assert len(path["hops"]) == 1
    assert path["hops"][0]["edge"]["origin"] == "general_knowledge"
    assert path["hops"][0]["edge"]["source_id"] == "dense-passage-retrieval"
    assert path["hops"][0]["edge"]["relation"] == "used_for"
    assert path["hops"][0]["reached_concept_id"] == "dense-passage-retrieval"
    assert path["target_edge"]["paper_id"] == 3
    assert path["target_edge"]["relation"] == "proposes"
    assert path["target_edge"]["evidence"][0]["quote"] == (
        "We propose dense passage retrieval, which embeds questions and passages with a dual encoder."
    )

    # Papers 1 and 2 both hold the merged concept, which is 2 of 3 papers and so a document frequency of 0.67,
    # above the default generic_concept_threshold of 0.4; traversal must therefore never stand on it.
    for candidate in (*direct_candidates, *graph_candidates):
        for candidate_path in candidate["paths"]:
            assert "retrieval-augmented-generation" not in collect_concept_ids(candidate_path)


def test_raising_the_generic_concept_threshold_lets_the_merged_concept_link_the_papers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = build_index(tmp_path, capsys)
    arguments = ["--data-dir", str(data_dir), "--embedder", "fake", "--top-k", "1"]
    assert search.main([*arguments, "--generic-concept-threshold", "0.7", QUERY]) == 0
    output = read_output(capsys)
    assert output["status"] == "ok"

    graph_candidates = output["graph_candidates"]
    # Paper 2 is now reached three ways against one for paper 3, and the more reached paper is listed first.
    assert [candidate["paper_id"] for candidate in graph_candidates] == [2, 3]

    paths = find_candidate(graph_candidates, 2)["paths"]
    shared_concept = [
        path
        for path in paths
        if not path["hops"]
        and path["source_edge"]["concept_id"] == "retrieval-augmented-generation"
        and path["target_edge"]["concept_id"] == "retrieval-augmented-generation"
    ]
    assert len(shared_concept) == 1

    stated_by_paper_one = [
        path
        for path in paths
        if len(path["hops"]) == 1
        and path["hops"][0]["edge"]["origin"] == "paper"
        and path["hops"][0]["edge"]["relation"] == "is_a"
    ]
    assert len(stated_by_paper_one) == 1
    hop = stated_by_paper_one[0]["hops"][0]
    assert hop["edge"]["source_id"] == "graph-guided-retrieval"
    assert hop["reached_concept_id"] == "retrieval-augmented-generation"
