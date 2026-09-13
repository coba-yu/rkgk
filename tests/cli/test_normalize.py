import copy
import json
import shutil
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from rkgk.cli import normalize
from rkgk.cli.normalize import main
from rkgk.domain.agents import StructuredOutputAgentError
from rkgk.domain.embedders import EmbedderError

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"

EXTRACTIONS: dict[int, dict[str, Any]] = {
    1: {
        "schema_version": 1,
        "paper_id": 1,
        "summary_ja": "この論文は会議論文検索のための検索拡張生成パイプラインを提案する。",
        "concepts": [
            {"local_id": "c1", "name": "Retrieval-Augmented Generation", "type": "method", "aliases": ["RAG"]},
            {"local_id": "c2", "name": "Page-Aligned Chunking", "type": "method"},
        ],
        "paper_concepts": [],
    },
    2: {
        "schema_version": 1,
        "paper_id": 2,
        "summary_ja": "この論文は知識グラフを検索の骨格として使う。",
        "concepts": [
            {"local_id": "c1", "name": "RAG", "type": "method"},
            {"local_id": "c2", "name": "Knowledge Graph", "type": "method"},
        ],
        "paper_concepts": [],
    },
}

GROUPING: dict[str, Any] = {"groups": [[{"paper_id": 1, "local_id": "c1"}, {"paper_id": 2, "local_id": "c1"}]]}

VALID_MERGE: dict[str, Any] = {
    "concepts": [
        {
            "id": "retrieval-augmented-generation",
            "canonical_name": "Retrieval-Augmented Generation",
            "type": "method",
            "aliases": ["RAG"],
            "merged_from": [{"paper_id": 1, "local_id": "c1"}, {"paper_id": 2, "local_id": "c1"}],
        }
    ]
}

VALID_RELATIONS: dict[str, Any] = {
    "concept_relations": [
        {
            "source_id": "page-aligned-chunking",
            "target_id": "retrieval-augmented-generation",
            "relation": "part_of",
            "rationale": "Chunking is the indexing step of a retrieval-augmented generation pipeline.",
        }
    ]
}

UNKNOWN_RELATIONS: dict[str, Any] = {
    "concept_relations": [
        {
            "source_id": "dense-retrieval",
            "target_id": "retrieval-augmented-generation",
            "relation": "used_for",
            "rationale": "Dense retrieval finds the passages the pipeline generates from.",
        }
    ]
}


def read_output(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


def copy_fixture(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    shutil.copytree(FIXTURE_DIR, data_dir)
    return data_dir


def write_extractions(data_dir: Path) -> None:
    for paper_id, extraction in EXTRACTIONS.items():
        path = data_dir / "papers" / f"{paper_id:04d}" / "extraction.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(extraction, ensure_ascii=False), encoding="utf-8")


class FakeAgent:
    """Stands in for the Claude CLI so the command tests never spawn an agent."""

    def __init__(self, *payloads: object) -> None:
        self._payloads = list(payloads)

    def answer(self, prompt: str, schema: dict[str, object]) -> object:
        return self._payloads.pop(0)


def install_agent(monkeypatch: pytest.MonkeyPatch, *payloads: object) -> None:
    monkeypatch.setattr(normalize, "ClaudeCodeAgent", lambda model=None: FakeAgent(*payloads))


def run(data_dir: Path, *extra: str) -> int:
    # The fake embedder keeps the command out of sentence-transformers, which the tests do not install.
    return main(["--data-dir", str(data_dir), "--embedder", "fake", "--concurrency", "1", *extra])


def install_failing_agent(monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    class FailingAgent:
        def answer(self, prompt: str, schema: dict[str, object]) -> object:
            raise StructuredOutputAgentError(message)

    monkeypatch.setattr(normalize, "ClaudeCodeAgent", lambda model=None: FailingAgent())


def build_incomplete_merge() -> dict[str, Any]:
    """Cover only the concept of paper 1, leaving the other concept of the group in no `merged_from`."""
    payload = copy.deepcopy(VALID_MERGE)
    payload["concepts"][0]["merged_from"] = [{"paper_id": 1, "local_id": "c1"}]
    return payload


def build_unknown_grouping() -> dict[str, Any]:
    return {"groups": [[{"paper_id": 1, "local_id": "c1"}, {"paper_id": 2, "local_id": "c9"}]]}


def test_run_normalizes_every_extracted_paper_and_writes_both_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = copy_fixture(tmp_path)
    write_extractions(data_dir)
    install_agent(monkeypatch, GROUPING, VALID_MERGE, VALID_RELATIONS)
    assert run(data_dir) == 0
    output = read_output(capsys)
    assert output["status"] == "ok"
    assert output["papers"] == 2
    assert output["concepts"] == 3
    assert output["merged_concepts"] == 1
    assert output["concept_relations"] == 1
    assert output["groups"] == 1
    assert output["attempts"] == {"grouping": 1, "merge_calls": 1, "merge_rounds": 1, "relations": 1}
    assert output["paths"] == [
        str(data_dir / "normalization" / "concepts.json"),
        str(data_dir / "normalization" / "concept_relations.json"),
    ]
    written = json.loads((data_dir / "normalization" / "concepts.json").read_text(encoding="utf-8"))
    assert [concept["id"] for concept in written["concepts"]] == [
        "retrieval-augmented-generation",
        "page-aligned-chunking",
        "knowledge-graph",
    ]


def test_run_normalizes_a_concept_no_group_holds_without_asking_the_agent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = copy_fixture(tmp_path)
    write_extractions(data_dir)
    install_agent(monkeypatch, {"groups": []}, {"concept_relations": []})
    assert run(data_dir) == 0
    output = read_output(capsys)
    assert output["groups"] == 0
    assert output["concepts"] == 4
    assert output["attempts"]["merge_calls"] == 0


def test_run_counts_the_attempts_each_stage_needed_to_correct_itself(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = copy_fixture(tmp_path)
    write_extractions(data_dir)
    install_agent(
        monkeypatch,
        build_unknown_grouping(),
        GROUPING,
        build_incomplete_merge(),
        VALID_MERGE,
        UNKNOWN_RELATIONS,
        VALID_RELATIONS,
    )
    assert run(data_dir) == 0
    assert read_output(capsys)["attempts"] == {
        "grouping": 2,
        "merge_calls": 2,
        "merge_rounds": 1,
        "relations": 2,
    }


def test_run_reports_the_issues_of_the_merge_when_the_agent_keeps_failing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = copy_fixture(tmp_path)
    write_extractions(data_dir)
    rejected = build_incomplete_merge()
    install_agent(monkeypatch, GROUPING, rejected, rejected)
    assert run(data_dir, "--max-attempts", "2") == 1
    output = read_output(capsys)
    assert output["status"] == "invalid"
    assert output["stage"] == "merge"
    assert output["attempts"] == 2
    assert output["issues"][0]["path"] == "groups[0].concepts"
    assert "paper 2 'c1'" in output["issues"][0]["message"]
    assert not (data_dir / "normalization").exists()


def test_run_reports_the_grouping_stage_when_the_agent_keeps_naming_concepts_that_do_not_exist(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = copy_fixture(tmp_path)
    write_extractions(data_dir)
    rejected = build_unknown_grouping()
    install_agent(monkeypatch, rejected, rejected)
    assert run(data_dir, "--max-attempts", "2") == 1
    output = read_output(capsys)
    assert output["status"] == "invalid"
    assert output["stage"] == "grouping"
    assert output["issues"][0]["path"] == "groups[0][1]"
    assert not (data_dir / "normalization").exists()


def test_run_reports_the_relation_stage_and_writes_nothing_when_it_keeps_failing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = copy_fixture(tmp_path)
    write_extractions(data_dir)
    install_agent(monkeypatch, GROUPING, VALID_MERGE, UNKNOWN_RELATIONS, UNKNOWN_RELATIONS)
    assert run(data_dir, "--max-attempts", "2") == 1
    output = read_output(capsys)
    assert output["status"] == "invalid"
    assert output["stage"] == "relations"
    assert output["attempts"] == 2
    assert output["issues"][0]["path"] == "concept_relations[0].source_id"
    assert "dense-retrieval" in output["issues"][0]["message"]
    assert not (data_dir / "normalization").exists()


def test_run_reports_every_paper_that_has_not_been_extracted_yet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = copy_fixture(tmp_path)
    install_agent(monkeypatch, GROUPING, VALID_MERGE, VALID_RELATIONS)
    assert run(data_dir) == 2
    output = read_output(capsys)
    assert output["status"] == "error"
    assert "1, 2" in output["message"]


def test_run_reports_an_agent_that_cannot_be_started(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = copy_fixture(tmp_path)
    write_extractions(data_dir)
    install_failing_agent(monkeypatch, "claude was not found, so no normalization can run")
    assert run(data_dir) == 2
    output = read_output(capsys)
    assert output["status"] == "error"
    assert "claude was not found" in output["message"]


def test_run_reports_a_data_directory_without_an_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_agent(monkeypatch, GROUPING, VALID_MERGE, VALID_RELATIONS)
    assert run(tmp_path / "empty") == 2
    assert read_output(capsys)["status"] == "error"


def test_run_reports_an_index_without_papers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "papers").mkdir(parents=True)
    (data_dir / "papers" / "index.json").write_text('{"papers": []}', encoding="utf-8")
    install_agent(monkeypatch, GROUPING, VALID_MERGE, VALID_RELATIONS)
    assert run(data_dir) == 2
    assert read_output(capsys)["message"] == "no papers in the index"


def test_run_passes_the_chosen_model_to_the_agent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = copy_fixture(tmp_path)
    write_extractions(data_dir)
    seen: list[str | None] = []

    def _build(model: str | None = None) -> FakeAgent:
        seen.append(model)
        return FakeAgent(GROUPING, VALID_MERGE, VALID_RELATIONS)

    monkeypatch.setattr(normalize, "ClaudeCodeAgent", _build)
    assert run(data_dir, "--model", "claude-opus-4") == 0
    assert seen == ["claude-opus-4"]


def test_run_passes_the_chosen_embedding_model_to_the_qwen3_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def _build(model_name: str) -> object:
        seen.append(model_name)
        raise SystemExit(0)

    monkeypatch.setattr(normalize, "Qwen3Embedder", _build)
    with pytest.raises(SystemExit):
        main(["--data-dir", str(tmp_path), "--embedding-model", "some/other-model"])
    assert seen == ["some/other-model"]


def test_run_reports_an_embedder_that_cannot_be_set_up(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = copy_fixture(tmp_path)
    write_extractions(data_dir)

    class FailingEmbedder:
        @property
        def model_name(self) -> str:
            return "failing"

        def embed_documents(self, texts: Sequence[str]) -> Any:
            raise EmbedderError("sentence-transformers is not installed")

        def embed_queries(self, texts: Sequence[str]) -> Any:
            return self.embed_documents(texts)

    install_agent(monkeypatch, GROUPING, VALID_MERGE, VALID_RELATIONS)
    monkeypatch.setattr(normalize, "Qwen3Embedder", lambda model_name: FailingEmbedder())
    assert main(["--data-dir", str(data_dir)]) == 2
    output = read_output(capsys)
    assert output["status"] == "error"
    assert "sentence-transformers is not installed" in output["message"]


def test_version_is_printed(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert version("rkgk") in capsys.readouterr().out


def test_help_mentions_that_the_command_takes_no_paper(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    output = capsys.readouterr().out
    assert "--data-dir" in output
    assert "paper_id" not in output


@pytest.mark.parametrize("option", ["--max-attempts", "--neighbors", "--concurrency"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_a_count_below_one_is_a_usage_error(
    option: str, value: str, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--data-dir", str(tmp_path), option, value])
    assert caught.value.code == 2
    assert "must be at least 1" in capsys.readouterr().err


def test_help_names_the_embedder_and_the_grouping_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    output = capsys.readouterr().out
    assert "--embedder" in output
    assert "--embedding-model" in output
    assert "--neighbors" in output
    assert "--concurrency" in output
