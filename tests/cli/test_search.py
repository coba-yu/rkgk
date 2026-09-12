import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from rkgk.cli import build, search
from rkgk.cli.search import main
from rkgk.infrastructure.fake_embedder import FakeEmbedder
from tests.cli.test_build import copy_fixture, write_extractions, write_normalization

QUERY = "検索拡張生成の論文を読みたい"


def read_output(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


def build_index(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    """Leave a data directory with the index the build command writes from the fixture papers.

    What the build printed is read away here, so a test reads the JSON of the search and not of both commands.
    """
    data_dir = copy_fixture(tmp_path)
    write_extractions(data_dir)
    write_normalization(data_dir)
    assert build.main(["--data-dir", str(data_dir), "--embedder", "fake"]) == 0
    capsys.readouterr()
    return data_dir


def record_model_names(monkeypatch: pytest.MonkeyPatch, embedder: FakeEmbedder) -> list[str]:
    """Replace the Qwen3 embedder by a fake one, collecting the model name the command asked for."""
    seen: list[str] = []

    def _build(model_name: str) -> FakeEmbedder:
        seen.append(model_name)
        return embedder

    monkeypatch.setattr(search, "Qwen3Embedder", _build)
    return seen


def test_run_reports_the_direct_candidates_with_their_title_summary_and_hits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = build_index(tmp_path, capsys)
    assert main(["--data-dir", str(data_dir), "--embedder", "fake", QUERY]) == 0
    output = read_output(capsys)
    assert output["status"] == "ok"
    assert output["queries"] == [QUERY]
    assert output["config"]["top_k"] == 10
    # The fixture index holds nine items and the default top_k covers all of them, so both papers are found.
    assert [candidate["paper_id"] for candidate in output["direct_candidates"]] == [1, 2]
    assert output["graph_candidates"] == []
    for candidate in output["direct_candidates"]:
        assert candidate["title"]
        assert candidate["summary_ja"]
        assert candidate["hits"]


def test_run_reports_a_data_directory_whose_index_has_not_been_built_yet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = copy_fixture(tmp_path)
    assert main(["--data-dir", str(data_dir), "--embedder", "fake", QUERY]) == 2
    output = read_output(capsys)
    assert output["status"] == "error"
    assert str(data_dir / "index" / "manifest.json") in output["message"]


def test_run_embeds_the_queries_with_the_model_the_index_was_built_with(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = build_index(tmp_path, capsys)
    seen = record_model_names(monkeypatch, FakeEmbedder())
    assert main(["--data-dir", str(data_dir), QUERY]) == 0
    assert seen == ["fake-8"]
    assert read_output(capsys)["status"] == "ok"


def test_run_reports_a_chosen_model_that_is_not_the_one_of_the_index_as_invalid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = build_index(tmp_path, capsys)
    record_model_names(monkeypatch, FakeEmbedder(dimension=4))
    assert main(["--data-dir", str(data_dir), "--embedding-model", "fake-4", QUERY]) == 1
    output = read_output(capsys)
    assert output["status"] == "invalid"
    assert output["reason"] == "embedding_model_mismatch"
    assert output["index_model"] == "fake-8"
    assert output["embedder_model"] == "fake-4"
    assert "fake-8" in output["message"]


@pytest.mark.parametrize(
    ("option", "value", "problem"),
    [
        ("--top-k", "0", "must be at least 1"),
        ("--max-hops", "-1", "must be at least 0"),
        ("--max-graph-candidates", "-1", "must be at least 0"),
        ("--generic-concept-threshold", "1.5", "must be between 0.0 and 1.0"),
    ],
)
def test_a_setting_outside_its_range_is_a_usage_error(
    option: str, value: str, problem: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--data-dir", str(tmp_path), option, value, QUERY])
    assert caught.value.code == 2
    assert problem in capsys.readouterr().err


def test_version_is_printed(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert version("rkgk") in capsys.readouterr().out


def test_help_mentions_the_queries_and_the_number_of_hits_to_take(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    output = capsys.readouterr().out
    assert "QUERY" in output
    assert "--top-k" in output
