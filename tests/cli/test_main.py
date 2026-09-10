from importlib.metadata import version

import pytest

from rkgk.cli import main


def test_version_is_printed(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert version("rkgk") in capsys.readouterr().out


def test_no_subcommand_prints_help_and_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out


def test_help_lists_the_extract_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    assert "extract" in capsys.readouterr().out


def test_an_unknown_command_is_rejected() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["publish"])
    assert caught.value.code == 2
