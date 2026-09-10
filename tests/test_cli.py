import argparse
from importlib.metadata import version

import pytest

from rkgk.cli import _COMMANDS, main, register_command


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    assert version("rkgk") in capsys.readouterr().out


def test_no_subcommand_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out


def test_registered_command_is_callable() -> None:
    before = list(_COMMANDS)

    def _register(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("name")
        subparser.set_defaults(func=lambda args: 0 if args.name == "ok" else 1)

    register_command("dummy", "a dummy command for testing")(_register)
    try:
        assert main(["dummy", "ok"]) == 0
        assert main(["dummy", "nope"]) == 1
    finally:
        _COMMANDS[:] = before
