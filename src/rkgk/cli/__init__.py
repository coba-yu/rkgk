"""Command-line entry point for rkgk.

Each command lives in its own module and adds itself to the parser in `register(subparsers)`, so a new command
is one import plus one entry in `COMMANDS` and never a change to `main`.
"""

import argparse
from importlib.metadata import version

from rkgk.cli import extract
from rkgk.cli._output import EXIT_ERROR

COMMANDS = (extract,)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rkgk")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('rkgk')}",
    )
    subparsers = parser.add_subparsers(dest="command")
    for command in COMMANDS:
        command.register(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return EXIT_ERROR
    return args.func(args)
