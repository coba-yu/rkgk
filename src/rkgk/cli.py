"""Command-line entry point for rkgk.

Subcommands register themselves via `register_command` so that later PRs can
add functionality without editing `main`.
"""

import argparse
from collections.abc import Callable
from importlib.metadata import version

Registrar = Callable[[argparse.ArgumentParser], None]

# Each entry is (name, help, register_fn). register_fn(subparser) must add
# whatever arguments the command needs and call
# subparser.set_defaults(func=...) to wire up its handler.
_COMMANDS: list[tuple[str, str, Registrar]] = []


def register_command(name: str, help: str) -> Callable[[Registrar], Registrar]:
    """Decorator that registers a subcommand's argument setup function."""

    def decorator(register_fn: Registrar) -> Registrar:
        _COMMANDS.append((name, help, register_fn))
        return register_fn

    return decorator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rkgk")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('rkgk')}",
    )
    subparsers = parser.add_subparsers(dest="command")
    for name, help_text, register_fn in _COMMANDS:
        subparser = subparsers.add_parser(name, help=help_text)
        register_fn(subparser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
