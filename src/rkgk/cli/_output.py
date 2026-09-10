"""How every subcommand talks to its caller.

A command prints one JSON object on stdout and reports the outcome as an exit code, so an agent can branch on the
code and read the details from the same output without parsing prose.
"""

import json

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_ERROR = 2


def print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False))
