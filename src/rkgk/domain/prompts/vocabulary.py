"""Builds the vocabulary section both prompts put between their task and their rules.

The wording stays in Python rather than in a .md because it is a heading and one sentence introducing text that
`describe_vocabulary()` generates from the enums.
"""

from rkgk.domain.models.vocabulary import describe_vocabulary


def build_vocabulary_section() -> list[str]:
    """Write the heading, the sentence that binds the agent to the names, and the rendered enums.

    The rules that follow name relations such as `proposes`, so this section comes before them.
    """
    return [
        "## Vocabulary",
        "",
        "ノード種別と関係名は以下のものを使い、表記は示したとおりにする。",
        "",
        describe_vocabulary().rstrip("\n"),
    ]
