"""Turns a concept name into the slug that identifies it in the global vocabulary.

The merge stage derives its own slugs, but the concepts no group holds never reach an agent, so their slug is
derived here instead; both sides therefore read the same rule.
It owns no data and depends on nothing but the slug pattern the domain models declare.
"""

import re

# Everything a slug may not hold, in runs, so that `Page-Aligned  Chunking` folds to one `-` and not to three.
_SEPARATORS = re.compile(r"[^a-z0-9]+")


def derive_slug(name: str) -> str:
    """Fold a name into the slug shape: lower case, runs of anything else as one `-`, no `-` at either end.

    A name written entirely outside `[a-z0-9]`, such as a Japanese one, leaves nothing to keep and comes back
    as the empty string; naming the concept is then the caller's job, because only it knows what to fall back on.
    """
    return _SEPARATORS.sub("-", name.lower()).strip("-")
