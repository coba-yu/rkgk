"""Port for the token counting the domain needs.

Chunking only has to know how many tokens a piece of text costs, so it can pack paragraphs under a budget.
Naming the port here keeps the domain free of a specific tokenizer library: the real runs use the Qwen3
tokenizer, while the tests count whitespace-separated words.
Counting is the whole contract, with no encode or decode: every implementation stays trivial, and chunk text
remains the original characters instead of a decode of an encode.
"""

from typing import Protocol


class Tokenizer(Protocol):
    """Something that can say how many tokens a text costs."""

    def count_tokens(self, text: str) -> int: ...
