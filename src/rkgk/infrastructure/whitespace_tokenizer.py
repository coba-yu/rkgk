"""Tokenizer that counts whitespace-separated words.

It stands in for a real tokenizer in tests and local runs, where loading a model's vocabulary would cost more
than the answer is worth.
Its counts are close enough to size a chunk, but never the same as a real tokenizer's, so an index built for
retrieval must be built with the real one.
"""


class WhitespaceTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text.split())
