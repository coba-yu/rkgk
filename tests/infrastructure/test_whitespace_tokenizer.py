from rkgk.infrastructure.whitespace_tokenizer import WhitespaceTokenizer


def test_whitespace_tokenizer_counts_words_across_every_kind_of_whitespace() -> None:
    assert WhitespaceTokenizer().count_tokens("one two\nthree\tfour  five\n\nsix") == 6


def test_whitespace_tokenizer_counts_nothing_in_an_empty_string() -> None:
    assert WhitespaceTokenizer().count_tokens("") == 0
