import re

import pytest

from rkgk.domain.models.base import SLUG_PATTERN
from rkgk.domain.services.slugs import derive_slug


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Knowledge Graph", "knowledge-graph"),
        ("Retrieval-Augmented Generation", "retrieval-augmented-generation"),
        ("GPT-4", "gpt-4"),
        ("Page-Aligned  Chunking", "page-aligned-chunking"),
        ("  Dense Retrieval  ", "dense-retrieval"),
        ("End-to-End (E2E)", "end-to-end-e2e"),
        ("k-NN / ANN", "k-nn-ann"),
        ("rag", "rag"),
    ],
)
def test_a_name_becomes_its_lower_case_words_joined_by_one_hyphen(name: str, expected: str) -> None:
    assert derive_slug(name) == expected


@pytest.mark.parametrize("name", ["Knowledge Graph", "GPT-4", "End-to-End (E2E)", "k-NN / ANN"])
def test_a_derived_slug_matches_the_pattern_the_models_require(name: str) -> None:
    assert re.match(SLUG_PATTERN, derive_slug(name)) is not None


@pytest.mark.parametrize("name", ["検索拡張生成", "---", "   ", "…"])
def test_a_name_that_leaves_nothing_a_slug_can_keep_becomes_the_empty_string(name: str) -> None:
    assert derive_slug(name) == ""
