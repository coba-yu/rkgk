import pytest

from rkgk.domain.models.paper import Page, Paper, PaperMeta, PaperPreprocessInfo
from rkgk.domain.models.paper_extraction import (
    ExtractedConcept,
    ExtractedConceptEdge,
    ExtractedEvidence,
    ExtractedPaperConceptEdge,
    PaperExtraction,
)
from rkgk.domain.models.vocabulary import ConceptRelationType, ConceptType, PaperConceptRelation
from rkgk.domain.services.paper_extraction import check_extraction_against_paper, normalize_whitespace

PAPER = Paper(
    meta=PaperMeta(
        id=1,
        title="Retrieval-Augmented Generation for Conference Paper Search",
        authors=("Ada Lovelace",),
        year=2026,
        venue="NeurIPS",
        page_count=2,
        preprocess=PaperPreprocessInfo(tool="pymupdf", version="1.24.0", processed_at="2026-01-01T00:00:00Z"),
    ),
    pages=(
        Page(number=1, text="We study how a retrieval-augmented\ngeneration pipeline helps a reader.\n"),
        Page(number=2, text="The pipeline splits every paper into page-aligned chunks.\n"),
    ),
)

CONCEPTS = (
    ExtractedConcept(local_id="c1", name="Retrieval-Augmented Generation", type=ConceptType.METHOD),
    ExtractedConcept(local_id="c2", name="Page-Aligned Chunking", type=ConceptType.METHOD),
)


def evidence(page: int = 1, quote: str = "retrieval-augmented") -> tuple[ExtractedEvidence, ...]:
    return (ExtractedEvidence(page=page, quote=quote),)


def build_result(**overrides: object) -> PaperExtraction:
    payload: dict[str, object] = {
        "schema_version": 1,
        "paper_id": 1,
        "summary_ja": "この論文は検索拡張生成のパイプラインを提案する。",
        "concepts": CONCEPTS,
        "paper_concepts": (
            ExtractedPaperConceptEdge(concept_id="c1", relation=PaperConceptRelation.PROPOSES, evidence=evidence()),
        ),
    }
    payload.update(overrides)
    return PaperExtraction.model_validate(payload)


def test_a_result_that_matches_the_paper_is_accepted() -> None:
    assert check_extraction_against_paper(build_result(), PAPER) == ()


def test_a_paper_id_that_differs_from_the_paper_is_reported() -> None:
    issues = check_extraction_against_paper(build_result(paper_id=2), PAPER)
    assert [(issue.path, "2" in issue.message) for issue in issues] == [("paper_id", True)]


def test_a_page_beyond_the_paper_is_reported_with_its_path() -> None:
    edge = ExtractedPaperConceptEdge(
        concept_id="c1", relation=PaperConceptRelation.USES, evidence=evidence(page=3, quote="pipeline")
    )
    issues = check_extraction_against_paper(build_result(paper_concepts=(edge,)), PAPER)
    assert [issue.path for issue in issues] == ["paper_concepts[0].evidence[0].page"]
    assert "outside 1..2" in issues[0].message


def test_a_quote_that_is_not_on_the_page_is_reported_with_its_path() -> None:
    edge = ExtractedPaperConceptEdge(
        concept_id="c1", relation=PaperConceptRelation.USES, evidence=evidence(page=2, quote="dense retrieval")
    )
    issues = check_extraction_against_paper(build_result(paper_concepts=(edge,)), PAPER)
    assert [issue.path for issue in issues] == ["paper_concepts[0].evidence[0].quote"]
    assert "page 2" in issues[0].message


def test_a_quote_from_another_page_of_the_same_paper_is_reported() -> None:
    edge = ExtractedPaperConceptEdge(
        concept_id="c1", relation=PaperConceptRelation.USES, evidence=evidence(page=2, quote="retrieval-augmented")
    )
    issues = check_extraction_against_paper(build_result(paper_concepts=(edge,)), PAPER)
    assert [issue.path for issue in issues] == ["paper_concepts[0].evidence[0].quote"]


def test_a_quote_spanning_a_line_break_matches_after_whitespace_normalization() -> None:
    edge = ExtractedPaperConceptEdge(
        concept_id="c1",
        relation=PaperConceptRelation.USES,
        evidence=evidence(page=1, quote="  retrieval-augmented generation   pipeline  "),
    )
    assert check_extraction_against_paper(build_result(paper_concepts=(edge,)), PAPER) == ()


def test_a_quote_in_a_concept_relation_is_checked_with_its_own_path() -> None:
    edge = ExtractedConceptEdge(
        source_id="c1",
        target_id="c2",
        relation=ConceptRelationType.USED_FOR,
        evidence=evidence(page=1, quote="not in the paper"),
    )
    issues = check_extraction_against_paper(build_result(concept_relations=(edge,)), PAPER)
    assert [issue.path for issue in issues] == ["concept_relations[0].evidence[0].quote"]


def test_every_issue_is_reported_instead_of_only_the_first() -> None:
    edge = ExtractedPaperConceptEdge(
        concept_id="c1",
        relation=PaperConceptRelation.USES,
        evidence=(
            ExtractedEvidence(page=9, quote="pipeline"),
            ExtractedEvidence(page=1, quote="dense retrieval"),
        ),
    )
    issues = check_extraction_against_paper(build_result(paper_id=2, paper_concepts=(edge,)), PAPER)
    assert [issue.path for issue in issues] == [
        "paper_id",
        "paper_concepts[0].evidence[0].page",
        "paper_concepts[0].evidence[1].quote",
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("a  b", "a b"), (" a\nb ", "a b"), ("a\t\tb", "a b"), ("", ""), ("\n", "")],
)
def test_normalize_whitespace_collapses_runs_and_strips(text: str, expected: str) -> None:
    assert normalize_whitespace(text) == expected
