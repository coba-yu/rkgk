import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rkgk.domain.models.paper import Page, Paper, PaperMeta, PaperPreprocessInfo, parse_page
from rkgk.domain.models.paper_extraction import (
    EXTRACTION_SCHEMA_VERSION,
    LOCAL_CONCEPT_ID_PATTERN,
    ExtractedConcept,
    ExtractedConceptEdge,
    ExtractedEvidence,
    ExtractedPaperConceptEdge,
    ExtractionIssue,
    PaperExtraction,
    build_extraction_prompt,
    build_extraction_schema,
    check_extraction_against_paper,
    normalize_whitespace,
)
from rkgk.domain.models.vocabulary import (
    ConceptRelationType,
    ConceptType,
    PaperConceptRelation,
    describe_vocabulary,
)

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


def test_a_result_declares_the_current_schema_version() -> None:
    assert build_result().schema_version == EXTRACTION_SCHEMA_VERSION


def test_another_schema_version_is_rejected() -> None:
    with pytest.raises(ValidationError):
        build_result(schema_version=2)


def test_duplicate_local_ids_are_rejected() -> None:
    duplicate = ExtractedConcept(local_id="c1", name="Dense Retrieval", type=ConceptType.METHOD)
    with pytest.raises(ValidationError, match="'c1' more than once"):
        build_result(concepts=(CONCEPTS[0], duplicate))


@pytest.mark.parametrize("local_id", ["c0", "c01", "concept-1", "1", ""])
def test_local_ids_outside_the_pattern_are_rejected(local_id: str) -> None:
    with pytest.raises(ValidationError):
        ExtractedConcept(local_id=local_id, name="Dense Retrieval", type=ConceptType.METHOD)


def test_paper_concept_edge_naming_an_undeclared_concept_is_rejected_with_that_id() -> None:
    edge = ExtractedPaperConceptEdge(concept_id="c9", relation=PaperConceptRelation.USES, evidence=evidence())
    with pytest.raises(ValidationError, match="undeclared concept 'c9'"):
        build_result(paper_concepts=(edge,))


def test_concept_relation_naming_an_undeclared_concept_is_rejected_with_that_id() -> None:
    edge = ExtractedConceptEdge(
        source_id="c1", target_id="c7", relation=ConceptRelationType.USED_FOR, evidence=evidence()
    )
    with pytest.raises(ValidationError, match="undeclared concept 'c7'"):
        build_result(concept_relations=(edge,))


def test_the_same_concept_and_relation_twice_is_rejected() -> None:
    edge = ExtractedPaperConceptEdge(concept_id="c1", relation=PaperConceptRelation.PROPOSES, evidence=evidence())
    with pytest.raises(ValidationError, match="repeats 'c1' with relation 'proposes'"):
        build_result(paper_concepts=(edge, edge))


def test_the_same_concept_pair_and_relation_twice_is_rejected() -> None:
    edge = ExtractedConceptEdge(
        source_id="c1", target_id="c2", relation=ConceptRelationType.RELATED_TO, evidence=evidence()
    )
    with pytest.raises(ValidationError, match="repeats 'c1' -> 'c2' with relation 'related_to'"):
        build_result(concept_relations=(edge, edge))


def test_the_same_concept_pair_with_another_relation_is_accepted() -> None:
    result = build_result(
        concept_relations=(
            ExtractedConceptEdge(
                source_id="c1", target_id="c2", relation=ConceptRelationType.RELATED_TO, evidence=evidence()
            ),
            ExtractedConceptEdge(
                source_id="c1", target_id="c2", relation=ConceptRelationType.PART_OF, evidence=evidence()
            ),
        )
    )
    assert len(result.concept_relations) == 2


def test_a_relation_from_a_concept_to_itself_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        ExtractedConceptEdge(source_id="c1", target_id="c1", relation=ConceptRelationType.IS_A, evidence=evidence())


def test_an_edge_without_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedPaperConceptEdge(concept_id="c1", relation=PaperConceptRelation.USES, evidence=())


@pytest.mark.parametrize("quote", ["", "   \n "])
def test_a_blank_quote_is_rejected(quote: str) -> None:
    with pytest.raises(ValidationError):
        ExtractedEvidence(page=1, quote=quote)


def test_a_blank_summary_is_rejected() -> None:
    with pytest.raises(ValidationError):
        build_result(summary_ja="  ")


def test_a_result_without_concepts_is_rejected() -> None:
    with pytest.raises(ValidationError):
        build_result(concepts=())


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


def test_the_schema_describes_the_summary_and_the_local_id_pattern() -> None:
    schema = build_extraction_schema()
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert "summary_ja" in properties
    assert LOCAL_CONCEPT_ID_PATTERN in json.dumps(schema)


def test_the_schema_is_json_serializable() -> None:
    assert json.loads(json.dumps(build_extraction_schema())) == build_extraction_schema()


FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures"
PROMPT_SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "extraction_prompt.md"


def load_fixture_paper() -> Paper:
    paper_dir = FIXTURE_DIR / "papers" / "0001"
    meta = PaperMeta.model_validate(json.loads((paper_dir / "paper.json").read_text(encoding="utf-8")))
    pages = tuple(
        parse_page(number, (paper_dir / "pages" / f"{number:03d}.md").read_text(encoding="utf-8"))
        for number in range(1, meta.page_count + 1)
    )
    return Paper(meta=meta, pages=pages)


def test_the_prompt_for_the_fixture_paper_matches_the_snapshot() -> None:
    assert build_extraction_prompt(load_fixture_paper()) == PROMPT_SNAPSHOT_PATH.read_text(encoding="utf-8")


def test_the_prompt_is_deterministic() -> None:
    paper = load_fixture_paper()
    assert build_extraction_prompt(paper) == build_extraction_prompt(paper)


def test_the_prompt_carries_the_vocabulary_and_the_paper_id() -> None:
    prompt = build_extraction_prompt(load_fixture_paper())
    assert describe_vocabulary().rstrip("\n") in prompt
    assert "Set `paper_id` to 1." in prompt
    assert f"Set `schema_version` to {EXTRACTION_SCHEMA_VERSION}." in prompt


def test_the_prompt_holds_every_page_with_its_number_and_text() -> None:
    prompt = build_extraction_prompt(load_fixture_paper())
    assert prompt.count("## Page ") == 3
    assert "## Page 2\n\n## Method" in prompt
    assert "<!-- equation: 1 -->" in prompt


def test_the_prompt_ends_by_asking_for_the_json_alone() -> None:
    assert build_extraction_prompt(load_fixture_paper()).endswith("Return only the JSON object.\n")


def test_a_first_attempt_mentions_neither_a_previous_answer_nor_issues() -> None:
    prompt = build_extraction_prompt(load_fixture_paper())
    assert "Previous attempt" not in prompt
    assert "## Issues" not in prompt


def test_a_retry_repeats_the_rejected_json_and_the_issues() -> None:
    previous = {"paper_id": 1, "summary_ja": "要約"}
    issues = (ExtractionIssue(path="paper_concepts[0].evidence[0].quote", message="is not found on page 1"),)
    prompt = build_extraction_prompt(load_fixture_paper(), previous, issues)
    assert '"summary_ja": "要約"' in prompt
    assert "- paper_concepts[0].evidence[0].quote: is not found on page 1" in prompt
    assert "Return a complete corrected JSON object that fixes every issue above." in prompt
