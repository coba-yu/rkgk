"""Checks that an extraction's evidence really appears in the paper it claims to describe.

This module reads the extraction and paper models but owns no data of its own.
"""

import re

from rkgk.domain.models.paper import Paper
from rkgk.domain.models.paper_extraction import PageEvidence, PaperExtraction, PaperExtractionIssue

_WHITESPACE_RUN = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse every run of whitespace into one space and strip the ends.

    A quote and the page it came from differ in line breaks and indentation once the agent has copied it,
    so both sides are normalized before they are compared.
    """
    return _WHITESPACE_RUN.sub(" ", text).strip()


def _check_evidence(
    evidence: tuple[PageEvidence, ...], prefix: str, page_texts: dict[int, str], page_count: int
) -> list[PaperExtractionIssue]:
    issues: list[PaperExtractionIssue] = []
    for index, item in enumerate(evidence):
        path = f"{prefix}.evidence[{index}]"
        if item.page not in page_texts:
            issues.append(
                PaperExtractionIssue(path=f"{path}.page", message=f"page {item.page} is outside 1..{page_count}")
            )
            continue
        if normalize_whitespace(item.quote) not in page_texts[item.page]:
            issues.append(
                PaperExtractionIssue(path=f"{path}.quote", message=f"is not found in the text of page {item.page}")
            )
    return issues


def check_extraction_against_paper(result: PaperExtraction, paper: Paper) -> tuple[PaperExtractionIssue, ...]:
    """Report every place where the extraction disagrees with the paper text.

    All issues are collected instead of raising on the first one, because an agent fixing its output needs the
    whole list to converge in one more attempt.
    """
    issues: list[PaperExtractionIssue] = []
    if result.paper_id != paper.meta.id:
        issues.append(
            PaperExtractionIssue(
                path="paper_id", message=f"is {result.paper_id}, but the paper being checked is {paper.meta.id}"
            )
        )
    page_texts = {page.number: normalize_whitespace(page.text) for page in paper.pages}
    for index, edge in enumerate(result.paper_concepts):
        issues += _check_evidence(edge.evidence, f"paper_concepts[{index}]", page_texts, paper.meta.page_count)
    for index, concept_edge in enumerate(result.concept_relations):
        issues += _check_evidence(
            concept_edge.evidence, f"concept_relations[{index}]", page_texts, paper.meta.page_count
        )
    return tuple(issues)
