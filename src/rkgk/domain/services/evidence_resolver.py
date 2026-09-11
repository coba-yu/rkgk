"""Ties the evidence of an extraction to the chunks that retrieval returns.

An extraction quotes a page, because chunks do not exist when the agent reads the paper.
Once the paper is chunked, this module finds the chunk that holds each quote so an answer can point at the
chunk it retrieved.
It reads the extraction and chunk models and owns no data.
"""

from collections.abc import Sequence

from rkgk.domain.models.chunk import Chunk
from rkgk.domain.models.graph import Evidence, EvidenceResolutionError, UnresolvedEvidence
from rkgk.domain.models.paper_extraction import ExtractedEvidence, PaperExtraction
from rkgk.domain.services.paper_extraction import normalize_whitespace


def find_chunk(page: int, quote: str, chunks: Sequence[Chunk]) -> Chunk | None:
    """Return the first chunk of the page whose text contains the quote after whitespace normalization.

    Only an exact match counts, as in the extraction check, so the chunk id never points at text that merely
    resembles the quote.
    The first chunk wins when several contain the quote: a repeated sentence is the same evidence wherever it
    appears, and the earliest occurrence is the one a reader meets first.
    """
    needle = normalize_whitespace(quote)
    for chunk in chunks:
        if chunk.page_start <= page <= chunk.page_end and needle in normalize_whitespace(chunk.text):
            return chunk
    return None


def resolve_extraction_evidence(
    extraction: PaperExtraction, chunks: Sequence[Chunk]
) -> dict[ExtractedEvidence, Evidence]:
    """Attach a chunk id to every piece of evidence in the extraction, keyed by the evidence it came from.

    The result is a mapping rather than a rebuilt extraction, so the same quote cited on several edges is
    resolved once and every edge looks up the same answer.
    Every unresolved quote is collected before raising, because a quote fails here only when it straddles a
    chunk boundary, and whoever fixes the extraction needs the whole list.
    """
    foreign = [chunk.id for chunk in chunks if chunk.paper_id != extraction.paper_id]
    if foreign:
        raise ValueError(f"chunks {foreign} do not belong to paper {extraction.paper_id}")

    resolved: dict[ExtractedEvidence, Evidence] = {}
    unresolved: dict[ExtractedEvidence, UnresolvedEvidence] = {}
    edges = [*extraction.paper_concepts, *extraction.concept_relations]
    for item in (item for edge in edges for item in edge.evidence):
        if item in resolved or item in unresolved:
            continue
        chunk = find_chunk(item.page, item.quote, chunks)
        if chunk is None:
            unresolved[item] = UnresolvedEvidence(paper_id=extraction.paper_id, page=item.page, quote=item.quote)
            continue
        resolved[item] = Evidence(page=item.page, quote=item.quote, chunk_id=chunk.id)
    if unresolved:
        raise EvidenceResolutionError(tuple(unresolved.values()))
    return resolved
