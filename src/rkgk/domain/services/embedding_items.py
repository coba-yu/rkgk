"""Lists what the index embeds: every chunk, every Japanese summary, and every normalized concept.

A Japanese theme has to find English papers, and chunk text alone answers that poorly, so the summaries written
in Japanese and the concept names are embedded next to the chunks as extra ways in.
It reads the chunk, extraction and normalization models and owns no data.
"""

from collections.abc import Sequence

from rkgk.domain.models.chunk import Chunk
from rkgk.domain.models.concept_normalization import NormalizedConcept
from rkgk.domain.models.embedding import EmbeddedItem, EmbeddedItemKind
from rkgk.domain.models.paper_extraction import PaperExtraction


def concept_embedding_text(concept: NormalizedConcept) -> str:
    """Render a concept as one short text: the name with its aliases in brackets, then the description.

    The aliases sit next to the name so that a query in another spelling, or in Japanese, lands on the
    same vector.
    """
    heading = concept.canonical_name
    if concept.aliases:
        heading = f"{heading} ({', '.join(concept.aliases)})"
    if not concept.description:
        return heading
    return f"{heading}\n{concept.description}"


def build_embedding_items(
    chunks: Sequence[Chunk],
    extractions: Sequence[PaperExtraction],
    concepts: Sequence[NormalizedConcept],
) -> tuple[EmbeddedItem, ...]:
    """Return the items in a fixed order: chunks, then summaries, then concepts, each in the order given.

    The order is the row order of the vectors, so it is fixed here rather than left to the caller.
    """
    items = [
        EmbeddedItem(kind=EmbeddedItemKind.CHUNK, ref=chunk.id, paper_id=chunk.paper_id, text=chunk.text)
        for chunk in chunks
    ]
    items.extend(
        EmbeddedItem(
            kind=EmbeddedItemKind.SUMMARY,
            ref=str(extraction.paper_id),
            paper_id=extraction.paper_id,
            text=extraction.summary_ja,
        )
        for extraction in extractions
    )
    items.extend(
        EmbeddedItem(kind=EmbeddedItemKind.CONCEPT, ref=concept.id, text=concept_embedding_text(concept))
        for concept in concepts
    )
    return tuple(items)
