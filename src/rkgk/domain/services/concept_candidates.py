"""Finds the pairs of extracted concepts that may be the same, by comparing their embeddings.

Asking an agent about every pair of a corpus is quadratic and asking it about all concepts at once is one answer
too large to finish, so the embeddings narrow the field first: each concept keeps its nearest neighbours and the
pairs go to the grouping stage as a hint.
The cosine is never a threshold here, only a number the agent is shown, because two spellings of one method can
sit far apart and two unrelated methods of one field can sit close together.
It reads the extraction and normalization models and owns no data.
"""

from collections.abc import Sequence

from rkgk.domain.embedders import Embedder
from rkgk.domain.models.concept_normalization import CandidatePair, LocalConceptRef
from rkgk.domain.models.paper_extraction import ExtractedConcept, PaperExtraction
from rkgk.domain.services.vector_search import compute_cosine_scores, normalize_rows, rank_top_rows

DEFAULT_NEIGHBORS = 5


def build_extracted_concept_embedding_text(concept: ExtractedConcept) -> str:
    """Render an extracted concept the way `concept_embedding_text` renders a normalized one.

    The two formats are the same on purpose: a concept is compared with other concepts here and with a query in
    the index, and a concept that reads differently in the two places would be embedded twice as two things.
    """
    heading = concept.name
    if concept.aliases:
        heading = f"{heading} ({', '.join(concept.aliases)})"
    if not concept.description:
        return heading
    return f"{heading}\n{concept.description}"


def collect_candidate_pairs(
    extractions: Sequence[PaperExtraction], embedder: Embedder, neighbors: int = DEFAULT_NEIGHBORS
) -> tuple[CandidatePair, ...]:
    """Embed every extracted concept and pair each one with its `neighbors` closest others, best pair first.

    A pair is unordered, so `(a, b)` and `(b, a)` come out once, holding the score of whichever of the two saw
    the other first; the two differ only in the last bits of a float, and picking one keeps the output stable.
    Concepts of one paper are compared like any others, because a paper can extract the same idea twice under
    two names.
    """
    if neighbors < 1:
        raise ValueError(f"neighbors must be at least 1, got {neighbors}")
    refs = tuple(
        LocalConceptRef(paper_id=extraction.paper_id, local_id=concept.local_id)
        for extraction in extractions
        for concept in extraction.concepts
    )
    if len(refs) < 2:
        # One concept has nobody to be paired with, and no concept at all has nothing to embed.
        return ()
    texts = [
        build_extracted_concept_embedding_text(concept)
        for extraction in extractions
        for concept in extraction.concepts
    ]
    vectors = normalize_rows(embedder.embed_documents(texts))
    scores = compute_cosine_scores(vectors, vectors)

    pairs: list[CandidatePair] = []
    seen: set[tuple[int, int]] = set()
    for index, row in enumerate(scores):
        taken = 0
        # One rank more than asked for, because a concept scores 1.0 against itself and takes the first place.
        for other in rank_top_rows(row, neighbors + 1):
            if other == index:
                continue
            if taken == neighbors:
                break
            taken += 1
            key = (min(index, other), max(index, other))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(CandidatePair(left=refs[key[0]], right=refs[key[1]], score=float(row[other])))
    # A stable sort, so pairs of equal score keep the order the concepts were extracted in and two runs over one
    # corpus write the same prompt.
    return tuple(sorted(pairs, key=lambda pair: -pair.score))
