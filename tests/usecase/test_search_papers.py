from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from rkgk.domain.models.concept_normalization import (
    ConceptNormalization,
    GeneralKnowledgeEdge,
    LocalConceptRef,
    NormalizedConcept,
)
from rkgk.domain.models.embedding import EmbeddedItemKind
from rkgk.domain.models.manifest import EmbeddingModelMismatchError
from rkgk.domain.models.paper_extraction import (
    ExtractedConcept,
    ExtractedConceptEdge,
    ExtractedPaperConceptEdge,
    PageEvidence,
    PaperExtraction,
)
from rkgk.domain.models.search import SearchConfig, SearchResult, TraversalPath
from rkgk.domain.models.vocabulary import ConceptRelationType, ConceptType, PaperConceptRelation
from rkgk.domain.repositories.index import IndexNotFoundError
from rkgk.infrastructure.whitespace_tokenizer import WhitespaceTokenizer
from rkgk.usecase.build_index import BuildIndexUseCase
from rkgk.usecase.search_papers import SearchPapersUseCase
from tests.usecase.fakes import (
    FakeConceptNormalizationRepository,
    FakeIndexRepository,
    FakePaperExtractionRepository,
    FakePaperRepository,
    build_paper,
)

RAG = "retrieval-augmented-generation"
CHUNKING = "page-aligned-chunking"
KNOWLEDGE_GRAPH = "knowledge-graph"

S3_URI_OF_PAPER_1 = "s3://rkgk-papers/0001.pdf"

PAPERS = {
    1: build_paper(
        1,
        "We study a retrieval-augmented generation pipeline.\n",
        "The pipeline splits every paper into page-aligned chunks.\n",
        title="Retrieval-Augmented Generation for Conference Paper Search",
        s3_uri=S3_URI_OF_PAPER_1,
    ),
    2: build_paper(
        2,
        "We use a knowledge graph as a retrieval backbone.\n",
        title="Knowledge Graphs as Retrieval Backbones",
    ),
    3: build_paper(
        3,
        "We apply retrieval-augmented generation to conference paper search.\n",
        title="Applying Retrieval-Augmented Generation to Paper Search",
    ),
}

SUMMARY_OF_PAPER_1 = "この論文は検索拡張生成のパイプラインを提案する。"
SUMMARY_OF_PAPER_2 = "この論文は知識グラフを検索の骨格として使う。"
SUMMARY_OF_PAPER_3 = "この論文は検索拡張生成を論文検索に適用する。"

EXTRACTIONS = (
    PaperExtraction(
        schema_version=1,
        paper_id=1,
        summary_ja=SUMMARY_OF_PAPER_1,
        concepts=(
            ExtractedConcept(local_id="c1", name="Retrieval-Augmented Generation", type=ConceptType.METHOD),
            ExtractedConcept(local_id="c2", name="Page-Aligned Chunking", type=ConceptType.METHOD),
        ),
        paper_concepts=(
            ExtractedPaperConceptEdge(
                concept_id="c1",
                relation=PaperConceptRelation.PROPOSES,
                evidence=(PageEvidence(page=1, quote="retrieval-augmented generation pipeline"),),
            ),
            ExtractedPaperConceptEdge(
                concept_id="c2",
                relation=PaperConceptRelation.USES,
                evidence=(PageEvidence(page=2, quote="page-aligned chunks"),),
            ),
        ),
        concept_relations=(
            ExtractedConceptEdge(
                source_id="c2",
                target_id="c1",
                relation=ConceptRelationType.PART_OF,
                evidence=(PageEvidence(page=2, quote="splits every paper into page-aligned chunks"),),
            ),
        ),
    ),
    PaperExtraction(
        schema_version=1,
        paper_id=2,
        summary_ja=SUMMARY_OF_PAPER_2,
        concepts=(ExtractedConcept(local_id="c1", name="Knowledge Graph", type=ConceptType.METHOD),),
        paper_concepts=(
            ExtractedPaperConceptEdge(
                concept_id="c1",
                relation=PaperConceptRelation.USES,
                evidence=(PageEvidence(page=1, quote="knowledge graph as a retrieval backbone"),),
            ),
        ),
    ),
    PaperExtraction(
        schema_version=1,
        paper_id=3,
        summary_ja=SUMMARY_OF_PAPER_3,
        concepts=(ExtractedConcept(local_id="c1", name="Retrieval-Augmented Generation", type=ConceptType.METHOD),),
        paper_concepts=(
            ExtractedPaperConceptEdge(
                concept_id="c1",
                relation=PaperConceptRelation.USES,
                evidence=(PageEvidence(page=1, quote="retrieval-augmented generation to conference paper search"),),
            ),
        ),
    ),
)

NORMALIZATION = ConceptNormalization(
    schema_version=1,
    concepts=(
        NormalizedConcept(
            id=RAG,
            canonical_name="Retrieval-Augmented Generation",
            type=ConceptType.METHOD,
            aliases=("RAG",),
            merged_from=(LocalConceptRef(paper_id=1, local_id="c1"), LocalConceptRef(paper_id=3, local_id="c1")),
        ),
        NormalizedConcept(
            id=CHUNKING,
            canonical_name="Page-Aligned Chunking",
            type=ConceptType.METHOD,
            merged_from=(LocalConceptRef(paper_id=1, local_id="c2"),),
        ),
        NormalizedConcept(
            id=KNOWLEDGE_GRAPH,
            canonical_name="Knowledge Graph",
            type=ConceptType.METHOD,
            merged_from=(LocalConceptRef(paper_id=2, local_id="c1"),),
        ),
    ),
    concept_relations=(
        GeneralKnowledgeEdge(
            source_id=KNOWLEDGE_GRAPH,
            target_id=RAG,
            relation=ConceptRelationType.USED_FOR,
            rationale="A knowledge graph supplies the structure the pipeline walks.",
        ),
    ),
)

DIMENSION = 4

QUERY_RAG = "検索拡張生成の論文を読みたい"
QUERY_UNRELATED = "量子計算の論文を読みたい"

# The query points along the first axis; the summary of paper 1 points the same way and scores 1.0, the summary of
# paper 3 points between two axes and scores about 0.71, and every other item gets a zero vector and scores 0.
SCRIPT = {
    QUERY_RAG: (1.0, 0.0, 0.0, 0.0),
    SUMMARY_OF_PAPER_1: (1.0, 0.0, 0.0, 0.0),
    SUMMARY_OF_PAPER_3: (1.0, 1.0, 0.0, 0.0),
}

# Three papers hold retrieval-augmented-generation in two of them, a document frequency of 0.67 that the default
# threshold of 0.4 drops, so every test that wants to see the walk has to widen the threshold.
FOLLOW_EVERY_CONCEPT = 1.0


class ScriptedEmbedder:
    """Embeds the texts a test scripted with the vectors it gave them, and every other text with zeros.

    A zero vector scores 0 against everything, so a test picks what a query lands on by scripting only the texts
    it means to be found, and records every batch of queries it was asked to embed.
    """

    def __init__(self, script: Mapping[str, Sequence[float]], dimension: int = DIMENSION) -> None:
        self._script = {text: np.asarray(vector, dtype=np.float32) for text, vector in script.items()}
        self._dimension = dimension
        self.embedded_queries: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return f"scripted-{self._dimension}"

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        zeros = np.zeros(self._dimension, dtype=np.float32)
        return np.stack([self._script.get(text, zeros) for text in texts])

    def embed_queries(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.embedded_queries.append(list(texts))
        return self.embed_documents(texts)


def build_index(embedder: ScriptedEmbedder) -> FakeIndexRepository:
    """Run the real build so that search reads the artifacts a build would have left behind."""
    index = FakeIndexRepository()
    normalization_repository = FakeConceptNormalizationRepository()
    normalization_repository.save(NORMALIZATION)
    BuildIndexUseCase(
        FakePaperRepository(PAPERS),
        FakePaperExtractionRepository(*EXTRACTIONS),
        normalization_repository,
        WhitespaceTokenizer(),
        embedder,
        index,
    ).execute()
    return index


def search(queries: Sequence[str] = (QUERY_RAG,), **overrides: object) -> SearchResult:
    embedder = ScriptedEmbedder(SCRIPT)
    return build_use_case(build_index(embedder), embedder).execute(queries, SearchConfig.model_validate(overrides))


def build_use_case(index: FakeIndexRepository, embedder: ScriptedEmbedder) -> SearchPapersUseCase:
    return SearchPapersUseCase(
        FakePaperRepository(PAPERS), FakePaperExtractionRepository(*EXTRACTIONS), index, embedder
    )


def describe_paths(paths: Sequence[TraversalPath]) -> list[tuple[int, str, tuple[str, ...]]]:
    """Each path as (source paper, concept left by, concepts hopped to); every path ends at the candidate."""
    return [
        (path.source_paper_id, path.source_edge.concept_id, tuple(hop.reached_concept_id for hop in path.hops))
        for path in paths
    ]


def test_the_direct_candidates_carry_the_title_the_summary_and_the_hits_in_the_order_of_the_vector_search() -> None:
    result = search(top_k=2, generic_concept_threshold=FOLLOW_EVERY_CONCEPT)
    assert [candidate.paper_id for candidate in result.direct_candidates] == [1, 3]
    assert [candidate.title for candidate in result.direct_candidates] == [
        "Retrieval-Augmented Generation for Conference Paper Search",
        "Applying Retrieval-Augmented Generation to Paper Search",
    ]
    assert [candidate.summary_ja for candidate in result.direct_candidates] == [
        SUMMARY_OF_PAPER_1,
        SUMMARY_OF_PAPER_3,
    ]
    first, second = result.direct_candidates
    assert [(hit.kind, hit.ref, hit.query) for hit in first.hits] == [(EmbeddedItemKind.SUMMARY, "1", QUERY_RAG)]
    assert [(hit.kind, hit.ref, hit.query) for hit in second.hits] == [(EmbeddedItemKind.SUMMARY, "3", QUERY_RAG)]
    assert first.hits[0].score > second.hits[0].score


def test_the_papers_reached_from_a_direct_candidate_become_graph_candidates_ordered_by_how_often() -> None:
    result = search(top_k=1, generic_concept_threshold=FOLLOW_EVERY_CONCEPT)
    assert [candidate.paper_id for candidate in result.direct_candidates] == [1]
    assert result.direct_candidates[0].paths == ()
    assert [candidate.paper_id for candidate in result.graph_candidates] == [3, 2]
    reached_3, reached_2 = result.graph_candidates
    assert describe_paths(reached_3.paths) == [(1, RAG, ()), (1, CHUNKING, (RAG,))]
    assert describe_paths(reached_2.paths) == [(1, RAG, (KNOWLEDGE_GRAPH,))]
    assert reached_3.title == "Applying Retrieval-Augmented Generation to Paper Search"
    assert reached_3.summary_ja == SUMMARY_OF_PAPER_3
    assert reached_3.hits == ()


def test_a_path_between_two_direct_candidates_is_attached_to_the_candidate_it_reached() -> None:
    result = search(top_k=2, generic_concept_threshold=FOLLOW_EVERY_CONCEPT)
    found_1, found_3 = result.direct_candidates
    assert describe_paths(found_1.paths) == [(3, RAG, ()), (3, RAG, (CHUNKING,))]
    assert describe_paths(found_3.paths) == [(1, RAG, ()), (1, CHUNKING, (RAG,))]
    # Paper 3 is a direct candidate, so the paths that reached it make no second entry of their own.
    assert [candidate.paper_id for candidate in result.graph_candidates] == [2]


def test_no_graph_candidate_is_listed_when_none_is_allowed_while_the_direct_paths_stay() -> None:
    result = search(top_k=2, generic_concept_threshold=FOLLOW_EVERY_CONCEPT, max_graph_candidates=0)
    assert result.graph_candidates == ()
    assert [candidate.paper_id for candidate in result.direct_candidates] == [1, 3]
    assert all(candidate.paths for candidate in result.direct_candidates)


def test_a_concept_most_of_the_corpus_holds_is_left_out_of_the_traversal_at_the_default_threshold() -> None:
    result = search(top_k=1)
    assert result.config.generic_concept_threshold == 0.4
    assert [candidate.paper_id for candidate in result.direct_candidates] == [1]
    assert result.direct_candidates[0].paths == ()
    assert result.graph_candidates == ()


def test_the_queries_and_the_configuration_are_recorded_in_the_result() -> None:
    config = SearchConfig(top_k=2, max_hops=0, max_graph_candidates=3, generic_concept_threshold=0.9)
    embedder = ScriptedEmbedder(SCRIPT)
    result = build_use_case(build_index(embedder), embedder).execute([QUERY_RAG, QUERY_UNRELATED], config)
    assert result.queries == (QUERY_RAG, QUERY_UNRELATED)
    assert result.config == config


def test_the_s3_uri_is_listed_for_a_paper_that_has_one_and_is_none_for_a_paper_that_has_not() -> None:
    result = search(top_k=2, generic_concept_threshold=FOLLOW_EVERY_CONCEPT)
    assert [candidate.s3_uri for candidate in result.direct_candidates] == [S3_URI_OF_PAPER_1, None]


def test_an_embedder_of_another_model_than_the_index_is_reported_before_a_query_is_embedded() -> None:
    index = build_index(ScriptedEmbedder(SCRIPT))
    other_embedder = ScriptedEmbedder(SCRIPT, dimension=DIMENSION + 1)
    with pytest.raises(EmbeddingModelMismatchError) as caught:
        build_use_case(index, other_embedder).execute([QUERY_RAG], SearchConfig())
    assert caught.value.index_model == "scripted-4"
    assert caught.value.embedder_model == "scripted-5"
    assert other_embedder.embedded_queries == []


def test_an_index_that_has_not_been_built_yet_is_reported_as_not_found() -> None:
    with pytest.raises(IndexNotFoundError, match="manifest not found"):
        build_use_case(FakeIndexRepository(), ScriptedEmbedder(SCRIPT)).execute([QUERY_RAG], SearchConfig())


@pytest.mark.parametrize(
    ("queries", "message"),
    [
        ([], "queries must not be empty"),
        (["  "], "queries must contain non-whitespace characters"),
        ([QUERY_RAG, QUERY_RAG], "queries repeats"),
    ],
)
def test_a_query_list_that_cannot_be_searched_is_rejected_before_the_index_is_read(
    queries: list[str], message: str
) -> None:
    embedder = ScriptedEmbedder(SCRIPT)
    # The index repository is empty, so reading it would raise IndexNotFoundError instead of this ValueError.
    with pytest.raises(ValueError, match=message):
        build_use_case(FakeIndexRepository(), embedder).execute(queries, SearchConfig())
    assert embedder.embedded_queries == []
