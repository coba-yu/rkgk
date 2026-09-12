from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from rkgk.domain.models.concept_normalization import (
    ConceptNormalization,
    ConceptNormalizationValidationError,
    GeneralKnowledgeEdge,
    LocalConceptRef,
    MissingPaperExtractionsError,
    NormalizedConcept,
)
from rkgk.domain.models.graph import EvidenceResolutionError
from rkgk.domain.models.paper import Paper
from rkgk.domain.models.paper_extraction import (
    ExtractedConcept,
    ExtractedConceptEdge,
    ExtractedPaperConceptEdge,
    PageEvidence,
    PaperExtraction,
    PaperExtractionMismatchError,
)
from rkgk.domain.models.vocabulary import ConceptRelationType, ConceptType, PaperConceptRelation
from rkgk.domain.repositories.concept_normalization import ConceptNormalizationNotFoundError
from rkgk.domain.services.chunking import DEFAULT_MAX_TOKENS
from rkgk.infrastructure.fake_embedder import FakeEmbedder
from rkgk.infrastructure.whitespace_tokenizer import WhitespaceTokenizer
from rkgk.usecase.build_index import BuildIndexUseCase
from tests.usecase.fakes import (
    FakeConceptNormalizationRepository,
    FakeIndexRepository,
    FakePaperExtractionRepository,
    FakePaperRepository,
    build_paper,
)

PAPERS = {
    1: build_paper(
        1,
        "We study a retrieval-augmented generation pipeline.\n",
        "The pipeline splits every paper into page-aligned chunks.\n",
    ),
    2: build_paper(2, "We use a knowledge graph as a retrieval backbone.\n"),
}

EXTRACTION_1 = PaperExtraction(
    schema_version=1,
    paper_id=1,
    summary_ja="この論文は検索拡張生成のパイプラインを提案する。",
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
)

EXTRACTION_2 = PaperExtraction(
    schema_version=1,
    paper_id=2,
    summary_ja="この論文は知識グラフを検索の骨格として使う。",
    concepts=(ExtractedConcept(local_id="c1", name="Knowledge Graph", type=ConceptType.METHOD),),
    paper_concepts=(
        ExtractedPaperConceptEdge(
            concept_id="c1",
            relation=PaperConceptRelation.USES,
            evidence=(PageEvidence(page=1, quote="knowledge graph as a retrieval backbone"),),
        ),
    ),
)

EXTRACTIONS = (EXTRACTION_1, EXTRACTION_2)

NORMALIZATION = ConceptNormalization(
    schema_version=1,
    concepts=(
        NormalizedConcept(
            id="retrieval-augmented-generation",
            canonical_name="Retrieval-Augmented Generation",
            type=ConceptType.METHOD,
            aliases=("RAG",),
            merged_from=(LocalConceptRef(paper_id=1, local_id="c1"),),
        ),
        NormalizedConcept(
            id="page-aligned-chunking",
            canonical_name="Page-Aligned Chunking",
            type=ConceptType.METHOD,
            merged_from=(LocalConceptRef(paper_id=1, local_id="c2"),),
        ),
        NormalizedConcept(
            id="knowledge-graph",
            canonical_name="Knowledge Graph",
            type=ConceptType.METHOD,
            merged_from=(LocalConceptRef(paper_id=2, local_id="c1"),),
        ),
    ),
    concept_relations=(
        GeneralKnowledgeEdge(
            source_id="knowledge-graph",
            target_id="retrieval-augmented-generation",
            relation=ConceptRelationType.USED_FOR,
            rationale="A knowledge graph supplies the structure the pipeline walks.",
        ),
    ),
)


class RecordingEmbedder:
    """Embeds like the fake embedder and keeps every batch of texts it was asked for, in order."""

    def __init__(self) -> None:
        self._embedder = FakeEmbedder()
        self.documents: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return self._embedder.model_name

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.documents.append(list(texts))
        return self._embedder.embed_documents(texts)

    def embed_queries(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return self._embedder.embed_queries(texts)


def build_use_case(
    index: FakeIndexRepository,
    embedder: RecordingEmbedder,
    papers: dict[int, Paper] = PAPERS,
    extractions: tuple[PaperExtraction, ...] = EXTRACTIONS,
    normalization: ConceptNormalization | None = NORMALIZATION,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> BuildIndexUseCase:
    normalization_repository = FakeConceptNormalizationRepository()
    if normalization is not None:
        normalization_repository.save(normalization)
    return BuildIndexUseCase(
        FakePaperRepository(papers),
        FakePaperExtractionRepository(*extractions),
        normalization_repository,
        WhitespaceTokenizer(),
        embedder,
        index,
        max_tokens=max_tokens,
    )


def build_mismatched_extraction() -> PaperExtraction:
    payload = EXTRACTION_2.model_dump()
    payload["paper_concepts"][0]["evidence"][0]["quote"] = "a sentence the paper never wrote"
    return PaperExtraction.model_validate(payload)


def test_the_build_saves_the_chunks_the_embeddings_the_graph_and_the_manifest() -> None:
    index = FakeIndexRepository()
    result = build_use_case(index, RecordingEmbedder()).execute()
    assert index.chunks == result.chunks
    assert index.table == result.embeddings
    assert index.graph == result.graph
    assert index.manifest == result.manifest
    assert [chunk.id for chunk in result.chunks] == ["1:0", "1:1", "2:0"]
    assert len(result.graph.concepts) == 3
    assert len(result.graph.paper_concepts) == 3
    assert len(result.graph.concept_relations) == 2


def test_the_manifest_records_the_model_the_dimension_the_budget_and_the_papers() -> None:
    index = FakeIndexRepository()
    manifest = build_use_case(index, RecordingEmbedder(), max_tokens=64).execute().manifest
    assert manifest.embedding_model == "fake-8"
    assert manifest.embedding_dimension == 8
    assert manifest.chunk_max_tokens == 64
    assert manifest.paper_ids == (1, 2)
    assert manifest.domain_model_version == 1


def test_the_manifest_is_saved_after_the_artifacts_it_describes() -> None:
    index = FakeIndexRepository()
    build_use_case(index, RecordingEmbedder()).execute()
    assert index.saved == ["chunks", "embeddings", "graph", "manifest"]


def test_the_embedder_is_asked_for_the_chunks_then_the_summaries_then_the_concepts() -> None:
    embedder = RecordingEmbedder()
    result = build_use_case(FakeIndexRepository(), embedder).execute()
    assert len(embedder.documents) == 1
    assert embedder.documents[0] == [
        *(chunk.text for chunk in result.chunks),
        "この論文は検索拡張生成のパイプラインを提案する。",
        "この論文は知識グラフを検索の骨格として使う。",
        "Retrieval-Augmented Generation (RAG)",
        "Page-Aligned Chunking",
        "Knowledge Graph",
    ]


def test_every_paper_without_an_extraction_is_reported_before_anything_is_embedded() -> None:
    embedder = RecordingEmbedder()
    index = FakeIndexRepository()
    with pytest.raises(MissingPaperExtractionsError, match="2") as caught:
        build_use_case(index, embedder, extractions=(EXTRACTION_1,)).execute()
    assert caught.value.paper_ids == (2,)
    assert embedder.documents == []
    assert index.saved == []


def test_an_extraction_whose_quote_is_no_longer_in_the_paper_is_reported_with_its_paper() -> None:
    index = FakeIndexRepository()
    with pytest.raises(PaperExtractionMismatchError, match="paper 2: paper_concepts\\[0\\]") as caught:
        build_use_case(index, RecordingEmbedder(), extractions=(EXTRACTION_1, build_mismatched_extraction())).execute()
    assert list(caught.value.issues_by_paper) == [2]
    assert index.saved == []


def test_a_data_directory_without_a_normalization_is_reported() -> None:
    with pytest.raises(ConceptNormalizationNotFoundError):
        build_use_case(FakeIndexRepository(), RecordingEmbedder(), normalization=None).execute()


def test_a_normalization_that_misses_a_paper_is_reported_as_invalid() -> None:
    partial = ConceptNormalization(schema_version=1, concepts=NORMALIZATION.concepts[:2], concept_relations=())
    with pytest.raises(ConceptNormalizationValidationError, match="paper 2 'c1' is in no merged_from"):
        build_use_case(FakeIndexRepository(), RecordingEmbedder(), normalization=partial).execute()


def test_quotes_that_no_single_chunk_holds_are_reported_for_every_paper_at_once() -> None:
    index = FakeIndexRepository()
    with pytest.raises(EvidenceResolutionError) as caught:
        build_use_case(index, RecordingEmbedder(), max_tokens=2).execute()
    assert {item.paper_id for item in caught.value.unresolved} == {1, 2}
    assert index.saved == []


def test_an_index_without_papers_is_reported_before_anything_is_read() -> None:
    with pytest.raises(ValueError, match="no papers in the index"):
        build_use_case(FakeIndexRepository(), RecordingEmbedder(), papers={}).execute()


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_fewer_than_one_token_per_chunk_is_rejected_before_anything_runs(max_tokens: int) -> None:
    with pytest.raises(ValueError, match="max_tokens must be at least 1"):
        build_use_case(FakeIndexRepository(), RecordingEmbedder(), max_tokens=max_tokens)
