"""Domain model of rkgk.

The vocabulary and the entities are versioned together: index manifests record `DOMAIN_MODEL_VERSION`,
so an index built with an older model can be detected and rebuilt.
"""

from rkgk.domain.agents import StructuredOutputAgent, StructuredOutputAgentError
from rkgk.domain.models.base import SLUG_PATTERN, Entity, Slug
from rkgk.domain.models.chunk import Chunk
from rkgk.domain.models.concept_normalization import (
    CONCEPT_NORMALIZATION_SCHEMA_VERSION,
    ConceptNormalization,
    ConceptNormalizationIssue,
    ConceptNormalizationRun,
    ConceptNormalizationValidationError,
    GeneralKnowledgeEdge,
    LocalConceptRef,
    MissingPaperExtractionsError,
    NormalizedConcept,
    build_concept_normalization_schema,
)
from rkgk.domain.models.graph import Concept, ConceptEdge, Evidence, PaperConceptEdge
from rkgk.domain.models.paper import (
    MARKER_PATTERN,
    MarkerKind,
    Page,
    PageMarker,
    Paper,
    PaperIndexEntry,
    PaperMeta,
    PaperPreprocessInfo,
    build_paper_dir_name,
    parse_page,
)
from rkgk.domain.models.paper_extraction import (
    EXTRACTION_SCHEMA_VERSION,
    LOCAL_CONCEPT_ID_PATTERN,
    ExtractedConcept,
    ExtractedConceptEdge,
    ExtractedEvidence,
    ExtractedPaperConceptEdge,
    LocalConceptId,
    PaperExtraction,
    PaperExtractionIssue,
    PaperExtractionRun,
    PaperExtractionValidationError,
    build_paper_extraction_schema,
)
from rkgk.domain.models.vocabulary import (
    ConceptRelationSpec,
    ConceptRelationType,
    ConceptType,
    ConceptTypeSpec,
    Origin,
    OriginSpec,
    PaperConceptRelation,
    RelationSpec,
    describe_vocabulary,
    traversable_concept_relations,
    traversable_concept_types,
    traversable_paper_relations,
)
from rkgk.domain.prompts.concept_normalization import build_concept_normalization_prompt
from rkgk.domain.prompts.paper_extraction import build_paper_extraction_prompt
from rkgk.domain.repositories.concept_normalization import (
    ConceptNormalizationArtifactInvalidError,
    ConceptNormalizationArtifactUnreadableError,
    ConceptNormalizationNotFoundError,
    ConceptNormalizationRepository,
    ConceptNormalizationRepositoryError,
)
from rkgk.domain.repositories.paper import (
    PaperArtifactInvalidError,
    PaperArtifactUnreadableError,
    PaperNotFoundError,
    PaperRepository,
    PaperRepositoryError,
)
from rkgk.domain.repositories.paper_extraction import (
    PaperExtractionArtifactInvalidError,
    PaperExtractionArtifactUnreadableError,
    PaperExtractionNotFoundError,
    PaperExtractionRepository,
    PaperExtractionRepositoryError,
)
from rkgk.domain.services.concept_normalization import check_normalization_against_extractions
from rkgk.domain.services.paper_extraction import check_extraction_against_paper, normalize_whitespace
from rkgk.domain.services.validation import format_error_path

DOMAIN_MODEL_VERSION = 1

__all__ = [
    "CONCEPT_NORMALIZATION_SCHEMA_VERSION",
    "DOMAIN_MODEL_VERSION",
    "EXTRACTION_SCHEMA_VERSION",
    "LOCAL_CONCEPT_ID_PATTERN",
    "MARKER_PATTERN",
    "SLUG_PATTERN",
    "Chunk",
    "Concept",
    "ConceptEdge",
    "ConceptNormalization",
    "ConceptNormalizationArtifactInvalidError",
    "ConceptNormalizationArtifactUnreadableError",
    "ConceptNormalizationIssue",
    "ConceptNormalizationNotFoundError",
    "ConceptNormalizationRepository",
    "ConceptNormalizationRepositoryError",
    "ConceptNormalizationRun",
    "ConceptNormalizationValidationError",
    "ConceptRelationSpec",
    "ConceptRelationType",
    "ConceptType",
    "ConceptTypeSpec",
    "Entity",
    "Evidence",
    "ExtractedConcept",
    "ExtractedConceptEdge",
    "ExtractedEvidence",
    "ExtractedPaperConceptEdge",
    "GeneralKnowledgeEdge",
    "LocalConceptId",
    "LocalConceptRef",
    "MarkerKind",
    "MissingPaperExtractionsError",
    "NormalizedConcept",
    "Origin",
    "OriginSpec",
    "Page",
    "PageMarker",
    "Paper",
    "PaperArtifactInvalidError",
    "PaperArtifactUnreadableError",
    "PaperConceptEdge",
    "PaperConceptRelation",
    "PaperExtraction",
    "PaperExtractionArtifactInvalidError",
    "PaperExtractionArtifactUnreadableError",
    "PaperExtractionIssue",
    "PaperExtractionNotFoundError",
    "PaperExtractionRepository",
    "PaperExtractionRepositoryError",
    "PaperExtractionRun",
    "PaperExtractionValidationError",
    "PaperIndexEntry",
    "PaperMeta",
    "PaperNotFoundError",
    "PaperPreprocessInfo",
    "PaperRepository",
    "PaperRepositoryError",
    "RelationSpec",
    "Slug",
    "StructuredOutputAgent",
    "StructuredOutputAgentError",
    "build_concept_normalization_prompt",
    "build_concept_normalization_schema",
    "build_paper_dir_name",
    "build_paper_extraction_prompt",
    "build_paper_extraction_schema",
    "check_extraction_against_paper",
    "check_normalization_against_extractions",
    "describe_vocabulary",
    "format_error_path",
    "normalize_whitespace",
    "parse_page",
    "traversable_concept_relations",
    "traversable_concept_types",
    "traversable_paper_relations",
]
