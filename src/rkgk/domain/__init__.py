"""Domain model of rkgk.

The vocabulary and the entities are versioned together: index manifests record `DOMAIN_MODEL_VERSION`,
so an index built with an older model can be detected and rebuilt.
"""

from rkgk.domain.base import SLUG_PATTERN, Entity, Slug
from rkgk.domain.chunk import Chunk
from rkgk.domain.graph import Concept, ConceptEdge, Evidence, PaperConceptEdge
from rkgk.domain.paper import (
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
from rkgk.domain.repositories import PaperRepository, PaperRepositoryError
from rkgk.domain.vocabulary import (
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

DOMAIN_MODEL_VERSION = 1

__all__ = [
    "DOMAIN_MODEL_VERSION",
    "MARKER_PATTERN",
    "SLUG_PATTERN",
    "Chunk",
    "Concept",
    "ConceptEdge",
    "ConceptRelationSpec",
    "ConceptRelationType",
    "ConceptType",
    "ConceptTypeSpec",
    "Entity",
    "Evidence",
    "MarkerKind",
    "Origin",
    "OriginSpec",
    "Page",
    "PageMarker",
    "Paper",
    "PaperConceptEdge",
    "PaperConceptRelation",
    "PaperIndexEntry",
    "PaperMeta",
    "PaperPreprocessInfo",
    "PaperRepository",
    "PaperRepositoryError",
    "RelationSpec",
    "Slug",
    "build_paper_dir_name",
    "describe_vocabulary",
    "parse_page",
    "traversable_concept_relations",
    "traversable_concept_types",
    "traversable_paper_relations",
]
