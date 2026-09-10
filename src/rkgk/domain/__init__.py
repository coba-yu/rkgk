"""Domain model of rkgk.

The vocabulary and the entities are versioned together: index manifests record `DOMAIN_MODEL_VERSION`,
so an index built with an older model can be detected and rebuilt.
"""

from rkgk.domain.entities import (
    SLUG_PATTERN,
    Chunk,
    Concept,
    ConceptEdge,
    Entity,
    Evidence,
    PaperConceptEdge,
    PaperMeta,
    PaperPreprocessInfo,
    Slug,
    paper_dir_name,
)
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
    "Origin",
    "OriginSpec",
    "PaperConceptEdge",
    "PaperConceptRelation",
    "PaperMeta",
    "PaperPreprocessInfo",
    "RelationSpec",
    "Slug",
    "describe_vocabulary",
    "paper_dir_name",
    "traversable_concept_relations",
    "traversable_concept_types",
    "traversable_paper_relations",
]
