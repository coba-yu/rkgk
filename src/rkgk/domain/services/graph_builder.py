"""Joins the extractions and the normalization into the one knowledge graph the index stores.

An extraction speaks in ids that are local to one paper, and a normalization says which of those local concepts
became which global slug, so neither artifact is a graph on its own.
This module replaces every local id by its slug, attaches the resolved evidence to every edge, and appends the
relations the normalization added from general knowledge.
It reads the extraction, normalization and graph models and owns no data.
"""

from collections.abc import Mapping, Sequence

import networkx as nx

from rkgk.domain.models.concept_normalization import ConceptNormalization
from rkgk.domain.models.graph import Concept, ConceptEdge, Evidence, KnowledgeGraph, PaperConceptEdge
from rkgk.domain.models.paper_extraction import ExtractedEvidence, PaperExtraction
from rkgk.domain.models.vocabulary import ConceptRelationType, Origin, PaperConceptRelation


def paper_node_id(paper_id: int) -> str:
    """Render a paper id as the id of its node in the networkx view."""
    # Papers and concepts share one node namespace, and a slug cannot contain ':' (SLUG_PATTERN in
    # models/base.py), so this prefix can never collide with a concept id.
    return f"paper:{paper_id}"


def _slug_for(slug_of: Mapping[tuple[int, str], str], paper_id: int, local_id: str) -> str:
    """Return the global slug the local concept of that paper was merged into."""
    slug = slug_of.get((paper_id, local_id))
    if slug is None:
        raise ValueError(f"paper {paper_id} concept {local_id!r} is in no normalized concept")
    return slug


def _resolved_evidence_of(
    resolved_evidence: Mapping[int, Mapping[ExtractedEvidence, Evidence]],
    paper_id: int,
    extracted: Sequence[ExtractedEvidence],
) -> tuple[Evidence, ...]:
    """Return the resolved evidence, the one carrying a chunk id, of every quote of one edge."""
    of_paper = resolved_evidence.get(paper_id, {})
    items: list[Evidence] = []
    for item in extracted:
        resolved = of_paper.get(item)
        if resolved is None:
            raise ValueError(f"paper {paper_id} page {item.page}: quote {item.quote!r} has no resolved evidence")
        items.append(resolved)
    return tuple(items)


def _union(existing: tuple[Evidence, ...], added: tuple[Evidence, ...]) -> tuple[Evidence, ...]:
    """Append the evidence that is not there yet, keeping the order of first occurrence."""
    merged = list(existing)
    for item in added:
        if item not in merged:
            merged.append(item)
    return tuple(merged)


def build_knowledge_graph(
    extractions: Sequence[PaperExtraction],
    normalization: ConceptNormalization,
    resolved_evidence: Mapping[int, Mapping[ExtractedEvidence, Evidence]],
) -> KnowledgeGraph:
    """Build the graph of every extracted paper, addressed by the slugs of the normalization.

    `resolved_evidence` is keyed by paper id and each value is what `resolve_extraction_evidence` returned for
    that paper, so the quotes of an edge become evidence that points at a chunk.
    The normalization is taken as it is: the build use case checks it against the extractions before calling
    here, and repeating the check would only report the same problems twice.
    """
    slug_of = {
        (ref.paper_id, ref.local_id): concept.id for concept in normalization.concepts for ref in concept.merged_from
    }

    concepts = tuple(
        Concept(
            id=concept.id,
            canonical_name=concept.canonical_name,
            type=concept.type,
            aliases=concept.aliases,
            description=concept.description,
            # Distinct papers, not merged local concepts: one paper naming a concept twice must not make it look
            # twice as common as it is.
            paper_count=len({ref.paper_id for ref in concept.merged_from}),
        )
        for concept in normalization.concepts
    )

    paper_edges: dict[tuple[int, str, PaperConceptRelation], tuple[Evidence, ...]] = {}
    concept_edges: dict[tuple[str, str, ConceptRelationType, int], tuple[Evidence, ...]] = {}
    for extraction in extractions:
        paper_id = extraction.paper_id
        for edge in extraction.paper_concepts:
            key = (paper_id, _slug_for(slug_of, paper_id, edge.concept_id), edge.relation)
            # Two local concepts of one paper can share a slug, and `KnowledgeGraph` rejects a repeated
            # (paper, concept, relation) edge; both sets of quotes justify the one relation, so they merge.
            paper_edges[key] = _union(
                paper_edges.get(key, ()), _resolved_evidence_of(resolved_evidence, paper_id, edge.evidence)
            )
        for relation in extraction.concept_relations:
            source_id = _slug_for(slug_of, paper_id, relation.source_id)
            target_id = _slug_for(slug_of, paper_id, relation.target_id)
            if source_id == target_id:
                # Not an error: the normalizer merged both ends on purpose, so the relation now says a concept
                # relates to itself and carries nothing; raising would block the whole build over that choice.
                continue
            relation_key = (source_id, target_id, relation.relation, paper_id)
            concept_edges[relation_key] = _union(
                concept_edges.get(relation_key, ()),
                _resolved_evidence_of(resolved_evidence, paper_id, relation.evidence),
            )

    paper_concepts = tuple(
        PaperConceptEdge(paper_id=paper_id, concept_id=concept_id, relation=relation, evidence=evidence)
        for (paper_id, concept_id, relation), evidence in paper_edges.items()
    )
    concept_relations = [
        ConceptEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            origin=Origin.PAPER,
            paper_id=paper_id,
            evidence=evidence,
        )
        for (source_id, target_id, relation, paper_id), evidence in concept_edges.items()
    ]
    concept_relations.extend(
        ConceptEdge(
            source_id=edge.source_id,
            target_id=edge.target_id,
            relation=edge.relation,
            origin=Origin.GENERAL_KNOWLEDGE,
            rationale=edge.rationale,
        )
        for edge in normalization.concept_relations
    )

    return KnowledgeGraph(
        paper_ids=tuple(extraction.paper_id for extraction in extractions),
        concepts=concepts,
        paper_concepts=paper_concepts,
        concept_relations=tuple(concept_relations),
    )


def to_networkx(graph: KnowledgeGraph) -> nx.MultiDiGraph:
    """Render the stored graph as the directed multigraph that traversal runs on.

    A multigraph, because one pair of nodes can be joined more than once: a paper may both propose and use the
    same concept, two papers may state the same relation between two concepts, and general knowledge may state
    it again without any paper behind it.
    Nodes and edges keep the domain objects, the enum members and the evidence, as attributes: this view is
    rebuilt in memory from `KnowledgeGraph`, which is what gets stored, so nothing here has to be serializable.
    """
    view: nx.MultiDiGraph = nx.MultiDiGraph()
    for paper_id in graph.paper_ids:
        view.add_node(paper_node_id(paper_id), kind="paper", paper_id=paper_id)
    for concept in graph.concepts:
        view.add_node(
            concept.id,
            kind="concept",
            canonical_name=concept.canonical_name,
            type=concept.type,
            paper_count=concept.paper_count,
            document_frequency=graph.document_frequency(concept.id),
        )
    for edge in graph.paper_concepts:
        view.add_edge(
            paper_node_id(edge.paper_id),
            edge.concept_id,
            key=edge.relation.value,
            relation=edge.relation,
            evidence=edge.evidence,
        )
    for relation in graph.concept_relations:
        # The key spells out what makes the edge unique, the tuple `KnowledgeGraph` validates, so two relations
        # of the same name from different papers or origins stay two parallel edges.
        key = f"{relation.relation.value}:{relation.origin.value}"
        if relation.paper_id is not None:
            key = f"{key}:{relation.paper_id}"
        view.add_edge(
            relation.source_id,
            relation.target_id,
            key=key,
            relation=relation.relation,
            origin=relation.origin,
            paper_id=relation.paper_id,
            evidence=relation.evidence,
            rationale=relation.rationale,
        )
    return view
