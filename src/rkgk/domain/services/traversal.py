"""Walks the knowledge graph from the papers vector search found to the papers standing next to them.

A direct candidate explains itself through its hits, and this module explains its neighbours: the papers that hold
a concept it holds, and the papers a concept relation further away.
Every path comes back whole, both paper edges and every hop kept as the graph stores them, so a later stage can
show why a paper was reached without opening the graph again.
Walking and ranking are separate here, so another ranking can be tried without touching the walk.
It runs on the networkx view of the graph and owns no data.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import networkx as nx

from rkgk.domain.models.graph import ConceptEdge, KnowledgeGraph, PaperConceptEdge
from rkgk.domain.models.search import ConceptHop, PaperPaths, SearchConfig, TraversalPath
from rkgk.domain.services.graph_builder import build_paper_node_id, to_networkx


class TraversalStrategy(Protocol):
    """Something that returns every path the graph offers from the papers vector search found.

    A path that reaches another source paper is returned too: the paper is already a candidate, and the path is
    still the explanation of how the two are related.
    """

    def traverse(
        self, graph: KnowledgeGraph, source_paper_ids: Sequence[int], config: SearchConfig
    ) -> tuple[TraversalPath, ...]: ...


class ConceptNeighborhoodTraversal:
    """Walks from each source paper to the concepts it holds, and from there to the papers around them.

    A concept a large share of the corpus holds joins papers that have nothing to do with each other, so a concept
    above `generic_concept_threshold` is neither started from nor hopped to; the same goes for a concept type the
    configuration leaves out.
    Parallel edges each yield their own path, because a paper that both proposes and uses a concept, and two papers
    that assert the same relation, are separate pieces of evidence that the result models keep apart.
    """

    def traverse(
        self, graph: KnowledgeGraph, source_paper_ids: Sequence[int], config: SearchConfig
    ) -> tuple[TraversalPath, ...]:
        """Walk from every source paper in the given order, each paper in the edge order of the graph."""
        view = to_networkx(graph)
        walk = _ConceptWalk(view=view, eligible=_collect_eligible_concepts(view, config), config=config)
        paths: list[TraversalPath] = []
        for paper_id in source_paper_ids:
            node_id = build_paper_node_id(paper_id)
            if node_id not in view:
                raise ValueError(f"paper {paper_id} is not in the graph")
            for _, concept_id, attributes in view.out_edges(node_id, data=True):
                if attributes["relation"] not in config.paper_relations or concept_id not in walk.eligible:
                    continue
                source_edge = _rebuild_paper_concept_edge(paper_id, concept_id, attributes)
                paths.extend(walk.walk_from_concept(source_edge, concept_id, (), frozenset({concept_id}), depth=0))
        return tuple(paths)


@dataclass(frozen=True)
class _ConceptWalk:
    """What stays the same while one traversal runs: the graph view, the concepts it may stand on, the settings."""

    view: nx.MultiDiGraph
    eligible: frozenset[str]
    config: SearchConfig

    def walk_from_concept(
        self,
        source_edge: PaperConceptEdge,
        standing_on: str,
        hops: tuple[ConceptHop, ...],
        visited: frozenset[str],
        depth: int,
    ) -> list[TraversalPath]:
        """Take the papers on this concept, then walk one concept further while hops are left."""
        paths = [
            TraversalPath(source_edge=source_edge, hops=hops, target_edge=target_edge)
            for target_edge in self.find_target_edges(standing_on, source_edge.paper_id)
        ]
        if depth >= self.config.max_hops:
            return paths
        for hop in self.find_hops(standing_on, visited):
            paths.extend(
                self.walk_from_concept(
                    source_edge,
                    hop.reached_concept_id,
                    (*hops, hop),
                    visited | {hop.reached_concept_id},
                    depth=depth + 1,
                )
            )
        return paths

    def find_target_edges(self, concept_id: str, source_paper_id: int) -> list[PaperConceptEdge]:
        """Return the edges by which a paper other than the source paper of the path holds this concept."""
        edges: list[PaperConceptEdge] = []
        for node_id, _, attributes in self.view.in_edges(concept_id, data=True):
            # A concept is pointed at by papers and by other concepts; only a paper ends a path.
            if self.view.nodes[node_id]["kind"] != "paper":
                continue
            paper_id = self.view.nodes[node_id]["paper_id"]
            if paper_id == source_paper_id or attributes["relation"] not in self.config.paper_relations:
                continue
            edges.append(_rebuild_paper_concept_edge(paper_id, concept_id, attributes))
        return edges

    def find_hops(self, concept_id: str, visited: frozenset[str]) -> list[ConceptHop]:
        """Return the hops leaving this concept, a concept relation followed forward first, then backward.

        A relation is followed against its direction too, because "X is_a Y" relates the two papers whichever end
        the path came in by; the hop names the concept it arrived at, so the relation can still be read as stated.
        """
        # TODO: Following is_a / part_of backward walks toward the generalization, which spreads to papers on the
        # broad concept; a per-direction setting in SearchConfig may be needed so that only one direction is followed.
        hops: list[ConceptHop] = []
        # Nothing but a concept relation leaves a concept: the view draws a paper edge from the paper to the concept.
        for _, reached_id, attributes in self.view.out_edges(concept_id, data=True):
            if self.accepts_hop(reached_id, attributes, visited):
                edge = _rebuild_concept_edge(concept_id, reached_id, attributes)
                hops.append(ConceptHop(edge=edge, reached_concept_id=reached_id))
        for reached_id, _, attributes in self.view.in_edges(concept_id, data=True):
            if self.view.nodes[reached_id]["kind"] != "concept":
                continue
            if self.accepts_hop(reached_id, attributes, visited):
                edge = _rebuild_concept_edge(reached_id, concept_id, attributes)
                hops.append(ConceptHop(edge=edge, reached_concept_id=reached_id))
        return hops

    def accepts_hop(self, reached_id: str, attributes: Mapping[str, Any], visited: frozenset[str]) -> bool:
        """A hop follows a listed relation to an eligible concept the path does not already stand on."""
        return (
            attributes["relation"] in self.config.concept_relations
            and reached_id in self.eligible
            and reached_id not in visited
        )


def _collect_eligible_concepts(view: nx.MultiDiGraph, config: SearchConfig) -> frozenset[str]:
    """Return the concepts a path may stand on: a listed type, held by few enough papers."""
    return frozenset(
        node_id
        for node_id, attributes in view.nodes(data=True)
        if attributes["kind"] == "concept"
        and attributes["type"] in config.concept_types
        and attributes["document_frequency"] <= config.generic_concept_threshold
    )


def _rebuild_paper_concept_edge(paper_id: int, concept_id: str, attributes: Mapping[str, Any]) -> PaperConceptEdge:
    """Read one paper-to-concept edge of the view back into the model the path carries."""
    return PaperConceptEdge(
        paper_id=paper_id,
        concept_id=concept_id,
        relation=attributes["relation"],
        evidence=attributes["evidence"],
    )


def _rebuild_concept_edge(source_id: str, target_id: str, attributes: Mapping[str, Any]) -> ConceptEdge:
    """Read one concept-to-concept edge of the view back into the model a hop carries."""
    return ConceptEdge(
        source_id=source_id,
        target_id=target_id,
        relation=attributes["relation"],
        origin=attributes["origin"],
        paper_id=attributes["paper_id"],
        evidence=attributes["evidence"],
        rationale=attributes["rationale"],
    )


def rank_reached_papers(paths: Sequence[TraversalPath]) -> tuple[PaperPaths, ...]:
    """Group the paths by the paper they reached, the most reached paper first, a tie going to the lower paper id.

    How many paths reach a paper is how many ways the graph relates it to what the query found, which is the whole
    evidence traversal has; the order inside a paper is left as the walk produced it.
    """
    paths_by_paper: dict[int, list[TraversalPath]] = {}
    for path in paths:
        paths_by_paper.setdefault(path.reached_paper_id, []).append(path)
    ranked = sorted(paths_by_paper.items(), key=lambda entry: (-len(entry[1]), entry[0]))
    return tuple(PaperPaths(paper_id=paper_id, paths=tuple(reached)) for paper_id, reached in ranked)


def select_graph_candidates(
    ranked: Sequence[PaperPaths], direct_paper_ids: Collection[int], limit: int
) -> tuple[PaperPaths, ...]:
    """Keep the first `limit` reached papers that vector search did not already find.

    A paper found both ways is listed once in the result, so its paths belong on its direct candidate and are
    attached there by the use case instead of making a second entry here.
    """
    direct = set(direct_paper_ids)
    return tuple(entry for entry in ranked if entry.paper_id not in direct)[:limit]
