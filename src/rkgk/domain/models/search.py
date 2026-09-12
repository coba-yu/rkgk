"""What a search returns, fixed before the stages that fill it exist.

Vector search fills the direct candidates: the papers whose chunk, summary or concept a query landed on, each
with every hit that led there.
Graph traversal fills the graph candidates: the papers reached from a direct candidate through a concept it
shares or a concept one relation away, each with every path that reached it.
A paper appears once in the whole result, so a paper found both ways keeps its hits and its paths on one entry.
The queries and the configuration travel with the result so that the JSON can be read on its own.
"""

from typing import Self

from pydantic import Field, field_validator, model_validator

from rkgk.domain.models.base import Entity, Slug
from rkgk.domain.models.embedding import EmbeddedItemKind
from rkgk.domain.models.graph import ConceptEdge, PaperConceptEdge


class SearchConfig(Entity):
    """The settings one search ran with, kept in the result so a reader knows how the candidates were chosen.

    Only the vector search setting exists yet; the traversal settings join it when traversal is implemented.
    """

    top_k: int = Field(default=10, ge=1)


class EmbeddedItemHit(Entity):
    """One embedded item that one query landed on, with the score and the text that matched.

    `kind` and `ref` name the item the way the embedding table does, so the chunk, the summary or the concept
    behind a hit can be looked up without a second search.
    The query is part of the hit because the same item can be hit by several queries, and each of those is a
    separate piece of evidence with its own score.
    """

    kind: EmbeddedItemKind
    ref: str = Field(min_length=1)
    query: str = Field(min_length=1)
    score: float
    text: str

    @field_validator("text")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain non-whitespace characters")
        return value


class PaperHits(Entity):
    """The hits of one paper, gathered by vector search before the paper becomes a candidate.

    Vector search knows the embedding table and the graph but not the titles, summaries or S3 URIs, so it returns
    this and the use case turns it into a `PaperCandidate`.
    """

    paper_id: int = Field(ge=1)
    hits: tuple[EmbeddedItemHit, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_repeated_hits(self) -> Self:
        seen: set[tuple[EmbeddedItemKind, str, str]] = set()
        for hit in self.hits:
            key = (hit.kind, hit.ref, hit.query)
            if key in seen:
                raise ValueError(f"hits repeats {hit.kind.value} {hit.ref!r} for the query {hit.query!r}")
            seen.add(key)
        return self


class ConceptHop(Entity):
    """One concept edge followed on a path, and the concept the path arrived at.

    The edge is kept whole, evidence and rationale included, so a result can quote the paper behind a paper-origin
    hop and mark a general-knowledge hop as unverified without opening the graph again; keeping the edge also keeps
    two hops apart when two papers assert the same relation, because the graph stores those as two edges.
    A path may follow an edge against its direction, so the concept it arrived at is named separately and a reader
    can still say "X is_a Y" the way the graph says it.
    """

    edge: ConceptEdge
    reached_concept_id: Slug

    @model_validator(mode="after")
    def _check_reached_end(self) -> Self:
        if self.reached_concept_id not in (self.edge.source_id, self.edge.target_id):
            raise ValueError(f"reached_concept_id must be {self.edge.source_id!r} or {self.edge.target_id!r}")
        return self

    @property
    def departed_concept_id(self) -> str:
        return self.edge.target_id if self.reached_concept_id == self.edge.source_id else self.edge.source_id


class TraversalPath(Entity):
    """How one paper was reached: source paper -edge-> concept [-edge-> concept]... -edge-> reached paper.

    Every step is a graph edge kept whole, so the paper-to-concept ends carry their quotes too and a reader can
    check each claim of the path against the paper it came from.
    """

    source_edge: PaperConceptEdge
    hops: tuple[ConceptHop, ...] = ()
    target_edge: PaperConceptEdge

    @model_validator(mode="after")
    def _check_edges_form_a_chain(self) -> Self:
        """Each step must leave from the concept the path stands on, so a path cannot skip a node or loop back."""
        if self.source_edge.paper_id == self.target_edge.paper_id:
            raise ValueError(f"a path must not lead from paper {self.source_edge.paper_id} back to itself")
        current = self.source_edge.concept_id
        for position, hop in enumerate(self.hops, start=1):
            if hop.departed_concept_id != current:
                raise ValueError(
                    f"hop {position} joins {hop.edge.source_id!r} and {hop.edge.target_id!r} "
                    f"but the path stands on {current!r}"
                )
            current = hop.reached_concept_id
        if self.target_edge.concept_id != current:
            raise ValueError(
                f"target_edge is attached to {self.target_edge.concept_id!r} but the path stands on {current!r}"
            )
        return self

    @property
    def source_paper_id(self) -> int:
        return self.source_edge.paper_id

    @property
    def reached_paper_id(self) -> int:
        return self.target_edge.paper_id

    @property
    def reached_concept_id(self) -> str:
        """The concept the reached paper is attached to: the last hop's end, or the first concept without hops."""
        return self.target_edge.concept_id


class PaperCandidate(Entity):
    """One paper of the result with everything that led to it.

    A direct candidate holds at least one hit and may also hold paths, because a paper found by vector search is
    still worth explaining through the graph; a graph candidate holds paths only.
    Every path ends at this paper, so the reached paper is not repeated outside the path.
    """

    paper_id: int = Field(ge=1)
    title: str = Field(min_length=1)
    s3_uri: str = Field(min_length=1)
    summary_ja: str = Field(min_length=1)
    hits: tuple[EmbeddedItemHit, ...] = ()
    paths: tuple[TraversalPath, ...] = ()

    @model_validator(mode="after")
    def _check_something_led_here(self) -> Self:
        if not self.hits and not self.paths:
            raise ValueError("a candidate needs at least one hit or one path")
        seen_hits: set[tuple[EmbeddedItemKind, str, str]] = set()
        for hit in self.hits:
            key = (hit.kind, hit.ref, hit.query)
            if key in seen_hits:
                raise ValueError(f"hits repeats {hit.kind.value} {hit.ref!r} for the query {hit.query!r}")
            seen_hits.add(key)
        seen_paths: set[TraversalPath] = set()
        for path in self.paths:
            if path.reached_paper_id != self.paper_id:
                raise ValueError(f"paths must end at the candidate paper {self.paper_id}, not {path.reached_paper_id}")
            if path in seen_paths:
                raise ValueError(f"paths repeats a path from paper {path.source_paper_id}")
            seen_paths.add(path)
        return self


class SearchResult(Entity):
    """What the search command prints: the direct candidates first, the graph candidates after them.

    The order inside each list is the order to read them in, decided by the stage that filled the list, so this
    object keeps it and does not sort.
    """

    queries: tuple[str, ...] = Field(min_length=1)
    config: SearchConfig
    direct_candidates: tuple[PaperCandidate, ...]
    graph_candidates: tuple[PaperCandidate, ...]

    @field_validator("queries")
    @classmethod
    def _reject_blank_or_repeated_queries(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        seen: set[str] = set()
        for query in value:
            if not query.strip():
                raise ValueError("queries must contain non-whitespace characters")
            if query in seen:
                raise ValueError(f"queries repeats {query!r}")
            seen.add(query)
        return value

    @model_validator(mode="after")
    def _check_candidates_against_each_other(self) -> Self:
        """A paper is listed once, hits come from the queries asked, and paths start at direct candidates.

        Paths start at direct candidates because traversal begins from what vector search found; a path from
        anywhere else would be an explanation nobody asked for.
        """
        direct_ids = {candidate.paper_id for candidate in self.direct_candidates}
        listed: set[int] = set()
        for candidate in (*self.direct_candidates, *self.graph_candidates):
            if candidate.paper_id in listed:
                raise ValueError(f"paper {candidate.paper_id} is listed more than once")
            listed.add(candidate.paper_id)
        queries = set(self.queries)
        for candidate in self.direct_candidates:
            if not candidate.hits:
                raise ValueError(f"direct candidate {candidate.paper_id} has no hit")
        for candidate in self.graph_candidates:
            if candidate.hits:
                raise ValueError(f"graph candidate {candidate.paper_id} has hits and belongs to direct_candidates")
            if not candidate.paths:
                raise ValueError(f"graph candidate {candidate.paper_id} has no path")
        for candidate in (*self.direct_candidates, *self.graph_candidates):
            for hit in candidate.hits:
                if hit.query not in queries:
                    raise ValueError(f"candidate {candidate.paper_id} has a hit for the unknown query {hit.query!r}")
            for path in candidate.paths:
                if path.source_paper_id not in direct_ids:
                    raise ValueError(
                        f"candidate {candidate.paper_id} has a path from paper {path.source_paper_id}, "
                        "which is not a direct candidate"
                    )
        return self
