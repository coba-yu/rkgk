from rkgk.domain.models.concept_normalization import (
    ConceptGrouping,
    GroupedConcept,
    LocalConceptRef,
    NormalizedConcept,
)
from rkgk.domain.models.paper_extraction import ExtractedConcept, PaperExtraction
from rkgk.domain.models.vocabulary import ConceptType
from rkgk.domain.services.concept_merging import (
    build_collision_issues,
    build_merge_from_groups,
    collect_groups,
    collect_solo_concepts,
    combine_colliding_groups,
    combine_groups,
    derive_solo_concept,
    find_slug_collisions,
)

EXTRACTIONS = (
    PaperExtraction(
        schema_version=1,
        paper_id=1,
        summary_ja="この論文は検索拡張生成のパイプラインを提案する。",
        concepts=(
            ExtractedConcept(
                local_id="c1",
                name="Retrieval-Augmented Generation",
                type=ConceptType.METHOD,
                aliases=("RAG",),
                description="Generation grounded in retrieved passages.",
            ),
            ExtractedConcept(local_id="c2", name="Page-Aligned Chunking", type=ConceptType.METHOD),
        ),
        paper_concepts=(),
    ),
    PaperExtraction(
        schema_version=1,
        paper_id=2,
        summary_ja="この論文は知識グラフを検索の骨格として使う。",
        concepts=(
            ExtractedConcept(local_id="c1", name="RAG", type=ConceptType.METHOD),
            ExtractedConcept(local_id="c2", name="Knowledge Graph", type=ConceptType.METHOD),
        ),
        paper_concepts=(),
    ),
)

GROUPING = ConceptGrouping(
    groups=((LocalConceptRef(paper_id=1, local_id="c1"), LocalConceptRef(paper_id=2, local_id="c1")),)
)


def build_concept(slug: str, paper_id: int, local_id: str) -> NormalizedConcept:
    return NormalizedConcept(
        id=slug,
        canonical_name=slug.replace("-", " ").title(),
        type=ConceptType.METHOD,
        merged_from=(LocalConceptRef(paper_id=paper_id, local_id=local_id),),
    )


def describe(group: tuple[GroupedConcept, ...]) -> list[str]:
    return [f"{item.paper_id}:{item.concept.local_id}" for item in group]


def test_a_group_of_references_is_looked_up_as_the_concepts_the_papers_extracted() -> None:
    groups = collect_groups(GROUPING, EXTRACTIONS)
    assert len(groups) == 1
    assert describe(groups[0]) == ["1:c1", "2:c1"]
    assert groups[0][0].concept.aliases == ("RAG",)


def test_a_grouping_without_groups_yields_no_group() -> None:
    assert collect_groups(ConceptGrouping(), EXTRACTIONS) == ()


def test_the_concepts_no_group_holds_come_out_in_the_order_the_extractions_declare_them() -> None:
    assert describe(collect_solo_concepts(GROUPING, EXTRACTIONS)) == ["1:c2", "2:c2"]


def test_every_concept_is_solo_when_the_grouping_bundled_nothing() -> None:
    assert describe(collect_solo_concepts(ConceptGrouping(), EXTRACTIONS)) == ["1:c1", "1:c2", "2:c1", "2:c2"]


def test_a_solo_concept_keeps_its_name_aliases_description_and_type() -> None:
    item = GroupedConcept(paper_id=1, concept=EXTRACTIONS[0].concepts[0])
    concept = derive_solo_concept(item)
    assert concept.id == "retrieval-augmented-generation"
    assert concept.canonical_name == "Retrieval-Augmented Generation"
    assert concept.type == ConceptType.METHOD
    assert concept.aliases == ("RAG",)
    assert concept.description == "Generation grounded in retrieved passages."
    assert concept.merged_from == (LocalConceptRef(paper_id=1, local_id="c1"),)


def test_a_solo_concept_whose_name_leaves_no_slug_is_named_after_where_it_was_extracted() -> None:
    item = GroupedConcept(
        paper_id=3, concept=ExtractedConcept(local_id="c7", name="検索拡張生成", type=ConceptType.METHOD)
    )
    concept = derive_solo_concept(item)
    assert concept.id == "concept-3-c7"
    assert concept.canonical_name == "検索拡張生成"


def test_groups_are_laid_end_to_end_in_the_order_they_were_given() -> None:
    groups = collect_groups(GROUPING, EXTRACTIONS)
    solos = collect_solo_concepts(GROUPING, EXTRACTIONS)
    assert describe(combine_groups([groups[0], (solos[0],)])) == ["1:c1", "2:c1", "1:c2"]


def test_the_concepts_of_every_group_become_one_merge_in_group_order() -> None:
    merge = build_merge_from_groups([(build_concept("rag", 1, "c1"),), (build_concept("knowledge-graph", 2, "c2"),)])
    assert [concept.id for concept in merge.concepts] == ["rag", "knowledge-graph"]


def test_groups_that_declare_different_slugs_do_not_collide() -> None:
    assert find_slug_collisions([(build_concept("rag", 1, "c1"),), (build_concept("knowledge-graph", 2, "c2"),)]) == {}


def test_a_slug_two_groups_declare_is_reported_with_both_groups() -> None:
    collisions = find_slug_collisions(
        [
            (build_concept("rag", 1, "c1"),),
            (build_concept("knowledge-graph", 2, "c2"),),
            (build_concept("rag", 2, "c1"),),
        ]
    )
    assert collisions == {"rag": (0, 2)}


def test_a_slug_one_group_declares_twice_is_not_a_collision() -> None:
    # `ConceptMerge` already refuses it, and joining a group with itself would leave the merge unchanged.
    assert find_slug_collisions([(build_concept("rag", 1, "c1"), build_concept("rag", 2, "c1"))]) == {}


def test_the_colliding_groups_are_joined_into_one_that_has_still_to_be_merged() -> None:
    groups = (
        (GroupedConcept(paper_id=1, concept=EXTRACTIONS[0].concepts[0]),),
        (GroupedConcept(paper_id=1, concept=EXTRACTIONS[0].concepts[1]),),
        (GroupedConcept(paper_id=2, concept=EXTRACTIONS[1].concepts[0]),),
    )
    concepts = (
        (build_concept("rag", 1, "c1"),),
        (build_concept("page-aligned-chunking", 1, "c2"),),
        (build_concept("rag", 2, "c1"),),
    )
    joined_groups, joined_concepts = combine_colliding_groups(groups, concepts)
    assert [describe(group) for group in joined_groups] == [["1:c2"], ["1:c1", "2:c1"]]
    assert [len(entry) for entry in joined_concepts] == [1, 0]


def test_two_collisions_that_share_a_group_are_joined_into_a_single_group() -> None:
    groups = (
        (GroupedConcept(paper_id=1, concept=EXTRACTIONS[0].concepts[0]),),
        (GroupedConcept(paper_id=1, concept=EXTRACTIONS[0].concepts[1]),),
        (GroupedConcept(paper_id=2, concept=EXTRACTIONS[1].concepts[0]),),
    )
    concepts = (
        (build_concept("rag", 1, "c1"),),
        (build_concept("rag", 1, "c2"), build_concept("chunking", 1, "c2")),
        (build_concept("chunking", 2, "c1"),),
    )
    joined_groups, joined_concepts = combine_colliding_groups(groups, concepts)
    assert [describe(group) for group in joined_groups] == [["1:c1", "1:c2", "2:c1"]]
    assert joined_concepts == ((),)


def test_nothing_moves_when_no_slug_collides() -> None:
    groups = ((GroupedConcept(paper_id=1, concept=EXTRACTIONS[0].concepts[0]),),)
    concepts = ((build_concept("rag", 1, "c1"),),)
    assert combine_colliding_groups(groups, concepts) == (groups, concepts)


def test_a_collision_is_reported_with_the_slug_and_the_concepts_of_each_group() -> None:
    groups = (
        (GroupedConcept(paper_id=1, concept=EXTRACTIONS[0].concepts[0]),),
        (GroupedConcept(paper_id=2, concept=EXTRACTIONS[1].concepts[0]),),
    )
    issues = build_collision_issues({"rag": (0, 1)}, groups)
    assert [issue.path for issue in issues] == ["concepts"]
    assert issues[0].message == (
        "'rag' is the id of a concept in 2 groups that were merged apart: paper 1 'c1'; paper 2 'c1'"
    )
