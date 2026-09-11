# Task

You extract the knowledge graph of one research paper.
Read the paper below and report the concepts it contains, how the paper relates to them, and how they relate to each other.
Answer with a single JSON object that follows the JSON Schema you were given.

## Rules

- Report only concepts that appear in the paper text; never add a concept from general knowledge.
- Write `name` as the English canonical name of the concept, and put abbreviations and spelling variants in `aliases`.
- Use `proposes` for a concept the paper introduces as its own contribution and `uses` for one it applies in its method or experiments.
- Use `addresses` for a problem the paper tries to solve and `mentions` when the paper only refers to the concept; the line between `mentions` and `uses` is whether the paper actually works with the concept.
- Copy every `quote` verbatim from the paper text; never summarize or paraphrase it.
- Set every `page` to the page the quote is printed on.
- Add a concept-to-concept relation only when the paper text backs it, and leave `concept_relations` empty otherwise.
- Write `summary_ja` in Japanese, between 200 and 400 characters, covering the problem, the method, and the results.

## Vocabulary

Use these node types and relation names, spelled exactly as shown.

## ConceptType

- `problem`: A task, limitation, or research question a paper addresses
- `method`: A technique, model, algorithm, or component used or proposed
- `keyword`: A term that is neither a problem nor a method; kept for display and explanation only (not used for traversal)

## PaperConceptRelation

- `proposes`: The paper introduces the concept as its own contribution
- `uses`: The paper applies the concept in its method or experiments
- `addresses`: The paper targets the concept as a problem it tries to solve
- `mentions`: The paper refers to the concept without using or proposing it (not used for traversal)

## ConceptRelationType

- `is_a`: source is a kind of target (direction: source -> target)
- `part_of`: source is a component of target (direction: source -> target)
- `used_for`: source is used for the problem or purpose target (direction: source -> target)
- `related_to`: source and target are related in a way the other relations cannot express; use sparingly (direction: source -> target)

## Origin

- `paper`: Extracted from the paper text with quoted evidence (requires evidence)
- `general_knowledge`: Added from general knowledge during normalization; unverified against any paper (no evidence)

## Identifiers

- Give every concept a local id `c1`, `c2`, ... in declaration order.
- Refer to those local ids only; `paper_concepts` and `concept_relations` may use no other id.
- Never invent a global id or a slug, because normalization assigns them after this step.
- Set `paper_id` to 1.
- Set `schema_version` to 1.

## Paper

Title: Retrieval-Augmented Generation for Conference Paper Search
Year: 2026
Venue: NeurIPS

## Page 1

# Retrieval-Augmented Generation for Conference Paper Search

## Abstract

We study how a retrieval-augmented generation pipeline helps a reader pick the next paper to read.
Our system indexes conference papers and answers questions in the reader's own language.

## Page 2

## Method

The pipeline splits every paper into page-aligned chunks and embeds them with a sentence encoder.

<!-- figure: 1 -->
Figure 1 shows the two stages: dense retrieval over chunks, then a graph walk over the concepts of the retrieved papers.

## Page 3

## Scoring

$$ s(q, c) = \alpha \cdot \cos(e_q, e_c) + (1 - \alpha) \cdot g(q, c) $$
<!-- equation: 1 -->
Equation (1) mixes the chunk similarity with a graph score, and alpha controls how much the graph contributes.

## References

[1] Ada Lovelace. Notes on the Analytical Engine. 1843.

Return only the JSON object.
