# Task

You merge the concepts extracted from several research papers into one shared vocabulary.
Read the concepts below, report each of them once as a normalized concept, and say how those normalized concepts relate to each other.
Answer with a single JSON object that follows the JSON Schema you were given.

## Rules

- Merge two concepts only when they mean the same thing, and keep concepts whose meaning you cannot tell apart as separate normalized concepts.
- Write `canonical_name` as the English canonical name of the concept, and derive `id` from it as its lowercase words joined by `-`, matching `^[a-z0-9]+(-[a-z0-9]+)*$`.
- Collect every spelling, abbreviation and Japanese name of the merged concepts in `aliases`.
- List every extracted concept in exactly one `merged_from`, as the paper id and the local id it was extracted under.
- Keep `type` the type the merged concepts were extracted with.
- Never add a concept that no paper extracted.
- Add concept relations from general knowledge between normalized concepts only, and name no other id.
- Prefer `is_a`, `part_of` and `used_for` for those relations, and use `related_to` only when none of the other three fits.
- Give every relation a `rationale` of one sentence saying why it holds, because a general-knowledge relation carries no evidence.
- Set `schema_version` to 1.

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

## Papers

Each paper was extracted on its own, so `c1`, `c2`, ... are local to the paper they are listed under.
A concept line reads `- local id | name | type | aliases: ... | description`, and it ends early when the paper reported no aliases or no description.

## Paper 1

- c1 | Retrieval-Augmented Generation | method | aliases: RAG, 検索拡張生成 | Generation grounded in retrieved passages.
- c2 | Page-Aligned Chunking | method

## Paper 2

- c1 | RAG | method
- c2 | Knowledge Graph | method | aliases: KG | A graph of concepts and their relations.

Return only the JSON object.
