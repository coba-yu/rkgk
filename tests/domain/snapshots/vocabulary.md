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
