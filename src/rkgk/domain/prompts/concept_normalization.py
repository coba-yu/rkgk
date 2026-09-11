"""Builds the prompt that asks an agent to normalize the concepts extracted from every paper.

Kept next to `check_normalization_against_extractions` so the rules stated here and the rules enforced there
cannot drift apart.
"""

import json

from rkgk.domain.models.concept_normalization import (
    CONCEPT_NORMALIZATION_SCHEMA_VERSION,
    ConceptNormalizationIssue,
)
from rkgk.domain.models.paper_extraction import PaperExtraction
from rkgk.domain.models.vocabulary import describe_vocabulary

_RULES = (
    "2 つの概念は同じものを指すときにだけ統合し、意味が同じかどうか判断できない概念は別々の正規化概念として残す。",
    "`canonical_name` には概念の英語の正式名称を書き、`id` はそれを小文字にした単語を `-` でつないで導き、"
    "`^[a-z0-9]+(-[a-z0-9]+)*$` に一致させる。",
    "統合した概念のすべての表記、略語、日本語名を `aliases` に集める。",
    "抽出されたすべての概念を、抽出元の論文 id と抽出時のローカル id として、ちょうど 1 つの `merged_from` "
    "に記載する。",
    "`type` は統合した概念が抽出されたときの type のままにする。",
    "どの論文も抽出していない概念を追加しない。",
    "一般知識に基づく概念間の関係は正規化概念どうしの間にだけ追加し、他の id は書かない。",
    "それらの関係には `is_a`、`part_of`、`used_for` を優先して使い、他の 3 つのどれも当てはまらないときにだけ "
    "`related_to` を使う。",
    "一般知識に基づく関係には根拠がないため、すべての関係に、それが成り立つ理由を述べた 1 文の `rationale` を付ける。",
    f"`schema_version` に {CONCEPT_NORMALIZATION_SCHEMA_VERSION} を設定する。",
)


def _describe_concepts(extraction: PaperExtraction) -> list[str]:
    lines = []
    for concept in extraction.concepts:
        parts = [concept.local_id, concept.name, concept.type.value]
        if concept.aliases:
            parts.append(f"aliases: {', '.join(concept.aliases)}")
        if concept.description:
            parts.append(concept.description)
        lines.append("- " + " | ".join(parts))
    return lines


def build_concept_normalization_prompt(
    extractions: tuple[PaperExtraction, ...],
    previous: object | None = None,
    issues: tuple[ConceptNormalizationIssue, ...] = (),
) -> str:
    """Write the instructions and the extracted concepts an agent needs to normalize them in one pass.

    A retry gets the rejected JSON and the issues appended, so the agent corrects its own answer instead of
    starting over and losing the parts that were already right.
    """
    lines = [
        "# Task",
        "",
        "複数の研究論文から抽出された概念を、1 つの共通語彙に統合する。",
        "以下の概念を読み、そのそれぞれを正規化概念として 1 回だけ報告し、"
        "それらの正規化概念どうしがどう関係するかを述べる。",
        "与えられた JSON Schema に従う単一の JSON オブジェクトで回答する。",
        "",
        "## Rules",
        "",
        *(f"- {rule}" for rule in _RULES),
        "",
        "## Vocabulary",
        "",
        "ノード種別と関係名は以下のものを使い、表記は示したとおりにする。",
        "",
        describe_vocabulary().rstrip("\n"),
        "",
        "## Papers",
        "",
        "論文ごとに個別に抽出したので、`c1`、`c2`、... はそれが記載されている論文の中でのみ通用する。",
        "概念の行は `- local id | name | type | aliases: ... | description` という形式で、論文が aliases や "
        "description を報告していない場合は途中で終わる。",
    ]
    for extraction in extractions:
        lines += ["", f"## Paper {extraction.paper_id}", "", *_describe_concepts(extraction)]
    if previous is not None:
        lines += [
            "",
            "## Previous attempt",
            "",
            "この JSON は却下された。",
            "",
            "```json",
            json.dumps(previous, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Issues",
            "",
            *(f"- {issue.path}: {issue.message}" for issue in issues),
            "",
            "上記のすべての問題を修正した、完全な JSON オブジェクトを返す。",
        ]
    lines += ["", "JSON オブジェクトだけを返す。"]
    return "\n".join(lines) + "\n"
