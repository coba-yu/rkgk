"""Builds the prompt that asks an agent to extract the knowledge graph of one paper.

Kept next to `check_extraction_against_paper` so the rules stated here and the rules enforced there cannot
drift apart.
"""

import json

from rkgk.domain.models.paper import Paper
from rkgk.domain.models.paper_extraction import EXTRACTION_SCHEMA_VERSION, PaperExtractionIssue
from rkgk.domain.models.vocabulary import describe_vocabulary

_RULES = (
    "論文本文に現れる概念だけを報告する。一般知識から概念を追加しない。",
    "`name` には概念の英語の正式名称を書き、略語や表記ゆれは `aliases` に入れる。",
    "論文が自身の貢献として導入する概念には `proposes` を使い、論文が手法または実験で適用する概念には `uses` を使う。",
    "論文が解こうとしている課題には `addresses` を使い、論文がその概念に言及しているだけのときは "
    "`mentions` を使う。`mentions` と `uses` の境目は、論文がその概念を実際に扱っているかどうかである。",
    "すべての `quote` は論文本文から一字一句そのまま写す。要約も言い換えもしない。",
    "すべての `page` には、その引用が印刷されているページを設定する。",
    "概念どうしの関係は論文本文が裏付けるときにだけ追加し、裏付けがなければ `concept_relations` は空のままにする。",
    "`summary_ja` は日本語で 200 文字以上 400 文字以下で書き、課題、手法、結果を含める。",
)


def _identifier_rules(paper_id: int) -> tuple[str, ...]:
    return (
        "すべての概念に、宣言した順で `c1`、`c2`、... というローカル id を付ける。",
        "参照するのはそれらのローカル id だけにする。`paper_concepts` と `concept_relations` では他の id を使わない。",
        "グローバル id や slug を作り出さない。この工程の後で正規化がそれらを割り当てるためである。",
        f"`paper_id` に {paper_id} を設定する。",
        f"`schema_version` に {EXTRACTION_SCHEMA_VERSION} を設定する。",
    )


def build_paper_extraction_prompt(
    paper: Paper, previous: object | None = None, issues: tuple[PaperExtractionIssue, ...] = ()
) -> str:
    """Write the instructions and the paper text an agent needs to extract this one paper.

    A retry gets the rejected JSON and the issues appended, so the agent corrects its own answer instead of
    starting over and losing the parts that were already right.
    """
    lines = [
        "# Task",
        "",
        "1 本の研究論文の知識グラフを抽出する。",
        "以下の論文を読み、そこに含まれる概念、論文とそれらの概念の関係、および概念どうしの関係を報告する。",
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
        "## Identifiers",
        "",
        *(f"- {rule}" for rule in _identifier_rules(paper.meta.id)),
        "",
        "## Paper",
        "",
        f"Title: {paper.meta.title}",
        f"Year: {paper.meta.year}",
        f"Venue: {paper.meta.venue}",
    ]
    for page in paper.pages:
        lines += ["", f"## Page {page.number}", "", page.text.rstrip("\n")]
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
