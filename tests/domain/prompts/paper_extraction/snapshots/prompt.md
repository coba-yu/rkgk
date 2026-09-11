# Task

1 本の研究論文の知識グラフを抽出する。
以下の論文を読み、そこに含まれる概念、論文とそれらの概念の関係、および概念どうしの関係を報告する。
与えられた JSON Schema に従う単一の JSON オブジェクトで回答する。

## Vocabulary

ノード種別と関係名は以下のものを使い、表記は示したとおりにする。

### ConceptType

- `problem`: 論文が取り組む課題、制約、または研究上の問い
- `method`: 使用または提案される技術、モデル、アルゴリズム、または構成要素
- `keyword`: problem でも method でもない用語。表示と説明のためだけに保持する (探索には使わない)

### PaperConceptRelation

- `proposes`: 論文がその概念を自身の貢献として導入する
- `uses`: 論文がその概念を手法または実験で適用する
- `addresses`: 論文がその概念を、解こうとしている課題として扱う
- `mentions`: 論文がその概念に言及するだけで、使用も提案もしていない (探索には使わない)

### ConceptRelationType

- `is_a`: source は target の一種 (方向: source -> target)
- `part_of`: source は target の構成要素 (方向: source -> target)
- `used_for`: source は課題または目的である target のために使われる (方向: source -> target)
- `related_to`: source と target が、他の関係では表現できない形で関連する。多用しない (方向: source -> target)

### Origin

- `paper`: 論文本文から、引用による根拠付きで抽出した (evidence が必須)
- `general_knowledge`: 正規化のときに一般知識から追加した。どの論文とも照合していない (evidence なし)

## Rules

- 論文本文に現れる概念だけを報告する。一般知識から概念を追加しない。
- `name` には概念の英語の正式名称を書き、略語や表記ゆれは `aliases` に入れる。
- 論文が自身の貢献として導入する概念には `proposes` を使い、論文が手法または実験で適用する概念には `uses` を使う。
- 論文が解こうとしている課題には `addresses` を使い、論文がその概念に言及しているだけのときは `mentions` を使う。`mentions` と `uses` の境目は、論文がその概念を実際に扱っているかどうかである。
- すべての `quote` は論文本文から一字一句そのまま写す。要約も言い換えもしない。
- すべての `page` には、その引用が印刷されているページを設定する。
- 概念どうしの関係は論文本文が裏付けるときにだけ追加し、裏付けがなければ `concept_relations` は空のままにする。
- `summary_ja` は日本語で 200 文字以上 400 文字以下で書き、課題、手法、結果を含める。

### Identifiers

- すべての概念に、宣言した順で `c1`、`c2`、... というローカル id を付ける。
- 参照するのはそれらのローカル id だけにする。`paper_concepts` と `concept_relations` では他の id を使わない。
- グローバル id や slug を作り出さない。この工程の後で正規化がそれらを割り当てるためである。
- `paper_id` には `# Paper` に示した Id を設定する。

# Paper

Id: 1
Title: Retrieval-Augmented Generation for Conference Paper Search
Year: 2026
Venue: NeurIPS

<page number="1">
# Retrieval-Augmented Generation for Conference Paper Search

## Abstract

We study how a retrieval-augmented generation pipeline helps a reader pick the next paper to read.
Our system indexes conference papers and answers questions in the reader's own language.
</page>

<page number="2">
## Method

The pipeline splits every paper into page-aligned chunks and embeds them with a sentence encoder.

<!-- figure: 1 -->
Figure 1 shows the two stages: dense retrieval over chunks, then a graph walk over the concepts of the retrieved papers.
</page>

<page number="3">
## Scoring

$$ s(q, c) = \alpha \cdot \cos(e_q, e_c) + (1 - \alpha) \cdot g(q, c) $$
<!-- equation: 1 -->
Equation (1) mixes the chunk similarity with a graph score, and alpha controls how much the graph contributes.

## References

[1] Ada Lovelace. Notes on the Analytical Engine. 1843.
</page>

JSON オブジェクトだけを返す。
