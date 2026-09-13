# Task

統合済みの正規化概念どうしの関係を、一般知識に基づいて提案する。
以下の概念一覧を読み、どの論文も述べていないが一般知識として成り立つ関係を報告する。
概念そのものは確定しているので、統合をやり直さない。
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

- 一般知識に基づく概念間の関係は正規化概念どうしの間にだけ追加し、他の id は書かない。
- 関係には `is_a`、`part_of`、`used_for` を優先して使い、他の 3 つのどれも当てはまらないときにだけ `related_to` を使う。
- 一般知識に基づく関係には根拠がないため、すべての関係に、それが成り立つ理由を述べた 1 文の `rationale` を付ける。
- 一覧にない概念を追加しない。
- `source_id` と `target_id` には異なる slug を書く。
- 同じ `source_id`、`target_id`、`relation` の組を繰り返さない。
- 論文が述べた関係として列挙された組は提案しない。

# Concepts

以下は統合済みの正規化概念であり、`source_id` と `target_id` にはここに並ぶ slug だけを書く。
概念の行は `- slug | canonical_name | type | aliases: ... | description` という形式で、概念が aliases や description を持たない場合は途中で終わる。

- retrieval-augmented-generation | Retrieval-Augmented Generation | method | aliases: RAG, 検索拡張生成 | Generation grounded in retrieved passages.
- page-aligned-chunking | Page-Aligned Chunking | method
- knowledge-graph | Knowledge Graph | method | aliases: KG | A graph of concepts and their relations.

# Paper-stated relations

以下は論文が本文で述べた関係で、構築時に論文由来の辺としてグラフに載るため、同じ `source_id`、`target_id`、`relation` の組は提案しない。
関係の行は `- source_id | relation | target_id` という形式で、論文が述べた関係が 1 つもない場合は「なし」とだけ書く。

- page-aligned-chunking | part_of | retrieval-augmented-generation
- knowledge-graph | used_for | retrieval-augmented-generation
