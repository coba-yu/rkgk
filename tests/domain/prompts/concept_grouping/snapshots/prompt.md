# Task

複数の研究論文から抽出された概念のうち、「同じ概念を指している可能性があるもの」を同じ群にまとめる。
群には 2 件以上の概念を入れ、同じ概念を指すかもしれない相手がいない概念はどの群にも入れない。
候補ペアの類似度は手がかりであって、類似度が高くても別の概念、低くても同じ概念と判断してよい。
このタスクでは実際の概念統合は行わず、統合候補となる概念をグループにまとめる作業のみを行う。
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

- 群どうしを重ねない。
- `paper_id` と `local_id` には、`# Papers` に並ぶ論文と概念のものだけを書く。
- 同じ概念を 2 つ以上の群に入れない。
- 群に別の概念が混ざることを避けるより、同じかもしれない概念を広めに束ねることを優先する。別の概念は次の段で別々の正規化概念に分けられる。

# Papers

論文ごとに個別に抽出したので、`c1`、`c2`、... はそれが記載されている論文の中でのみ通用する。
概念の行は `- local_id | name | type` という形式である。

<paper id="1">
- c1 | Retrieval-Augmented Generation | method
- c2 | Page-Aligned Chunking | method
</paper>

<paper id="2">
- c1 | RAG | method
- c2 | Knowledge Graph | method
</paper>

# Candidate pairs

以下は概念の名前と説明を埋め込んで近かった組で、どの概念を一緒に見るかの手がかりにする。
組の行は `- paper_id:local_id | paper_id:local_id | score` という形式で、`score` はコサイン類似度である。近い組が 1 つもない場合は「なし」とだけ書く。

- 1:c1 | 2:c1 | 0.912
- 1:c2 | 2:c2 | 0.400
