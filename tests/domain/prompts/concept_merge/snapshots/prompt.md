# Task

以下の概念は、複数の研究論文から抽出されたもののうち、同じ概念を指しているかもしれないものとして束ねられた統合候補である。
同じものを指す概念を 1 つの正規化概念にまとめ、別物と判断した概念は別の正規化概念にする。
以下のすべての概念を、ちょうど 1 つの `merged_from` に入れる。
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

- 2 つの概念は同じものを指すときにだけ統合し、意味が同じかどうか判断できない概念は別々の正規化概念として残す。
- `canonical_name` には概念の英語の正式名称を書き、`id` はそれを小文字にした単語を `-` でつないで導き、`^[a-z0-9]+(-[a-z0-9]+)*$` に一致させる。
- 統合した概念のすべての表記、略語、日本語名を `aliases` に集める。
- 以下に並ぶすべての概念を、その論文 id と抽出時のローカル id として、ちょうど 1 つの `merged_from` に記載する。
- `type` は統合した概念が抽出されたときの type のままにする。
- 以下に並んでいない概念を追加しない。

# Concepts

`paper_id:local_id` は抽出元の論文 id と、その論文の中でのみ通用するローカル id である。
概念の行は `- paper_id:local_id | name | type | aliases: ... | description` という形式で、論文が aliases や description を報告していない場合は途中で終わる。

- 1:c1 | Retrieval-Augmented Generation | method | aliases: RAG, 検索拡張生成 | Generation grounded in retrieved passages.
- 2:c1 | RAG | method
- 2:c2 | Knowledge Graph | method | aliases: KG | A graph of concepts and their relations.
