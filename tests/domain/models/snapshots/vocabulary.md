## ConceptType

- `problem`: 論文が取り組む課題、制約、または研究上の問い
- `method`: 使用または提案される技術、モデル、アルゴリズム、または構成要素
- `keyword`: problem でも method でもない用語。表示と説明のためだけに保持する (探索には使わない)

## PaperConceptRelation

- `proposes`: 論文がその概念を自身の貢献として導入する
- `uses`: 論文がその概念を手法または実験で適用する
- `addresses`: 論文がその概念を、解こうとしている課題として扱う
- `mentions`: 論文がその概念に言及するだけで、使用も提案もしていない (探索には使わない)

## ConceptRelationType

- `is_a`: source は target の一種 (方向: source -> target)
- `part_of`: source は target の構成要素 (方向: source -> target)
- `used_for`: source は課題または目的である target のために使われる (方向: source -> target)
- `related_to`: source と target が、他の関係では表現できない形で関連する。多用しない (方向: source -> target)

## Origin

- `paper`: 論文本文から、引用による根拠付きで抽出した (evidence が必須)
- `general_knowledge`: 正規化のときに一般知識から追加した。どの論文とも照合していない (evidence なし)
