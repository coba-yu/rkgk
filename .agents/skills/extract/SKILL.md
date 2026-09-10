---
name: rkgk-extract
description: 指定された論文 ID の抽出 JSON を作る。論文本文から概念・論文と概念の関係・概念間の関係を根拠付きで抽出し、rkgk に保存するときに使う。
---

# rkgk extract

指定された論文 ID について、本文から概念とその関係を抽出した JSON を作り、rkgk に保存する。
グローバルな概念 ID（slug）は後段の正規化が付けるので、このスキルでは論文内でのみ通じる ID を使う。

## 入力

`data/papers/NNNN/paper.json` を読み、`page_count` とタイトルを確認する。
`NNNN` は論文 ID を 4 桁ゼロ埋めしたものである。
`data/papers/NNNN/pages/*.md` を 1 ページ目から最終ページまで全て読む。
一部のページだけを読んで抽出してはならない。

## 出力形式

`uv run rkgk schema extraction` を実行して JSON Schema を取得し、その構造に従った JSON を作る。
`schema_version` は Schema が示す値をそのまま入れる。
`paper_id` は指定された論文 ID と一致させる。
概念の `local_id` は `c1`、`c2` のように 1 から始まる連番にする。
`paper_concepts` と `concept_relations` は `local_id` だけを参照する。

## 語彙

概念の種類と関係の名前は次の語彙だけを使う。
値は英語のまま書く。

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

## ルール

概念は本文に実際に出てくるものだけを挙げる。
本文にない一般知識の概念を足さない。
概念の `name` は英語の正式名称にし、略記や表記ゆれは `aliases` に入れる。
論文が自ら提案したものは `proposes`、既存のものを手法や実験で使っているなら `uses` を選ぶ。
解こうとしている問題は `addresses`、ただ言及しているだけなら `mentions` を選ぶ。
`mentions` と `uses` の区別が判断の分かれ目になるので、その概念を実際に動かしているかどうかで決める。
evidence の `quote` は本文からの逐語引用にする。
要約したり言い換えたりした文字列を `quote` に入れない。
evidence の `page` は、その引用が載っているページ番号にする。
概念間の関係は、本文に根拠がある場合だけ書く。
根拠がなければ `concept_relations` は空でよい。
`summary_ja` は日本語で 200〜400 字にし、論文の問題設定・手法・結果が分かるように書く。

## 手順

1. `data/papers/NNNN/paper.json` と全ページの Markdown を読む。
2. `uv run rkgk schema extraction` で JSON Schema を確認する。
3. 抽出結果の JSON を一時ファイル（例: `/tmp/extraction-NNNN.json`）に書く。
4. `uv run rkgk extract validate <paper_id> <file>` を実行する。
5. 出力が `{"status": "invalid", ...}` なら `issues` の `path` と `message` を読み、その箇所を直して 4 に戻る。
6. 出力が `{"status": "ok", ...}` になったら `uv run rkgk extract save <paper_id> <file>` を実行する。
7. `save` の出力に含まれる `path` を報告する。

`status` が `error` の場合は、論文の成果物か一時ファイルの読み取りに失敗している。
その場合はメッセージのパスを確認し、JSON を書き直すのではなく入力の場所を直す。

## 禁止事項

他の論文の概念一覧や既存の正規化結果を参照しない。
概念の名寄せは後段の正規化が行うので、この論文だけを見て抽出する。
概念に slug を付けない。
`extraction.json` を直接書き込まない。
保存は必ず `uv run rkgk extract save` を通す。
