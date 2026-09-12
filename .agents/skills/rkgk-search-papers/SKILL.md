---
name: rkgk-search-papers
description: 日本語のテーマから次に読む英語論文を探す。「〜の論文を探して」「〜について次に読む論文は」「〜に関連する論文」と言われたら使う。英語クエリを作って `uv run search` を実行し、候補を S3 リンク・日本語要約・読むべき理由・到達経路付きで報告する。
---

# 次に読む論文を探す

`search` コマンドはベクトル検索で直接候補を集め、Knowledge Graph をたどって候補を増やし、根拠と到達経路を 1 つの JSON で返す。
このスキルは、その前後にあるエージェントの仕事を定める。
英語クエリを作ること、JSON を読んで読むべき理由を書くこと、到達経路を根拠付きで説明することの 3 つである。

## 前提

- リポジトリのルートで実行する。
- `data/index/` に構築済みの index があること。
  無ければ S3 から取得するか、`uv run --extra embedding build` で構築するようユーザーに案内し、検索の途中で `extract` / `normalize` / `build` を実行しない。
- 埋め込みモデル（Qwen3）を使うため、`uv run --extra embedding` で起動する。

## S3 との同期（参考コマンド）

`data/` は S3 の base URI 配下と同じ構造で、base URI は環境変数 `RKGK_S3_URI`（`s3://bucket/prefix`）で渡す。
`RKGK_S3_URI` が無ければ、`.env.example` を `.env` にコピーして値を書くようユーザーに案内し、`set -a; source .env; set +a` で読み込んでから実行する。
index が無いとき、または抽出結果や index を S3 に戻すときに、ユーザーの依頼を受けてから次を実行する。

```
aws s3 sync "$RKGK_S3_URI/" data/ --exclude '*.pdf' --exclude '*.DS_Store'
aws s3 sync data/ "$RKGK_S3_URI/" --exclude '*.DS_Store'
```

取得では原本 PDF を除外する。検索は原本を開かないため。
どちらの向きも `--delete` を付けず、片側で消えたファイルをもう片側から消さない。

## 手順

### 1. 英語クエリを作る（必須）

論文は英語なので、日本語のテーマだけでは本文のチャンクに当たりにくい。
テーマを英語に訳したクエリを 1 つと、同義語や手法名で言い換えたクエリを 1〜2 個作る。
日本語のテーマは日本語要約と概念の別名に当たるので、そのまま一緒に渡す。
同じ文字列を 2 回渡すとコマンドが拒否するので、重複させない。

### 2. コマンドを実行する

```
uv run --extra embedding search "検索拡張生成の評価" "retrieval-augmented generation evaluation" "RAG benchmark"
```

位置引数 `QUERY` は 1 つ以上で、クエリごとに上位 `--top-k` 件を取り、和集合にする。
オプションは省略すると `SearchConfig` の既定値になる。

| オプション | 意味 | 既定値 |
| --- | --- | --- |
| `--data-dir` | `papers/` と `index/` を持つディレクトリ | `data` |
| `--embedder` | 埋め込みの実装。`qwen3` のまま使う | `qwen3` |
| `--embedding-model` | 読み込むモデル名。省略すると index の manifest の `embedding_model` | manifest の値 |
| `--top-k` | クエリごとに取る埋め込みアイテムの件数 | 10 |
| `--max-hops` | 概念間関係をたどる回数。0 なら概念を共有する論文だけ | 1 |
| `--max-graph-candidates` | グラフ経由で加える論文の上限 | 10 |
| `--generic-concept-threshold` | 文書頻度がこれを超える概念は探索の起点・経由地にしない | 0.4 |

候補が少ないときは `--top-k` を増やすか、クエリを増やす。
グラフ候補が多すぎるときは `--max-hops 0` か `--generic-concept-threshold` を下げる。
ユーザーが頼まない限り、既定値のまま 1 回で済ませる。

### 3. 終了コードと `status` で分岐する

| 終了コード | `status` | 対応 |
| --- | --- | --- |
| 0 | `ok` | 手順 4 へ進む |
| 1 | `invalid` | `reason` が `embedding_model_mismatch` なら、`--embedding-model` を省略して manifest の `index_model` に任せて再実行する。それでも合わなければ index の再構築を案内する |
| 2 | `error` | `message` をそのまま伝える。index が無い、manifest の `schema_version` が非対応、モデルが読めない、クエリが空か重複、のいずれか |

### 4. JSON を読む

`direct_candidates` はベクトル検索が見つけた論文で、読む順に並んでいる。
`graph_candidates` はグラフをたどって届いた論文で、到達経路数の多い順に並んでいる。
候補の順序は変えない。独自のスコアで並べ替えない。

| フィールド | 中身 |
| --- | --- |
| `queries` | 渡したクエリ |
| `config` | 実際に使った設定。`paper_relations` / `concept_relations` / `concept_types` は既定で vocabulary の traversable なもの |
| `direct_candidates[]` / `graph_candidates[]` | `PaperCandidate` の配列 |
| `paper_id`, `title`, `s3_uri`, `summary_ja` | 論文の識別子、題名、S3 URI（`paper.json` に無ければ `null`）、抽出時に保存した日本語要約 |
| `hits[]` | クエリが当たった埋め込みアイテム。`kind` は `chunk` / `summary` / `concept`、`ref` は chunk id か paper id か concept id、`query` は当たったクエリ、`score` はコサイン類似度、`text` は当たった本文 |
| `paths[]` | この論文に届いた到達経路。`source_edge` は起点論文と概念の辺、`hops[]` は概念間の辺と `reached_concept_id`、`target_edge` はこの論文と概念の辺 |
| `source_edge` / `target_edge` | `paper_id`, `concept_id`, `relation`（`proposes` / `uses` / `addresses`）, `evidence[]`（`page`, `quote`, `chunk_id`） |
| `hops[].edge` | `source_id`, `target_id`, `relation`（`is_a` / `part_of` / `used_for` / `related_to`）, `origin`（`paper` / `general_knowledge`）, `paper_id`, `evidence[]`, `rationale` |

`s3_uri` が `null` の論文は、環境変数 `RKGK_S3_URI` と `paper_id` から `$RKGK_S3_URI/papers/0001/paper.pdf` の形で組み立てる。
`RKGK_S3_URI` も無ければ、リンク無しで報告する。

### 5. 読むべき理由を書く

候補ごとに、テーマとの関係を日本語で 1〜2 文書く。
材料は `hits[].text` と `hits[].query`、`summary_ja`、`paths` だけにする。
本文を読んでいないことを推測で埋めない。
どのクエリがどの `kind` に当たったかを添えると、当たり方の強さが伝わる。

### 6. 到達経路を説明する

`paths` を持つ候補は、経路を 1 本ずつ次の形で説明する。

```
論文 1 は concept-a を uses（p.3「…」）
  → concept-a is_a concept-b（一般知識による補完、未検証）
  → 論文 5 は concept-b を proposes（p.1「…」）
```

`origin` が `paper` の辺は `evidence` の `quote` とページを引用する。
`origin` が `general_knowledge` の辺は `rationale` を添え、未検証の補完であると必ず明記する。
経路は辺の方向と逆にたどることがあるので、`hops[].edge` の `source_id` → `target_id` の向きで関係を読み、`reached_concept_id` で到達側を示す。

### 7. 報告する

次の順で、Markdown で報告する。

1. 使ったクエリと、既定値から変えた設定
2. 直接候補（`direct_candidates` の順）: 題名、S3 リンク、`summary_ja`、読むべき理由、あれば到達経路
3. グラフ候補（`graph_candidates` の順）: 同上。到達経路は必ず書く
4. 候補が 0 件なら、その旨とクエリの変え方

## やらないこと

- 検索の途中で `extract` / `normalize` / `build` を実行しない。
- 原本 PDF を勝手にダウンロードしない。
  頼まれたら `aws s3 cp <s3_uri> data/papers/0001/paper.pdf` のように 1 本ずつ取る。
- 結果を独自の順に並べ替えない。要約や根拠を書き換えない。
- 出力 JSON をファイルに保存しない。
