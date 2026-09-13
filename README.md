# rkgk

Research Knowledge Graph Kit。

興味のあるカンファレンスやジャーナルの論文を S3 に蓄積し、日本語でテーマを入力すると、次に読む英語論文を探せるようにする MVP である。
チャンク検索に加えて Knowledge Graph をたどり、「この論文は手法 X を使い、X は Y の一種」と根拠付きで説明することを重視する。
前処理成果物を S3 から取得し、`extract` → `normalize` → `build` で index を作り、`search` で引く、という一方向のパイプラインになっている。
リッチな UI は作らず、Codex と Claude Code の Agent Skill `rkgk-search-papers` を入口にする。

## 開発

```
uv sync
make test
make lint
```

## 全体の流れ

| 段階 | 実行するもの | LLM | 書き出す成果物 |
| --- | --- | --- | --- |
| 前処理 | 別プロジェクト。成果物を S3 から取得する | - | `data/papers/index.json`, `data/papers/NNNN/paper.json`, `data/papers/NNNN/pages/*.md` |
| 抽出 | `uv run extract <paper_id>` | Claude CLI を呼ぶ | `data/papers/NNNN/extraction.json` |
| 正規化 | `uv run --extra embedding normalize` | 埋め込みモデルを読み、Claude CLI を呼ぶ | `data/normalization/concepts.json`, `data/normalization/concept_relations.json` |
| 構築 | `uv run --extra embedding build` | 呼ばない | `data/index/manifest.json`, `chunks.jsonl`, `items.jsonl`, `embeddings.npy`, `graph.json` |
| 検索 | Skill `rkgk-search-papers` から `uv run --extra embedding search` | 呼ばない | 何も書かない（stdout の JSON だけ） |

`extract` と `normalize` は `claude -p` を起動するので、`claude` コマンドがインストールされ、ログイン済みである必要がある。
`normalize`、`build`、`search` は埋め込みモデル Qwen3（既定は `Qwen/Qwen3-Embedding-0.6B`）を読むので、`uv run --extra embedding` で起動する。
`data/` の完成形の例は `tests/fixtures/pipeline/` にある（論文 3 本分を手で書いたもの）。

## 利用手順

### 1. 前処理成果物を取得する

`.env.example` を `.env` にコピーし、`RKGK_S3_URI` に論文コーパスの base URI（`s3://bucket/prefix`）を書く。

```
cp .env.example .env
set -a; source .env; set +a
aws s3 sync "$RKGK_S3_URI/" data/ --exclude '*.pdf' --exclude '*.DS_Store'
```

`data/` は S3 の base URI 配下と 1 対 1 で対応する。

```
data/papers/index.json                       ← 採番台帳（前処理側が更新）
data/papers/0001/paper.json                  ← 書誌情報
data/papers/0001/pages/001.md                ← 本文（1 ページ 1 ファイル）
data/papers/0001/paper.pdf                   ← 原本（S3 にはあるが検索には要らない）
data/papers/0001/extraction.json             ← extract の出力
data/normalization/concepts.json             ← normalize の出力
data/normalization/concept_relations.json    ← normalize の出力
data/index/manifest.json                     ← build の出力
data/index/chunks.jsonl                      ← build の出力
data/index/items.jsonl                       ← build の出力
data/index/embeddings.npy                    ← build の出力
data/index/graph.json                        ← build の出力
```

ディレクトリ名は論文 ID の 4 桁ゼロ埋め、ページ名は 3 桁ゼロ埋めである。
取得では原本 PDF を除外する。検索は原本を開かないため。

### 2. 抽出する

論文 1 本ごとに実行する。

```
uv run extract 1
```

Claude が本文を読み、日本語要約、概念、論文と概念の関係（`proposes` / `uses` / `addresses` / `mentions`）をページ引用付きで返す。
概念 ID はその論文の中だけで通じる `c1`, `c2`, ... で、全体の語彙になるのは次の段階である。
結果は本文と照合され、合わなければ `--max-attempts`（既定 3）まで再試行し、それでも通らなければ `invalid` として `issues` を返す。
主なオプションは `--data-dir`（既定 `data`）、`--model`（Claude CLI に渡すモデル。省略すると CLI の既定）、`--max-attempts` である。

### 3. 正規化する

```
uv run --extra embedding normalize
```

論文単位ではなく `data/papers/index.json` の全論文をまとめて 1 回で処理する。
未抽出の論文が 1 本でもあると `error` で止まるので、先に全論文の `extract` を終わらせる。
概念の統合は 1 回の `claude -p` では入力が大きすぎて終わらないため、次の 4 段に分けている。

| 段 | 実行者 | 入力 | 出力 |
| --- | --- | --- | --- |
| 0 | Python | 各抽出概念の名前・別名・説明 | 埋め込みと、コサイン類似度付きの候補ペア一覧 |
| 1 | `claude -p` 1 回 | 概念一覧と候補ペア一覧 | 統合候補の群への分割（ローカル id のみ） |
| 2 | `claude -p` 群ごと（並列） | 群に入った概念の詳細 | 群ごとの正規化概念（1 つの群を複数に分けてよい） |
| 3 | Python | 全群の正規化概念 | 結合、分割の検証、slug 衝突の検出と回収 |

段 0 の類似度は閾値ではなく、段 1 に渡す判断材料である。
類似度が高くても別概念、低くても同一概念と判断してよい、と段 1 のプロンプトに書いてある。
群に入らなかった単独概念には `claude -p` を呼ばず、抽出時の名前・別名・説明をそのまま使い、名前から slug を導いて正規化概念にする。
段 3 で 2 つ以上の群が同じ slug を出したら衝突とみなし、衝突した群を 1 つに結合して段 2 をやり直す。
衝突が無くなるまで繰り返し、周回数が `--max-attempts` を超えたら `invalid` で止まる。

最後に、統合後の概念一覧を渡してもう 1 回 `claude -p` を呼び、どの論文にも書かれていない概念間関係（`is_a` / `part_of` / `used_for` / `related_to`）を、一般知識として `rationale` 付きで提案させる。
このとき、論文が本文で述べた関係も統合後の slug に直して既知の関係として渡し、同じ `source_id`、`target_id`、`relation` の組は提案させない。
同じ組が提案された場合は関係の段を却下して再試行させ、`build` でも同じ組の一般知識由来の辺はグラフに載せない。
関係の段が通らなければ、統合が通っていても何も保存しない。

主なオプションは次のとおりである。

| オプション | 既定 | 意味 |
| --- | --- | --- |
| `--data-dir` | `data` | 読み書きするデータディレクトリ |
| `--model` | Claude CLI の既定 | `claude -p` に渡すモデル |
| `--embedding-model` | `Qwen/Qwen3-Embedding-0.6B` | 段 0 で読む埋め込みモデル |
| `--neighbors` | 5 | 段 0 で 1 概念あたり何件の近い概念とペアにするか |
| `--concurrency` | 4 | 段 2 で同時に投げる群の数 |
| `--max-attempts` | 3 | 段ごとの再試行の上限、および段 3 の周回数の上限 |

`--max-attempts` は段ごとに数えるので、段 1 で最大 3 回、段 2 は群ごとに最大 3 回、関係の段で最大 3 回まで再試行する。
`ok` の `attempts` は `{"grouping": 1, "merge_calls": 12, "merge_rounds": 2, "relations": 1}` の形で、`merge_calls` は段 2 の呼び出し総数（再試行と衝突回収を含む）、`merge_rounds` は段 3 の周回数（衝突が無ければ 1）である。
`ok` にはこのほか `groups`（段 1 が作った群の数）が入る。
`invalid` は落ちた段を `stage` に入れ、値は `grouping` / `merge` / `relations` のいずれかである。

### 4. 構築する

```
uv run --extra embedding build
```

チャンク分割、埋め込み、Knowledge Graph 組み立て、manifest 書き出しをまとめて行う。
LLM を呼ばないため、古い成果物や手で直した成果物は再試行せず `invalid` として報告する。

| `reason` | 意味 | やり直すこと |
| --- | --- | --- |
| `extraction_mismatch` | `extraction.json` が本文と合わない | その論文を `extract` |
| `normalization_mismatch` | 正規化が抽出結果と合わない | `normalize` |
| `unresolved_evidence` | 引用が 1 つのチャンクに収まらない | その論文を `extract` |

主なオプションは `--data-dir`、`--embedder`（`qwen3` / `fake`、既定 `qwen3`）、`--embedding-model`（既定 `Qwen/Qwen3-Embedding-0.6B`）、`--max-tokens`（チャンクの語数上限、既定 512）である。
`--embedder fake` はテキストのハッシュからベクトルを作るので `--extra embedding` 無しで動くが、配線が通っていることしか確かめられず、検索の質は保証しない。

### 5. 検索する

Codex では `.agents/skills/rkgk-search-papers/`、Claude Code では `.claude/skills/rkgk-search-papers/`（本体は `.agents` 側に委譲）の Skill から使う。
Skill が日本語のテーマから英語クエリを作り、コマンドを実行し、候補ごとに読むべき理由と到達経路を書く。
Skill を使わずに直接叩くこともできる。

```
uv run --extra embedding search "検索拡張生成の評価" "retrieval-augmented generation evaluation"
```

位置引数 `QUERY` は 1 つ以上で、クエリごとに上位 `--top-k`（既定 10）件を取って和集合にする。
`--max-hops`（既定 1）で概念間関係をたどる回数、`--max-graph-candidates`（既定 10）でグラフ経由の候補数、`--generic-concept-threshold`（既定 0.4）で汎用すぎる概念の足切りを調整する。
`--embedding-model` を省略すると manifest の `embedding_model` が使われ、別のモデル名を渡すと `invalid` の `embedding_model_mismatch` で拒否される。

### 6. S3 に戻す

```
aws s3 sync data/ "$RKGK_S3_URI/" --exclude '*.DS_Store'
```

どちらの向きも `--delete` を付けず、片側で消えたファイルをもう片側から消さない。

## 再構築の範囲

| 変更の種類 | やり直す段階 |
| --- | --- |
| 抽出 JSON の形式や抽出プロンプトの変更 | 全論文を `extract` → `normalize` → `build` |
| 表記ゆれの統合や一般知識関係の変更 | `uv run --extra embedding normalize` → `build` |
| 埋め込みモデルやチャンク設定（`--embedding-model`, `--max-tokens`）の変更 | `build` のみ |
| 論文を 1 本追加 | その論文を `extract` → `normalize` → `build` |

manifest には index の schema 版、ドメインモデル版、埋め込みモデルと次元、チャンク設定、対象論文 ID が記録される。
版が合わない index は移行せず読み込みを拒否するので、`build` し直す。
手で直した成果物や古い成果物は、`build` が本文照合と整合チェックで検出する。

## 終了コードと出力

どのコマンドも stdout に JSON オブジェクトを 1 つだけ出力し、結果は終了コードで返す。
エージェントは終了コードで分岐し、詳細を同じ出力から読む。

| 終了コード | `status` | 意味 |
| --- | --- | --- |
| 0 | `ok` | 成功。成果物のパスや件数が入る |
| 1 | `invalid` | 入力や成果物の内容が通らない。`issues` か `reason` を読む |
| 2 | `error` | 実行できない。`message` を読む |
