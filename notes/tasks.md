# 実装計画（PR 単位のチェックリスト）

更新日: 2026-09-12
状態: PR01〜PR15 をマージ済み。フィクスチャ 3 本で抽出検証 → 正規化検証 → build → search が一巡し、README に利用手順を書いた。実データ検証で出た作業はフェーズ 7 に書く。
前提: `notes/mvp-architecture.md` の決定に従う。1 PR の差分はテストコードを除いて約 200 行を目安とする。ただし「同じ契約の両面」（スキーマとそれを生成する Skill、語彙とそれを使うエンティティ）は分けると片方だけで妥当性を判断できないため、350 行程度まで許容して1 PR にまとめる。各 PR は単体で `uv run ruff check` と `uv run pytest` が通る状態でマージする。

## 共通ルール

- ブランチ名: `pr{NN}-{slug}`（例: `pr02-domain-model`）。
- PR には対象モジュールのテストを含める。テスト行数は行数目安に含めない。
- 外部モデル（Qwen3）を要するテストは `@pytest.mark.slow` とし、既定では実行しない。それ以外は FakeEmbedder / FakeTokenizer で動かす。
- 設計に影響する判断が出たら実装を止めて相談し、`mvp-architecture.md` の決定記録に追記してから再開する。
- 依存関係は矢印で示す。矢印のない PR 同士は並行して作業できる。
- レイヤー配置は `mvp-architecture.md` の「レイヤー構成（DDD）」に従う。純粋なロジックは `domain`、外部アクセスは `infrastructure`、入口は `usecase`、CLI は usecase を呼ぶだけにする。

## フェーズ 1: 骨組みとドメインモデル

- [x] **PR01 プロジェクト骨組み**（約 120 行）
  - uv + Python 3.13、src layout、`pyproject.toml`（ruff / pytest 設定）、`src/rkgk/__init__.py`
  - `rkgk` CLI の入口（argparse、サブコマンド登録の仕組みのみ）
  - `Makefile` の雛形（`test`、`lint`）、`.gitignore` に `data/` と `.claude/skills/` を追加
  - 受け入れ: `uv run rkgk --help` が動く。`uv run pytest` が空で通る

- [x] **PR02 ドメインモデル**（約 330 行、語彙とエンティティを一体でレビュー）← PR01
  - 語彙: `ConceptType`（problem / method / keyword）、`PaperConceptRelation`（proposes / uses / addresses / mentions）、`ConceptRelationType`（is_a / part_of / used_for / related_to）、`Origin`（paper / general_knowledge）
  - 各関係のメタ情報（意味、方向、探索で使うか）を Enum に添える。抽出プロンプト用の説明文を生成する関数
  - エンティティ: pydantic v2 で `PaperMeta`（paper.json）、`Evidence`、`Concept`、`PaperConcept`、`ConceptRelation`、`Chunk`
  - `Evidence` は `page`、`quote`、`chunk_id?`。`ConceptRelation` は origin が paper のとき evidence 必須
  - 受け入れ: 探索対象の関係一覧が Enum から導ける。説明文がスナップショットテストで固定される。不正な組み合わせ（general_knowledge に evidence、mentions を探索に使う等）が検証で落ちる

## フェーズ 2: 入力と抽出

- [x] **PR03 前処理成果物のローダー**（約 160 行 + レビュー対応で層分割、例外を NotFound / Unreadable / Invalid に分類）← PR02
  - `domain/paper.py`: `Paper` / `Page` / `PageMarker` / `PaperIndexEntry` と `parse_page()`（`<!-- figure: N -->` / `<!-- equation: N -->` を位置付きで記録し本文は保持）
  - `domain/repositories.py`: `PaperRepository` Protocol と `PaperRepositoryError`
  - `infrastructure/file_paper_repository.py`: `papers/index.json`、`papers/NNNN/paper.json`、`pages/NNN.md` の読み込みと過不足検証
  - `usecase/load_paper.py` / `usecase/list_papers.py`
  - レビュー対応で `domain/entities.py` を `base.py` / `paper.py` / `graph.py` / `chunk.py` に分割
  - テスト: 合意形式のフィクスチャ論文を `tests/fixtures/papers/0001` に追加

- [x] **PR04 抽出スキーマと headless 抽出**（約 600 行。レビューで Skill から `claude -p` 駆動に変更）← PR03
  - `PaperExtraction`（summary_ja、concepts[] with 論文ローカル ID `c1`…、paper_concepts[]、concept_relations[]）
  - 参照整合性: 関係が宣言済み概念を指す、evidence の page が範囲内、quote が該当ページ本文に存在（空白正規化後の完全一致）
  - `uv run extract <id>` の 1 アクションのみ（schema / validate / save は持たない）。結果は `data/papers/NNNN/extraction.json`
  - `build_paper_extraction_prompt()`（domain の純粋関数、スナップショットで固定）、`StructuredOutputAgent` Protocol、`infrastructure/claude_code_agent.py`（`claude -p --output-format json --json-schema`）
  - `usecase/extract_paper.py`: 読み込み → 生成 → 検証 → issues を戻して再試行（上限 3 回） → 保存。`uv run extract <id>`
  - 受け入れ: 不正 JSON に対し path 付きで原因が出る。fake の Extractor で再試行回数が数えられる。実 CLI のテストは slow
  - 配置: `domain/models/paper_extraction.py`（`PaperExtraction` と参照整合性の検証ロジック）、`domain/repositories/paper_extraction.py` に `PaperExtractionRepository`、`domain/agents.py` に `StructuredOutputAgent`、`infrastructure/file_paper_extraction_repository.py` / `claude_code_agent.py`、`usecase/validate_paper_extraction.py` / `save_paper_extraction.py` / `extract_paper.py`、CLI は `cli/extract.py` として usecase を呼ぶだけ

## フェーズ 3: 正規化

- [x] **PR05 正規化スキーマと headless 正規化**（約 400 行。型名は `ConceptNormalization`、一般知識関係は evidence の代わりに `rationale` 必須。domain に `services/` と `prompts/` を新設し PR04 の検証・プロンプトも移設）← PR04
  - `NormalizationResult`: concepts[]（slug、canonical_name、type、aliases[]、merged_from[] = (paper_id, local_id) の列）、concept_relations[]（origin = general_knowledge）
  - 検証: 全論文の全ローカル概念がちょうど1つの slug に対応、slug の書式、自己参照禁止、関係の両端が存在
  - `uv run normalize`（`cli/normalize.py` をスクリプト `normalize` として登録。抽出と同じく 1 アクション） で `data/normalization/concepts.json` と `concept_relations.json` に保存
  - 全論文の抽出済み概念を圧縮した一覧（paper_id、local_id、名前、型、説明）をプロンプトに組み込む純粋関数
  - `build_normalization_prompt()`（統合の基準、曖昧なら分ける、一般知識関係の提案）、`usecase/normalize_concepts.py`（`StructuredOutputAgent` を再利用し、検証 → 再試行 → 保存）
  - 受け入れ: 未対応の概念や二重対応が検出される。2論文のフィクスチャから fake で input → 生成 → validate → save が通る
  - 配置: `domain/models/concept_normalization.py`、`domain/repositories/concept_normalization.py`、`infrastructure/file_concept_normalization_repository.py`、`usecase/normalize_concepts.py`、`cli/normalize.py`（スクリプト `normalize`）

## フェーズ 4: 構築

- [x] **PR06 チャンク化**（約 180 行。FakeTokenizer は `WhitespaceTokenizer` として実装。マーカー行はチャンク本文から落とす）← PR03
  - `Tokenizer` プロトコルと FakeTokenizer（テスト用、空白区切り）
  - ページごとに見出し・段落境界で分割し、上限 512 トークンまで連結。ページをまたがない。`Chunk.id = "{paper_id}:{seq}"`
  - 受け入れ: 段落が上限を超える場合の分割、ページ境界の維持、page_start / page_end の正しさ
  - 配置: `domain/tokenizers.py` に `Tokenizer` Protocol、`domain/services/chunking.py` に分割ロジック、`infrastructure/whitespace_tokenizer.py`（テスト用の FakeTokenizer 相当）

- [x] **PR07 根拠の解決**（約 120 行。戻り値は `PageEvidence` をキーにした辞書。エラー型は `domain/models/graph.py` に同居）← PR06
  - `Evidence.quote` を該当ページのチャンクから探して `chunk_id` を付与。空白正規化後の完全一致のみ
  - 見つからない場合は論文 ID・ページ・引用文を含むエラーを集約して返す
  - 受け入れ: チャンク境界に跨る引用のエラー化、複数一致時は最初のチャンク
  - 配置: `domain/services/evidence_resolver.py`（根拠解決の純粋関数）

- [x] **PR08 埋め込み**（非テスト約 440 行。`IndexRepository` Protocol を前倒しで定義。`Embedder` は `embed_documents` / `embed_queries` の 2 メソッド。sentence-transformers は extra `embedding`）← PR06
  - `Embedder` プロトコル、`FakeEmbedder`（決定的な疑似ベクトル）、`Qwen3Embedder`（sentence-transformers、MPS、遅延ロード）
  - 埋め込み対象の列挙: チャンク、日本語要約、概念（代表名 + 別名 + 説明）。対象の種別と参照先を持つ `EmbeddedItem`
  - `embeddings.npy` と `items.jsonl` の書き出し・読み込み
  - 受け入れ: FakeEmbedder で件数・次元・順序が一致。Qwen3 は slow テスト
  - 配置: `domain/embedders.py` に `Embedder` Protocol、`domain/models/embedding.py` に `EmbeddedItem`、`infrastructure/fake_embedder.py`、`infrastructure/qwen3_embedder.py`、`infrastructure/file_index_repository.py`（embeddings.npy / items.jsonl の書き出し・読み込み）

- [x] **PR09 グラフ構築**（約 180 行。`graph.json` は node-link 形式ではなく `KnowledgeGraph` の pydantic JSON、networkx は `to_networkx()` でメモリ上に組む。両端が同じ slug に統合された関係は捨てる）← PR05, PR07
  - 抽出結果と正規化結果から networkx の有向グラフを組み立てる。ノード: 論文、概念（slug）。辺: 論文→概念（関係型、evidence の chunk_id）、概念→概念（関係型、origin）
  - 概念ごとの文書頻度（含む論文数 / 全論文数）を属性として保持
  - `graph.json` の書き出し・読み込み（node-link 形式）
  - 受け入れ: ローカル ID が slug に置換される。origin と evidence が辺に残る
  - 配置: `domain/services/graph_builder.py`（networkx グラフの組み立てと文書頻度）、`infrastructure/file_index_repository.py` に graph.json の入出力を追加

- [x] **PR10 構築コマンド**（約 150 行。`chunks.jsonl` も index に含める。保存順は chunks → embeddings → graph → manifest。Tokenizer は当面 `WhitespaceTokenizer`）← PR08, PR09
  - `uv run build`: ロード → チャンク化 → 根拠解決 → 埋め込み → グラフ → `data/index/` に書き出し
  - `manifest.json`: schema_version、embedding model 名と次元、domain model version、chunk 設定、対象論文 ID 一覧
  - 未抽出・未正規化の論文があれば中断してどれかを表示。extraction.json の読み込み時に本文照合も行い、手編集の破損を検出
  - 受け入れ: FakeEmbedder でフィクスチャから index 一式が生成される
  - 配置: `domain/models/manifest.py`、`domain/repositories/index.py` に `IndexRepository`、`usecase/build_index.py`、`cli/build.py`

## フェーズ 5: 検索

- [x] **PR11 検索結果スキーマとベクトル検索**（約 200 行。concept ヒットは traversable な関係で結ばれた論文へ展開。レビューで `TraversalPath` を辺そのもので構成する形に変更）← PR10
  - `SearchResult` スキーマを先に固定: 直接候補、グラフ候補、各候補の S3 URI・日本語要約・根拠（ヒット種別 chunk / summary / concept とスコア）・到達経路、使用した設定。以降の PR はこの型を埋める
  - index の読み込みと manifest のバージョン検証（非対応なら明確なエラー）
  - 複数クエリを受け取り、それぞれ上位 k 件を numpy のコサイン類似で取り、和集合にする。ヒットを論文単位にまとめて直接候補にする
  - 受け入れ: 同一論文の重複が1件に集約され、全ヒットが根拠として残る
  - 配置: `domain/models/search.py`（`SearchResult`）、`domain/services/vector_search.py`（コサイン類似と和集合）、manifest 検証は `infrastructure/file_index_repository.py` の読み込み時

- [x] **PR12 グラフ探索戦略**（非テスト約 260 行。関係は順方向・逆方向の両方をたどる（要検討事項に記録）。順位付けと上限は戦略から分離した関数）← PR09, PR11
  - `TraversalStrategy` プロトコルと `SearchConfig`（深さ、対象関係、対象ノード型、汎用概念の閾値 0.4、追加候補数）
  - 合意した戦略のみ実装: 直接候補の概念（proposes / uses / addresses、problem / method）を起点に、共有論文と、概念間関係1ホップ先の論文を追加
  - 到達経路（起点論文 → 概念 → [関係 → 概念] → 論文、各辺の origin）を `SearchResult` の型で記録。並びは到達経路数の降順
  - 受け入れ: 閾値超えの汎用概念と keyword が既定で除外される。設定変更で含められる
  - 配置: `domain/models/search.py` に `SearchConfig`、`domain/services/traversal.py`（`TraversalStrategy` Protocol と合意した戦略）

- [x] **PR13 検索コマンド**（非テスト約 300 行。`--query` ではなく位置引数 `QUERY`。S3 URI は `paper.json` の `s3_uri` があれば載せ、無ければ `null`。manifest の埋め込みモデルと不一致なら `invalid`）← PR11, PR12
  - `uv run search --query <ja> --query <en> …` でベクトル検索と探索を組み合わせ、`SearchResult` を JSON で標準出力に出す。S3 URI は base URI 設定 + id
  - 受け入れ: 直接候補が先、グラフ候補が後。同一論文は1件で経路をすべて保持
  - 配置: `usecase/search_papers.py`、`cli/search.py`

- [x] **PR14 検索の入口と運用コマンド**（Python の変更なし。入口は Skill `rkgk-search-papers`（`.agents/skills/` に本体、`.claude/skills/` は委譲）。Makefile の `s3-pull` / `s3-push` は不採用で、`aws s3 sync` は Skill の参考コマンド、base URI は環境変数 `RKGK_S3_URI`）← PR13
  - 検索の入口（Skill にするか、`rkgk search` が `claude -p` で英語クエリ生成と読むべき理由を作るか）を PR13 の結果で決める
  - `Makefile` に `s3-pull` / `s3-push`（`aws s3 sync`）、base URI とバケットは環境変数
  - 受け入れ: SKILL.md の手順が検索コマンドの引数と一致するテスト
  - 配置: `.agents/skills/search/SKILL.md`、Makefile。Python の変更は無い想定

## フェーズ 6: 一巡

- [x] **PR15 結合テストと利用手順**（既存 fixture は変えず `tests/fixtures/pipeline/` に 3 本を新設。`extract` / `normalize` は本体を動かさず検証部分だけを手書き成果物に適用。3 本では共有概念の文書頻度 0.67 が既定閾値 0.4 を超えるため、共有概念経由の経路は閾値 0.7 の別テストで確認）← PR05, PR10, PR14
  - フィクスチャ論文を3本に増やし、抽出 JSON・正規化 JSON も手書きで用意
  - FakeEmbedder で extract validate → normalize → build → search が一巡する結合テスト
  - README に利用手順（前処理成果物の取得、各 Skill の実行順、再構築の範囲）
  - 受け入れ: グラフ経由の候補が到達経路付きで返る
  - 配置: `tests/` の結合テストと README。層の変更は無い想定

## フェーズ 7: 実データ検証で出た作業

SIGIR 2026 の 34 本で抽出と正規化を通した結果（`mvp-architecture.md` の「実データ検証の決定記録」）から出た作業。

- [ ] **PR16 正規化の多段化** ← PR05, PR08
  - `mvp-architecture.md` の「正規化コマンドの段構成」に従い、埋め込みによる候補ペア算出 → グルーピング → 群ごとの統合 → 結合と検証 → 一般知識関係の付与に分ける
  - 正規化コマンドも extra `embedding` を必要とする
  - 受け入れ: 群ごとの再試行ができ、抽出された全概念がちょうど 1 つの `merged_from` に現れることを機械的に保証する
- [ ] **`ClaudeCodeAgent` の失敗報告の改善** ← PR04
  - 失敗時のメッセージに `is_error` と `result` を含め、並列実行時の不透明な失敗（要検討事項）の原因を追えるようにする

## 並行できる組み合わせ

- PR03 完了後: PR04 と PR06 は並行可
- PR05 完了後: PR09（PR07 も必要）と PR08 は並行可
- PR11 完了後: PR12 は単独。PR13 は PR12 待ち

## 意図的に含めないもの

- Codex への対応、fallback の PDF 抽出、フィードバックログ、boto3、DB / ベクトルストア
- 合意した戦略以外の探索・順位付け
- 旧形式の manifest の読み込み
