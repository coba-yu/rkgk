# rkgk

Research Knowledge Graph Kit。
論文検索 RAG の MVP。

## 目的

興味のあるカンファレンスやジャーナルの論文を S3 に蓄積し、日本語でテーマを入力すると、次に読む英語論文を探せるようにする。
チャンク検索に加えて Knowledge Graph をたどり、「この論文は手法 X を使い、X は Y の一種」と根拠付きで説明できることを重視する。
RAG と Knowledge Graph の技術理解が主目的である。
リッチな UI は作らず、Codex と Claude Code の Agent Skill を入口にする。

## コーディング規約

- `from __future__ import annotations` は使わない。Python 3.13 以上のみ対象なので不要。
- コードには How を書く。
- テストコードには What を書く。
- コードコメントには Why / Why not を書く。
- エージェントに渡すプロンプトは、見出しとフィールド名・enum 値などの物理名を除いて日本語で書く。

## 依存関係

- `pyproject.toml` の直接依存（dependencies、optional-dependencies、dependency-groups）は、サプライチェーン攻撃対策として `==` で固定する。
- 固定するバージョンは `uv lock --upgrade` で解決できる最新版にし、`uv.lock` も同じ解決結果に更新する。

## ドキュメント

- 複数の文を1行にまとめない。1文ごとに改行する。
- PR、ドキュメント、コメントには、push されているコードだけを見て分かることだけを書く。ローカルのメモや会話の文脈に依存する記述をしない。

## Pull Request

- PR を作るときは `.github/pull_request_template.md` を読み、タイトルと本文をその指示に従って書く。

## レビューコメントへの対応

- コメント1件につき1コミット。まとめて直さない。
- 各コメントにコミット URL を添えて返信する。
- 設計に関わる指摘は、修正前に相談する。
