# rkgk

Research Knowledge Graph Kit。論文検索 RAG の MVP。

## 目的

興味のあるカンファレンスやジャーナルの論文を S3 に蓄積し、日本語でテーマを入力すると、次に読む英語論文を探せるようにする。チャンク検索に加えて Knowledge Graph をたどり、「この論文は手法 X を使い、X は Y の一種」と根拠付きで説明できることを重視する。RAG と Knowledge Graph の技術理解が主目的で、リッチな UI は作らず Codex と Claude Code の Agent Skill を入口にする。

## コーディング規約

- `from __future__ import annotations` は使わない。Python 3.13 以上のみ対象なので不要。
