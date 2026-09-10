# rkgk

Research Knowledge Graph Kit. 設計は `notes/mvp-architecture.md`、実装計画は `notes/tasks.md` を参照。

## コーディング規約

- `from __future__ import annotations` は使わない。Python 3.13 以上のみ対象なので不要。
- 型ヒントは PEP 604 / 585 の記法（`X | None`、`list[str]`）を使う。
- 開発コマンドは `make test` / `make lint` / `make fmt`。
