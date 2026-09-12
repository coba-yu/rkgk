.PHONY: test lint fmt

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check . && uv run ty check

fmt:
	uv run ruff format .
