.PHONY: test lint fmt skills

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check . && uv run ty check

fmt:
	uv run ruff format .

skills:
	rm -rf .claude/skills && mkdir -p .claude && cp -R .agents/skills .claude/skills
