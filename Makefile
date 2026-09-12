.PHONY: test lint fmt s3-pull s3-push require-s3-uri

# The corpus lives under one base URI, s3://bucket/prefix, and data/ mirrors the layout under it one to one.
# The originals are left in S3 on pull because a search never opens them; a PDF is copied down by hand when wanted.
# Neither direction deletes, so a file dropped on one side stays on the other until someone removes it on purpose.
RKGK_S3_URI ?=
DATA_DIR ?= data

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check . && uv run ty check

fmt:
	uv run ruff format .

s3-pull: require-s3-uri
	aws s3 sync $(RKGK_S3_URI)/ $(DATA_DIR)/ --exclude '*.pdf' --exclude '*.DS_Store'

s3-push: require-s3-uri
	aws s3 sync $(DATA_DIR)/ $(RKGK_S3_URI)/ --exclude '*.DS_Store'

require-s3-uri:
	@test -n "$(RKGK_S3_URI)" || { echo 'RKGK_S3_URI is not set; export it as s3://bucket/prefix' >&2; exit 1; }
