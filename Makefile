.PHONY: install dev test test-all lint serve worker demo frontend docker
install: ; pip install -e ".[dev,mcp,providers]"
dev: install ; cd frontend && npm install
test: ; pytest -q -m "not slow"
test-all: ; pytest -q
lint: ; ruff check saphire tests
serve: ; SAPHIRE_INLINE_JOBS=1 saphire serve --reload
worker: ; saphire worker
demo: ; saphire demo
frontend: ; cd frontend && npm run dev
docker: ; docker compose up --build
