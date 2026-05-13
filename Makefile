.PHONY: install test lint format api docker docker-up eval

install:
	uv sync --extra dev

test:
	uv run pytest tests/ -v --cov=src/pubmed_rag --cov-report=term-missing --cov-fail-under=35

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

api:
	uv run uvicorn api.main:app --reload --port 8001

docker:
	docker build -t pubmed-rag .

docker-up:
	docker compose up

eval:
	uv run python eval/evaluate.py --output eval/results.csv
