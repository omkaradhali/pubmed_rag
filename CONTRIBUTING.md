# Contributing to pubmed-rag

Thank you for your interest in contributing!

---

## Development Setup

### Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)

### Clone and install

```bash
git clone https://github.com/omkaradhali/pubmed_rag
cd pubmed_rag
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Environment

```bash
cp .env.example .env
# Set at minimum: NCBI_API_KEY (optional) and your chosen LLM provider key
```

---

## Running Tests

```bash
pytest -v
```

All 46 tests should pass. Tests use mocks for external dependencies (NCBI API, sentence-transformers, ChromaDB, LLM providers) so no API keys or running services are required.

---

## Linting and Formatting

This project uses [Ruff](https://github.com/astral-sh/ruff) for both linting and formatting.

```bash
ruff check src/ api/ tests/ eval/    # lint
ruff format src/ api/ tests/ eval/   # format
```

Pre-commit hooks run Ruff automatically on every commit:

```bash
pre-commit install
```

The `detect-secrets` hook also runs on commit to prevent accidental key exposure.

---

## Docker

```bash
docker build -t pubmed-rag .
docker compose up
```

> **Note:** If your firewall blocks Docker bridge outbound traffic (e.g. a UFW `DOCKER-USER` reject rule), build with `--network=host`:
> ```bash
> docker build --network=host -t pubmed-rag .
> ```

---

## Running the Evaluation

```bash
uv pip install -e ".[eval]"
ANTHROPIC_API_KEY=sk-... python eval/evaluate.py --output eval/results.csv
```

RAGAS uses Claude as an internal judge, so an `ANTHROPIC_API_KEY` is required. The evaluation runs 20 clinical oncology questions and scores faithfulness, answer relevancy, and context precision.

---

## Commit Style

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

| Type | When to use |
|---|---|
| `feat:` | New capability |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `test:` | Tests only |
| `refactor:` | Code change with no functional effect |
| `chore:` | Tooling, deps, CI |

Example: `feat: add hybrid BM25 + vector search to retrieve.py`

---

## Pull Requests

- Keep each PR focused on one concern.
- Add tests for new public functions.
- Run `pytest` and `ruff check` before opening a PR.
- If you're making a significant architecture decision, add an ADR to `docs/decisions/` following the existing format.

---

## Project Structure

See the [README](README.md#project-structure) for a full breakdown of the codebase.
