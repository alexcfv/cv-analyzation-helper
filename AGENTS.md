# AGENTS.md

## What this is

A Python CLI tool that ingests PDF resumes, creates vector embeddings via Mistral, and ranks candidates against a job query using similarity search + LLM explanations.

## Architecture

```
main.py                    # single entrypoint, interactive CLI (prompts for dir path)
embedding/embedder.py      # Mistral embeddings via raw HTTP requests
ingestion/loader.py        # scans folder for .pdf files
ingestion/parser.py        # pdfplumber → text, chunked (500 chars, 100 overlap)
db/vector_store.py         # ChromaDB wrapper (in-memory, NOT persisted between runs)
db/sqlite/connection.py    # SQLite context manager, DB at app.db
db/sqlite/migrations.py    # CREATE TABLE IF NOT EXISTS profiles
services/ranking.py        # score = 0.7*best_distance + 0.3*avg_distance; lower = better
services/explainer.py      # Mistral LLM explanation (openai SDK with Mistral base_url)
services/profile_builder.py# LLM JSON profile extraction (mistral-small-latest)
repositories/profile_repo.py # SQLite CRUD for profiles
models/search_results.py   # SearchResultItem dataclass
```

## Dependencies

No `requirements.txt` or `pyproject.toml`. Inferred from imports:
- `chromadb`, `python-dotenv`, `requests`, `openai`, `pdfplumber`

Install manually: `pip install chromadb python-dotenv requests openai pdfplumber`

## Environment

- Requires `MISTRAL_API_KEY` in `.env` (loaded via `python-dotenv`)
- `venv/` directory present; activate before running
- `app.db` is gitignored (SQLite store for profiles only)

## Key quirks

- **ChromaDB is in-memory** (`chromadb.Client()`) — vectors are lost on each run. Only profiles persist in SQLite.
- **Bug in `main.py:40`**: calls `store.search()` but the variable is named `vectore_store`. Will raise `NameError`.
- **No tests, no linter, no formatter, no type checker** — nothing to run for verification.
- **Interactive CLI**: `main.py` prompts for a directory path via `input()`. Not scriptable without modification.
- The `.env` file is gitignored but a committed copy may exist with a live API key. Never commit secrets.

## Running

```bash
python main.py
# prompts: "Enter resumes dir path: "
```
