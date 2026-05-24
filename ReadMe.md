AI-powered resume search tool. Ingests PDF resumes, indexes them via Mistral embeddings into ChromaDB, builds structured candidate profiles with LLM, and finds the best match for any job query using two-stage ranking.

## How it works

### 1. Indexing (`/index`)

- Scans a folder for `.pdf` files
- Extracts text via `pdfplumber`, splits into 500-char chunks with 100-char overlap
- Each chunk → Mistral embedding → stored in ChromaDB (persistent vector DB)
- For every new resume, LLM builds a structured profile (summary, skills, experience, projects, education) and saves it to SQLite
- Profiles persist between runs and are used during reranking (Stage B)

### 2. Search flow

```
User query → embedding → ChromaDB (top-10 chunks) → group by file → embedding_score → fetch profiles from SQLite → LLM rerank → final_score → explain top-3
```

**Stage A — Embedding retrieval**
- Query is converted to a Mistral embedding
- ChromaDB finds the 10 nearest chunks (by cosine distance)
- Chunks are grouped by source filename
- For each file: `embedding_score = 0.7 * best_chunk_distance + 0.3 * avg_chunk_distance`
- **Lower embedding_score = better match** (distance ~0 = identical, ~1.0 = unrelated)

**Stage B — LLM rerank via profiles**
- Structured profiles for those 10 candidates are loaded from SQLite
- **Single** LLM call sends all 10 profiles + query, asks to rate each 1-10
- Final score combines embedding distance and LLM rating:

`final_score = 0.3 * embedding_score + 0.7 * (1 - llm_rating / 10)`

- `embedding_score` is ~0.2–0.8 typically
- `llm_rating` is 1–10 (10 = best)
- `(1 - llm_rating / 10)` inverts LLM so both terms go the same direction: **lower = better**
- Typical final scores: ~0.15–0.6

**Stage C — Explanation**
- For top-3 candidates: LLM explains the match in under 50 words (skills, company, tasks)

**LLM calls per search query:**
- 1 embedding call (query → vector)
- 1 rerank call (10 profiles + query → JSON ratings)
- 3 explanation calls (one per top candidate)
- Total: **5 Mistral API calls per search**

## Commands

- `/index /path/to/resumes` — index PDFs and build profiles
- Send any text message — search for candidates

## Requirements

- Python 3.10+
- `MISTRAL_API_KEY` in `.env`
- `TELEGRAM_BOT_API_KEY` in `.env`

## Run

```bash
source venv/bin/activate
python bot.py
```
