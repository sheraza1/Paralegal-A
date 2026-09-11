# Paralegal AI — Document Analysis & Report Generation

Processes case PDFs (medical records, police reports, insurance correspondence), indexes them for semantic search, and generates a formatted demand letter with citations.

## How it works

- **`rag_assistant.py`** — extracts text from PDFs and builds a FAISS index over case sections/documents
- **`command_library.py`** — CLI to query the index for relevant sections
- **`citation_tracker.py`** — tracks and attributes source citations used in a report
- **`llm_integration.py`** — local LLM (Hugging Face transformers) for text generation
- **`mcp_server.py`** — Postgres connection layer (case/party/document tables)
- **`generate_report.py`** — pulls it together to produce a demand letter PDF

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` with your Postgres connection:

```
DB_NAME=...
DB_USER=...
DB_PASSWORD=...
DB_HOST=...
DB_PORT=...
```

Then load the schema:

```bash
psql -d your_db -f database_setup_pg.sql
```

## Usage

Search indexed case sections:

```bash
python command_library.py "total medical expenses" --case-id 2024-PI-001 --top-k 5
```

Generate a demand letter for a case:

```bash
python generate_report.py
```

(edit the case ID at the bottom of `generate_report.py` to target a different case)

## Tests

```bash
pytest tests/
```

## Notes

- No external LLM API key is required — `llm_integration.py` runs a local Hugging Face model.
- Case documents are expected under `sample_docs/<case-id>/`; this folder is git-ignored since it holds example case files with realistic (synthetic) personal and medical details — supply your own case PDFs there.
