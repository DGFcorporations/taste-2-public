# taste-2

This repository contains the backend API and CLI engine for a system that codifies and applies subjective decision-making standards using semantic search and Large Language Models (LLMs). It solves the problem of capturing personal or organizational rationale—what you prefer and why—into a computable format, enabling applications to autonomously evaluate content and make choices aligned with a specific profile.

## Tech Stack
- **Language:** Python 3.11+
- **API Framework:** FastAPI
- **Vector Database:** LanceDB
- **Embeddings:** Qwen3 via `fastembed`
- **Server:** Granian / Uvicorn

## Key Features
- **Standard Ingestion:** Parses documents to extract candidate rules, doctrine, and rationale.
- **Semantic Evaluation:** Uses an LLM judge to score and refine content against active standards.
- **Feedback Loop:** Logs user decisions to automatically propose new standards.
- **Context Packets:** Compiles active standards into structured context injections for downstream tasks.

## How to Run

Install the package and start the ASGI server via the provided CLI:

```bash
# Install the project
pip install .

# The CLI can print the server start command
taste serve
# Example output: granian --interface asgi --uds ./taste2.sock taste.api:app

# View all CLI operations
taste --help
```

## Repository Scope

This repository contains only the public mechanics of the system: the engine source code and documentation.

**Note on privacy:** The actual data corpus (standards, decision records, rationale, and persona layers) is intentionally excluded. That learned knowledge lives in a separate, private repository, ensuring the engine mechanics can be shared publicly without exposing personal data.
