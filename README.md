# Admission Test 1 - Safe RAG System

This project implements a Retrieval Augmented Generation (RAG) system capable of answering queries about Common Vulnerabilities and Exposures (CVEs) while strictly enforcing privacy guardrails to protect Personal Identifiable Information (PII).

## Features

- **RAG Architecture**: Uses ChromaDB for vector storage and retrieval.
- **Local LLM**: Utilizes `Qwen/Qwen2.5-0.5B-Instruct` for local inference.
- **Safety Guardrails**: Implements a "Safety Guard" using few-shot prompting to detect and block PII leakage (e.g., answering questions about people's private data) while correctly answering security-related queries.
- **Mac Optimization**: optimized to run efficiently on macOS (using CPU for maximum stability with small models).

## Project Structure

- `src/`: Core source code.
  - `pipeline.py`: Main RAG pipeline orchestrator.
  - `llm_engine.py`: Wrapper for the Hugging Face Transformers model.
  - `safety_guard.py`: Logic for constructing safe prompts and handling PII.
  - `vector_store.py`: Interface for ChromaDB.
  - `data_ingestion.py`: Utilities for loading and processing datasets.
- `data/`: Data directory.
  - `chroma_db/`: Persisted vector database.
  - `raw/`: Raw datasets (Personal and CVE data).
- `main.py`: Entry point for the application.

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the main application:

```bash
python main.py
```

The system will ingest the data (if not already done) and you can interact with the RAG pipeline.

## Safety Mechanism

The system uses a strict system prompt and few-shot examples to distinguish between:
1. **PII Requests** (e.g., "Who is John Doe?"): These are strictly REFUSED.
2. **Security Requests** (e.g., "Explain CVE-2024-1234"): These are ANSWERED in detail.