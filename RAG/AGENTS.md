# RAG Guide

This folder contains local retrieval and knowledge-base preparation code.

- `query.py` and `test_query_retrieval.py` are direct query/check entry points.
- `script/run_rag_pipeline.py` orchestrates the local ingestion pipeline.
- `script/crawl_docs.py`, `download_arxiv_pdfs.py`, `split_pdfs.py`, `extract_md.py`, `mineru_runner.py`, and `cut_into_chunks.py` prepare source documents.
- `script/write_qdrant.py` writes chunks to Qdrant for retrieval.

The runtime agent queries this data through `tools/GeneralTool.py`, not by importing every script here. Keep ingestion scripts runnable as standalone maintenance tools and keep service configuration in environment variables or `config/config.yml`.

Use `.venv` for Python commands. Some scripts depend on external services or large local data, so prefer targeted tests or dry runs when possible.
