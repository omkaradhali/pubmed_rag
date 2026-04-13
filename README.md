# pubmed_rag

A Retrieval-Augmented Generation (RAG) pipeline over PubMed biomedical literature.

## Overview

Fetches and indexes PubMed abstracts, embeds them into a vector store, and answers natural language queries grounded in retrieved papers.

## Stack

- **Data** — PubMed API (E-utilities)
- **Embeddings** — sentence-transformers
- **Vector Store** — TBD (FAISS / ChromaDB)
- **LLM** — Claude API
- **Backend** — FastAPI

## Project Structure

```
pubmed_rag/
├── data/          # raw and processed PubMed data
├── src/           # core pipeline code
├── notebooks/     # exploration and experiments
└── docs/          # design notes and references
```
