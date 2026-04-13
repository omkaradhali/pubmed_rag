"""
pubmed_rag — A RAG pipeline for biomedical literature.

Modules (added as built):
  ingest     — fetch PubMed abstracts via NCBI E-utilities
  chunk      — split abstracts into overlapping text chunks
  embed      — generate vector embeddings for chunks
  vectorstore — persist and query a ChromaDB collection
  retrieve   — semantic similarity search over embeddings
  generate   — LLM-based answer generation with citation enforcement
  pipeline   — end-to-end RAG chain
  app        — FastAPI web API
"""
