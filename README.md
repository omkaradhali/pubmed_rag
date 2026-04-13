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

## Citation

If you use this software in your research, please cite it:

```bibtex
@software{adhali2026pubmedrag,
  author = {Adhali, Omkar},
  title  = {pubmed\_rag},
  url    = {https://github.com/omkaradhali/pubmed_rag},
  year   = {2026},
  license = {MIT}
}
```

Or use the **"Cite this repository"** button on the GitHub repo page.

## License

MIT © [Omkar Adhali](https://github.com/omkaradhali)
