"""
Unit tests for retrieve.py.

The retrieve() function calls two external dependencies:
  - get_model().encode()  (sentence-transformers, loads ~100MB)
  - get_collection()      (ChromaDB, reads disk)

Both are mocked here — these tests exercise the pure logic:
distance→score conversion, min_score filtering, and result key shape.
"""

from unittest.mock import MagicMock, patch

import pytest

from pubmed_rag.retrieve import retrieve


def _chroma_response(
    documents: list[str],
    metadatas: list[dict],
    distances: list[float],
) -> dict:
    """Build a ChromaDB-style response dict (each field wrapped in an outer list)."""
    return {
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }


def _meta(
    pmid: str = "12345",
    title: str = "Test Paper",
    year: str = "2023",
    chunk_index: int = 0,
    chunk_total: int = 1,
) -> dict:
    """Minimal ChromaDB metadata dict matching the schema written by vectorstore.py."""
    return {
        "pmid": pmid,
        "title": title,
        "year": year,
        "doi": "",
        "doi_url": "",
        "pmc_id": "",
        "pmc_url": "",
        "journal": "",
        "authors": "[]",  # JSON-serialized list, as stored in ChromaDB
        "publication_types": "[]",
        "mesh_terms": "[]",
        "chunk_index": chunk_index,
        "chunk_total": chunk_total,
    }


@pytest.fixture
def mock_deps():
    """Patch get_model and get_collection at the point of use in retrieve.py."""
    with (
        patch("pubmed_rag.retrieve.get_model") as mock_get_model,
        patch("pubmed_rag.retrieve.get_collection") as mock_get_col,
    ):
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        # encode([query]) returns an object whose .tolist()[0] gives a flat vector
        mock_model.encode.return_value.tolist.return_value = [[0.1] * 384]
        mock_collection = MagicMock()
        mock_get_col.return_value = mock_collection
        yield mock_collection


# Tests


class TestRetrieve:
    def test_distance_converts_to_score(self, mock_deps):
        mock_deps.query.return_value = _chroma_response(
            documents=["some chunk"],
            metadatas=[_meta()],
            distances=[0.3],
        )
        results = retrieve("test query", n_results=1)
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(0.7, abs=1e-4)

    def test_score_rounded_to_four_decimal_places(self, mock_deps):
        distance = 0.123456789
        mock_deps.query.return_value = _chroma_response(
            documents=["text"],
            metadatas=[_meta()],
            distances=[distance],
        )
        results = retrieve("query")
        assert results[0]["score"] == round(1.0 - distance, 4)

    def test_min_score_filters_low_matches(self, mock_deps):
        # distances 0.8 and 0.2 → scores 0.2 and 0.8; only the 0.8 passes min_score=0.5
        mock_deps.query.return_value = _chroma_response(
            documents=["weak", "strong"],
            metadatas=[_meta(pmid="1"), _meta(pmid="2")],
            distances=[0.8, 0.2],
        )
        results = retrieve("query", n_results=2, min_score=0.5)
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(0.8, abs=1e-4)

    def test_min_score_zero_returns_all_results(self, mock_deps):
        mock_deps.query.return_value = _chroma_response(
            documents=["a", "b", "c"],
            metadatas=[_meta(pmid=str(i)) for i in range(3)],
            distances=[0.9, 0.5, 0.1],
        )
        results = retrieve("query", n_results=3, min_score=0.0)
        assert len(results) == 3

    def test_result_has_required_keys(self, mock_deps):
        mock_deps.query.return_value = _chroma_response(
            documents=["chunk text"],
            metadatas=[_meta()],
            distances=[0.2],
        )
        results = retrieve("query")
        expected_keys = {
            "text",
            "pmid",
            "title",
            "year",
            "doi",
            "doi_url",
            "pmc_id",
            "pmc_url",
            "pubmed_url",
            "journal",
            "authors",
            "publication_types",
            "mesh_terms",
            "chunk_index",
            "chunk_total",
            "score",
        }
        assert set(results[0].keys()) == expected_keys

    def test_empty_collection_returns_empty_list(self, mock_deps):
        mock_deps.query.return_value = _chroma_response(documents=[], metadatas=[], distances=[])
        assert retrieve("query") == []

    def test_all_results_filtered_by_high_min_score(self, mock_deps):
        mock_deps.query.return_value = _chroma_response(
            documents=["a", "b"],
            metadatas=[_meta(pmid="1"), _meta(pmid="2")],
            distances=[0.7, 0.6],  # scores 0.3 and 0.4 — both below min_score=0.9
        )
        results = retrieve("query", n_results=2, min_score=0.9)
        assert results == []
