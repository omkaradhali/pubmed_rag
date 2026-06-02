"""
Unit tests for retrieve.py — v0.2 parent-child resolution + dedup (D-042).

The retrieve() function calls three external dependencies:
  - get_model().encode()  (sentence-transformers, loads ~100MB)
  - get_collection()      (ChromaDB, reads disk)
  - get_parent()          (parents.py sidecar store)

All three are mocked here — these tests exercise the pure logic:
distance→score conversion, min_score filtering, parent-text resolution,
dedup-by-parent_id, and result key shape.
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
    chunk_id: str = "12345_p0_c0",
    parent_id: str = "12345_p0",
    chunk_role: str = "child",
) -> dict:
    """Minimal ChromaDB metadata dict matching the v0.2 schema (D-042)."""
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
        # v0.2 (D-042)
        "chunk_id": chunk_id,
        "chunk_role": chunk_role,
        "parent_id": parent_id,
    }


def _parent_doc(chunk_id: str = "12345_p0", text: str = "Full parent context.") -> dict:
    """Minimal ParentDoc dict matching what parents.get_parent returns."""
    return {
        "chunk_id": chunk_id,
        "chunk_role": "parent",
        "parent_id": None,
        "pmid": "12345",
        "title": "Test Paper",
        "year": "2023",
        "text": text,
        "chunk_index": 0,
        "chunk_total": 1,
        "doi": "",
        "doi_url": "",
        "pmc_id": "",
        "pmc_url": "",
        "authors": [],
        "journal": "",
        "publication_types": [],
        "mesh_terms": [],
    }


@pytest.fixture
def mock_deps():
    """Patch get_model, get_collection, get_parent, and the reranker at use site.

    The reranker defaults to an identity passthrough so these tests exercise the
    stage-1 dense + dedup logic in isolation (and never download MedCPT). Tests
    that want to verify rerank ordering set mock_deps["rerank"].side_effect.
    """
    with (
        patch("pubmed_rag.retrieve.embed_query") as mock_embed_query,
        patch("pubmed_rag.retrieve.get_collection") as mock_get_col,
        patch("pubmed_rag.retrieve.get_parent") as mock_get_parent,
        patch("pubmed_rag.retrieve.rerank") as mock_rerank,
    ):
        # embed_query(query) returns a flat query vector
        mock_embed_query.return_value = [0.1] * 384
        mock_collection = MagicMock()
        mock_get_col.return_value = mock_collection
        # Default: every parent_id resolves to a stub parent doc.
        mock_get_parent.side_effect = lambda chunk_id: _parent_doc(
            chunk_id=chunk_id, text=f"parent[{chunk_id}]"
        )
        # Default reranker: identity — preserve stage-1 order so dense-path
        # assertions hold without loading a cross-encoder.
        mock_rerank.side_effect = lambda query, candidates, **kwargs: candidates
        yield {
            "collection": mock_collection,
            "get_parent": mock_get_parent,
            "rerank": mock_rerank,
        }


# Tests
class TestRetrieve:
    def test_distance_converts_to_score(self, mock_deps):
        mock_deps["collection"].query.return_value = _chroma_response(
            documents=["child chunk"],
            metadatas=[_meta()],
            distances=[0.3],
        )
        results = retrieve("test query", n_results=1)
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(0.7, abs=1e-4)

    def test_score_rounded_to_four_decimal_places(self, mock_deps):
        distance = 0.123456789
        mock_deps["collection"].query.return_value = _chroma_response(
            documents=["text"],
            metadatas=[_meta()],
            distances=[distance],
        )
        results = retrieve("query", n_results=1)
        assert results[0]["score"] == round(1.0 - distance, 4)

    def test_min_score_filters_low_matches(self, mock_deps):
        # distances 0.8 and 0.2 → scores 0.2 and 0.8; only 0.8 passes min_score=0.5
        mock_deps["collection"].query.return_value = _chroma_response(
            documents=["weak", "strong"],
            metadatas=[
                _meta(pmid="1", chunk_id="1_p0_c0", parent_id="1_p0"),
                _meta(pmid="2", chunk_id="2_p0_c0", parent_id="2_p0"),
            ],
            distances=[0.8, 0.2],
        )
        results = retrieve("query", n_results=2, min_score=0.5)
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(0.8, abs=1e-4)

    def test_min_score_zero_returns_all_results(self, mock_deps):
        mock_deps["collection"].query.return_value = _chroma_response(
            documents=["a", "b", "c"],
            metadatas=[
                _meta(pmid=str(i), chunk_id=f"{i}_p0_c0", parent_id=f"{i}_p0") for i in range(3)
            ],
            distances=[0.9, 0.5, 0.1],
        )
        results = retrieve("query", n_results=3, min_score=0.0)
        assert len(results) == 3

    def test_result_has_required_keys(self, mock_deps):
        mock_deps["collection"].query.return_value = _chroma_response(
            documents=["child text"],
            metadatas=[_meta()],
            distances=[0.2],
        )
        results = retrieve("query", n_results=1)
        expected_keys = {
            "text",
            "child_text",
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
            "chunk_id",
            "parent_id",
            "chunk_index",
            "chunk_total",
            "score",
        }
        assert set(results[0].keys()) == expected_keys

    def test_empty_collection_returns_empty_list(self, mock_deps):
        mock_deps["collection"].query.return_value = _chroma_response(
            documents=[], metadatas=[], distances=[]
        )
        assert retrieve("query") == []

    def test_all_results_filtered_by_high_min_score(self, mock_deps):
        mock_deps["collection"].query.return_value = _chroma_response(
            documents=["a", "b"],
            metadatas=[
                _meta(pmid="1", chunk_id="1_p0_c0", parent_id="1_p0"),
                _meta(pmid="2", chunk_id="2_p0_c0", parent_id="2_p0"),
            ],
            distances=[0.7, 0.6],  # scores 0.3 and 0.4 — both below min_score=0.9
        )
        results = retrieve("query", n_results=2, min_score=0.9)
        assert results == []


class TestParentResolutionAndDedup:
    def test_text_field_is_parent_text(self, mock_deps):
        mock_deps["collection"].query.return_value = _chroma_response(
            documents=["the child fragment"],
            metadatas=[_meta()],
            distances=[0.2],
        )
        results = retrieve("query", n_results=1)
        assert results[0]["text"] == "parent[12345_p0]"
        assert results[0]["child_text"] == "the child fragment"

    def test_dedup_by_parent_id_keeps_best_child(self, mock_deps):
        # 3 children — first two share parent A (best at distance 0.1),
        # third belongs to parent B. After dedup we expect 2 unique parents.
        mock_deps["collection"].query.return_value = _chroma_response(
            documents=["A-best", "A-worse", "B-only"],
            metadatas=[
                _meta(chunk_id="X_p0_c0", parent_id="X_p0"),
                _meta(chunk_id="X_p0_c1", parent_id="X_p0"),
                _meta(chunk_id="Y_p0_c0", parent_id="Y_p0"),
            ],
            distances=[0.1, 0.2, 0.3],
        )
        results = retrieve("query", n_results=5)
        assert len(results) == 2
        # First result is the best-scoring child for parent X
        assert results[0]["parent_id"] == "X_p0"
        assert results[0]["child_text"] == "A-best"
        assert results[1]["parent_id"] == "Y_p0"

    def test_dedup_respects_n_results_cap(self, mock_deps):
        # 4 distinct parents available; only top 2 requested.
        mock_deps["collection"].query.return_value = _chroma_response(
            documents=["a", "b", "c", "d"],
            metadatas=[_meta(chunk_id=f"P{i}_p0_c0", parent_id=f"P{i}_p0") for i in range(4)],
            distances=[0.1, 0.2, 0.3, 0.4],
        )
        results = retrieve("query", n_results=2)
        assert len(results) == 2
        assert [r["parent_id"] for r in results] == ["P0_p0", "P1_p0"]

    def test_missing_parent_falls_back_to_child_text(self, mock_deps):
        # Simulate a parents.jsonl/ChromaDB out-of-sync state.
        mock_deps["get_parent"].side_effect = KeyError("missing")
        mock_deps["collection"].query.return_value = _chroma_response(
            documents=["the child fragment"],
            metadatas=[_meta()],
            distances=[0.2],
        )
        results = retrieve("query", n_results=1)
        assert len(results) == 1
        # Fallback: parent text == child text
        assert results[0]["text"] == "the child fragment"
        assert results[0]["child_text"] == "the child fragment"

    def test_chunk_id_and_parent_id_populated_in_result(self, mock_deps):
        mock_deps["collection"].query.return_value = _chroma_response(
            documents=["c"],
            metadatas=[_meta(chunk_id="42_p3_c7", parent_id="42_p3")],
            distances=[0.2],
        )
        results = retrieve("query", n_results=1)
        assert results[0]["chunk_id"] == "42_p3_c7"
        assert results[0]["parent_id"] == "42_p3"


class TestReranking:
    def test_rerank_reorders_over_dense(self, mock_deps):
        # Dense order is parent A (0.1) then parent B (0.2). A reranker that
        # reverses the pool should flip the returned parents to B, then A.
        mock_deps["collection"].query.return_value = _chroma_response(
            documents=["A child", "B child"],
            metadatas=[
                _meta(chunk_id="A_p0_c0", parent_id="A_p0"),
                _meta(chunk_id="B_p0_c0", parent_id="B_p0"),
            ],
            distances=[0.1, 0.2],
        )
        mock_deps["rerank"].side_effect = lambda query, candidates, **kwargs: list(
            reversed(candidates)
        )
        results = retrieve("query", n_results=2)
        assert [r["parent_id"] for r in results] == ["B_p0", "A_p0"]
        mock_deps["rerank"].assert_called_once()

    def test_score_field_stays_cosine_after_rerank(self, mock_deps):
        # Even when rerank reorders, `score` reports the stage-1 cosine value.
        mock_deps["collection"].query.return_value = _chroma_response(
            documents=["A child", "B child"],
            metadatas=[
                _meta(chunk_id="A_p0_c0", parent_id="A_p0"),
                _meta(chunk_id="B_p0_c0", parent_id="B_p0"),
            ],
            distances=[0.1, 0.2],  # cosine 0.9 and 0.8
        )
        mock_deps["rerank"].side_effect = lambda query, candidates, **kwargs: list(
            reversed(candidates)
        )
        results = retrieve("query", n_results=2)
        # B is returned first (reranked) but keeps its cosine 0.8
        assert results[0]["parent_id"] == "B_p0"
        assert results[0]["score"] == pytest.approx(0.8, abs=1e-4)
        # No internal rerank_score leaks into the result schema
        assert "rerank_score" not in results[0]

    def test_rerank_disabled_skips_reranker(self, mock_deps):
        mock_deps["collection"].query.return_value = _chroma_response(
            documents=["a child"],
            metadatas=[_meta()],
            distances=[0.2],
        )
        results = retrieve("query", n_results=1, rerank_enabled=False)
        assert len(results) == 1
        mock_deps["rerank"].assert_not_called()
