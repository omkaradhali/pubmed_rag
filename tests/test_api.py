"""
HTTP layer tests for pubmed_rag API.

The sentence_transformers model is stubbed in conftest.py before any test file is
imported, so these tests never trigger the ~90MB model download in CI.
run_pipeline_structured is mocked per-test via the mock_pipeline fixture.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from api.config import Settings, get_settings
from api.limiter import limiter
from api.main import app
from fastapi.testclient import TestClient
from slowapi.errors import RateLimitExceeded

from pubmed_rag.pipeline import PipelineResult, SourceChunk

client = TestClient(app)


# Test data helpers
def _source(**overrides) -> SourceChunk:
    defaults = dict(
        number=1,
        pmid="12345678",
        title="Trastuzumab in HER2-positive breast cancer",
        authors=["Smith J", "Lee A"],
        journal="NEJM",
        year="2023",
        publication_types=["Journal Article"],
        mesh_terms=["Breast Neoplasms", "Trastuzumab"],
        doi="10.1056/test",
        doi_url="https://doi.org/10.1056/test",
        pmc_id="PMC9999",
        pmc_url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9999/",
        pubmed_url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
        score=0.85,
        chunk_index=0,
        chunk_total=1,
        text="Trastuzumab improves outcomes in HER2-positive breast cancer.",
    )
    return SourceChunk(**{**defaults, **overrides})


def _result(**overrides) -> PipelineResult:
    defaults = dict(
        query="What treats HER2+ breast cancer?",
        answer="Trastuzumab is standard treatment [1].",
        sources=[_source()],
        llm_provider="anthropic",
        llm_model="claude-haiku-4-5-20251001",
        n_chunks_retrieved=1,
        n_chunks_requested=5,
        n_docs_in_corpus=1514,
        corpus_updated_at="2026-05-04",
        avg_score=0.85,
        min_score_retrieved=0.85,
        max_score_retrieved=0.85,
        confidence_tier="High",
        coverage_note=None,
    )
    return PipelineResult(**{**defaults, **overrides})


@pytest.fixture
def mock_pipeline():
    with patch("api.routers.ask.run_pipeline_structured") as mock:
        mock.return_value = _result()
        yield mock


# GET /health
class TestHealth:
    def test_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_response_shape(self):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"

    def test_request_id_header_present(self):
        assert "x-request-id" in client.get("/health").headers


# POST /ask
class TestAsk:
    def test_happy_path_returns_200(self, mock_pipeline):
        response = client.post("/ask", json={"query": "What treats HER2+ breast cancer?"})
        assert response.status_code == 200

    def test_response_has_required_keys(self, mock_pipeline):
        data = client.post("/ask", json={"query": "test"}).json()
        required = {
            "query",
            "answer",
            "sources",
            "llm_provider",
            "llm_model",
            "n_chunks_retrieved",
            "n_chunks_requested",
            "n_docs_in_corpus",
            "corpus_updated_at",
            "avg_score",
            "confidence_tier",
            "coverage_note",
        }
        assert required.issubset(set(data.keys()))

    def test_pipeline_called_with_correct_params(self, mock_pipeline):
        client.post(
            "/ask",
            json={"query": "pembrolizumab in MSI-H?", "n_results": 3, "min_score": 0.4},
        )
        mock_pipeline.assert_called_once_with(
            query="pembrolizumab in MSI-H?",
            mode="incremental",
            reldate=None,
            n_results=3,
            min_score=0.4,
        )

    def test_coverage_note_passed_through(self, mock_pipeline):
        mock_pipeline.return_value = _result(coverage_note="Corpus may lack relevant context.")
        data = client.post("/ask", json={"query": "test"}).json()
        assert data["coverage_note"] == "Corpus may lack relevant context."

    def test_pipeline_error_returns_500(self, mock_pipeline):
        mock_pipeline.side_effect = RuntimeError("ChromaDB unavailable")
        response = client.post("/ask", json={"query": "test"})
        assert response.status_code == 500

    def test_missing_query_returns_422(self):
        assert client.post("/ask", json={}).status_code == 422

    def test_n_results_below_min_returns_422(self):
        assert client.post("/ask", json={"query": "test", "n_results": 0}).status_code == 422

    def test_n_results_above_max_returns_422(self):
        assert client.post("/ask", json={"query": "test", "n_results": 21}).status_code == 422

    def test_invalid_mode_returns_422(self):
        assert client.post("/ask", json={"query": "test", "mode": "invalid"}).status_code == 422

    def test_min_score_out_of_range_returns_422(self):
        assert client.post("/ask", json={"query": "test", "min_score": 1.5}).status_code == 422

    def test_request_id_header_present(self, mock_pipeline):
        response = client.post("/ask", json={"query": "test"})
        assert "x-request-id" in response.headers


# ── Auth helpers ─────────────────────────────────────────────────────────────


@contextmanager
def _auth_override(api_keys: list[str]):
    """Override get_settings so verify_api_key sees a non-empty api_keys string."""
    mock_settings = MagicMock(spec=Settings)
    mock_settings.api_keys = ",".join(api_keys)  # str, not list (matches config field type)
    app.dependency_overrides[get_settings] = lambda: mock_settings
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_settings, None)


# POST /ask — auth (X-API-Key)
class TestAuth:
    def test_no_keys_configured_passes_through(self, mock_pipeline):
        # Default api_keys=[] — auth disabled, any call should succeed
        response = client.post("/ask", json={"query": "test"})
        assert response.status_code == 200

    def test_valid_key_returns_200(self, mock_pipeline):
        with _auth_override(["sk-test-abc"]):
            response = client.post(
                "/ask",
                json={"query": "test"},
                headers={"X-API-Key": "sk-test-abc"},
            )
            assert response.status_code == 200

    def test_missing_key_returns_401(self, mock_pipeline):
        with _auth_override(["sk-test-abc"]):
            response = client.post("/ask", json={"query": "test"})
            assert response.status_code == 401

    def test_invalid_key_returns_401(self, mock_pipeline):
        with _auth_override(["sk-test-abc"]):
            response = client.post(
                "/ask",
                json={"query": "test"},
                headers={"X-API-Key": "wrong-key"},
            )
            assert response.status_code == 401


# POST /ask — rate limiting
class TestRateLimit:
    def test_rate_limit_exceeded_returns_429(self, mock_pipeline):
        def _fake_check(request, func, in_middleware):
            mock_limit = MagicMock()
            mock_limit.error_message = ""
            request.state.view_rate_limit = (mock_limit, [])
            raise RateLimitExceeded(mock_limit)

        with (
            patch.object(limiter, "_check_request_limit", side_effect=_fake_check),
            patch.object(limiter, "_inject_headers", side_effect=lambda r, _: r),
        ):
            response = client.post("/ask", json={"query": "test"})
            assert response.status_code == 429
