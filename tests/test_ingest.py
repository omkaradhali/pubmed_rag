"""
Unit tests for ingest.py — Entrez history-server pagination in search_pubmed().

requests.get is mocked throughout — these tests exercise the pagination logic
(when to stop, how retstart/WebEnv/query_key get threaded through pages), not
live NCBI behavior. time.sleep is patched so tests don't actually wait out the
rate-limit delay between pages.
"""

from unittest.mock import MagicMock, patch

import pytest

from pubmed_rag import ingest as ingest_module
from pubmed_rag.ingest import search_pubmed


def _esearch_response(
    idlist: list[str], count: int, webenv: str = "", querykey: str = ""
) -> MagicMock:
    """Build a mocked requests.Response for one esearch call."""
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "esearchresult": {
            "idlist": idlist,
            "count": str(count),
            "webenv": webenv,
            "querykey": querykey,
        }
    }
    return response


@pytest.fixture(autouse=True)
def _no_real_sleep():
    """Every test in this module runs the pagination loop without real delays."""
    with patch("pubmed_rag.ingest.time.sleep"):
        yield


class TestSearchPubmedSinglePage:
    def test_result_fits_in_first_page_makes_one_call(self):
        with patch("pubmed_rag.ingest.requests.get") as mock_get:
            mock_get.return_value = _esearch_response(
                idlist=["1", "2", "3"], count=3, webenv="WE1", querykey="1"
            )
            pmids = search_pubmed("test query", max_results=10)

        assert pmids == ["1", "2", "3"]
        mock_get.assert_called_once()

    def test_empty_result_returns_empty_list(self):
        with patch("pubmed_rag.ingest.requests.get") as mock_get:
            mock_get.return_value = _esearch_response(idlist=[], count=0)
            pmids = search_pubmed("no matches")

        assert pmids == []
        mock_get.assert_called_once()

    def test_first_call_opens_history_session(self):
        with patch("pubmed_rag.ingest.requests.get") as mock_get:
            mock_get.return_value = _esearch_response(
                idlist=["1"], count=1, webenv="WE1", querykey="1"
            )
            search_pubmed("test query", max_results=1)

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["usehistory"] == "y"
        assert kwargs["params"]["retstart"] == 0

    def test_reldate_forwarded_to_every_call(self):
        with patch("pubmed_rag.ingest.requests.get") as mock_get:
            mock_get.return_value = _esearch_response(
                idlist=["1"], count=1, webenv="WE1", querykey="1"
            )
            search_pubmed("test query", max_results=1, reldate=30)

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["reldate"] == 30
        assert kwargs["params"]["datetype"] == "edat"


class TestSearchPubmedPagination:
    def test_pages_past_first_page_size_cap(self, monkeypatch):
        # Shrink the page-size cap so a small result set still exercises real
        # pagination without simulating thousands of PMIDs.
        monkeypatch.setattr(ingest_module, "ESEARCH_MAX_PAGE_SIZE", 3)

        page1 = _esearch_response(idlist=["1", "2", "3"], count=7, webenv="WE1", querykey="1")
        page2 = _esearch_response(idlist=["4", "5", "6"], count=7, webenv="WE1", querykey="1")
        page3 = _esearch_response(idlist=["7"], count=7, webenv="WE1", querykey="1")

        with patch("pubmed_rag.ingest.requests.get") as mock_get:
            mock_get.side_effect = [page1, page2, page3]
            pmids = search_pubmed("test query", max_results=7)

        assert pmids == ["1", "2", "3", "4", "5", "6", "7"]
        assert mock_get.call_count == 3

    def test_later_pages_reuse_webenv_and_querykey_not_term(self, monkeypatch):
        monkeypatch.setattr(ingest_module, "ESEARCH_MAX_PAGE_SIZE", 2)

        page1 = _esearch_response(idlist=["1", "2"], count=4, webenv="WE1", querykey="1")
        page2 = _esearch_response(idlist=["3", "4"], count=4, webenv="WE1", querykey="1")

        with patch("pubmed_rag.ingest.requests.get") as mock_get:
            mock_get.side_effect = [page1, page2]
            search_pubmed("test query", max_results=4)

        second_call_params = mock_get.call_args_list[1].kwargs["params"]
        assert second_call_params["WebEnv"] == "WE1"
        assert second_call_params["query_key"] == "1"
        assert second_call_params["retstart"] == 2
        assert "usehistory" not in second_call_params

    def test_stops_at_max_results_short_of_total_count(self, monkeypatch):
        # Total corpus has far more than we asked for — only page up to
        # max_results, never up to the full `count`.
        monkeypatch.setattr(ingest_module, "ESEARCH_MAX_PAGE_SIZE", 2)

        page1 = _esearch_response(idlist=["1", "2"], count=100, webenv="WE1", querykey="1")
        page2 = _esearch_response(idlist=["3"], count=100, webenv="WE1", querykey="1")

        with patch("pubmed_rag.ingest.requests.get") as mock_get:
            mock_get.side_effect = [page1, page2]
            pmids = search_pubmed("test query", max_results=3)

        assert pmids == ["1", "2", "3"]
        assert mock_get.call_count == 2

    def test_sleeps_between_pages_not_before_first_call(self, monkeypatch):
        monkeypatch.setattr(ingest_module, "ESEARCH_MAX_PAGE_SIZE", 2)

        page1 = _esearch_response(idlist=["1", "2"], count=4, webenv="WE1", querykey="1")
        page2 = _esearch_response(idlist=["3", "4"], count=4, webenv="WE1", querykey="1")

        with (
            patch("pubmed_rag.ingest.requests.get") as mock_get,
            patch("pubmed_rag.ingest.time.sleep") as mock_sleep,
        ):
            mock_get.side_effect = [page1, page2]
            search_pubmed("test query", max_results=4)

        mock_sleep.assert_called_once()

    def test_empty_page_before_target_reached_stops_instead_of_looping(self, monkeypatch):
        # Defensive case: NCBI hands back fewer PMIDs than `count` promised.
        # Pagination must stop, not spin forever re-requesting an empty page.
        monkeypatch.setattr(ingest_module, "ESEARCH_MAX_PAGE_SIZE", 2)

        page1 = _esearch_response(idlist=["1", "2"], count=10, webenv="WE1", querykey="1")
        page2 = _esearch_response(idlist=[], count=10, webenv="WE1", querykey="1")

        with patch("pubmed_rag.ingest.requests.get") as mock_get:
            mock_get.side_effect = [page1, page2]
            pmids = search_pubmed("test query", max_results=10)

        assert pmids == ["1", "2"]
        assert mock_get.call_count == 2
