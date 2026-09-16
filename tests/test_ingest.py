"""
Unit tests for ingest.py — Entrez history-server pagination in search_pubmed(),
plus the date-range-partition fallback for queries that exceed PubMed's
9,999-per-query ESearch retrieval cap.

requests.get is mocked throughout — these tests exercise the pagination and
partitioning logic (when to stop, how retstart/WebEnv/query_key get threaded
through pages, when/how the date-bisection fallback triggers), not live NCBI
behavior. time.sleep is patched so tests don't actually wait out the
rate-limit delay between calls.
"""

import datetime
from unittest.mock import MagicMock, patch

import pytest

from pubmed_rag import ingest as ingest_module
from pubmed_rag.ingest import _bisect_date_range, search_pubmed


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


def _date_range_responder(range_data: dict[tuple[str, str], tuple[int, list[str]]]):
    """
    Build a requests.get side_effect for date-partitioned calls.

    range_data maps (mindate_str, maxdate_str) -> (count, idlist) for that
    exact sub-range. A probe call (retmax=0) always gets idlist=[] regardless
    of what's configured, matching real esearch behavior; any other call
    (a leaf fetch) gets the configured idlist.
    """

    def _responder(*args, **kwargs):
        params = kwargs["params"]
        key = (params["mindate"], params["maxdate"])
        count, idlist = range_data[key]
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "esearchresult": {
                "count": str(count),
                "idlist": [] if params.get("retmax") == 0 else idlist,
            }
        }
        return response

    return _responder


class TestResolveDateBounds:
    def test_with_reldate_returns_window_ending_today(self):
        today = datetime.date.today()
        mindate, maxdate = ingest_module._resolve_date_bounds(30)
        assert maxdate == today
        assert mindate == today - datetime.timedelta(days=30)

    def test_without_reldate_returns_wide_fallback_ending_today(self):
        today = datetime.date.today()
        mindate, maxdate = ingest_module._resolve_date_bounds(None)
        assert maxdate == today
        assert mindate == datetime.date(1900, 1, 1)


class TestBisectDateRange:
    def test_single_leaf_under_cap_fetches_directly_no_bisection(self, monkeypatch):
        monkeypatch.setattr(ingest_module, "PUBMED_HARD_RESULT_CAP", 5)
        mindate = datetime.date(2020, 1, 1)
        maxdate = datetime.date(2020, 1, 10)

        range_data = {
            ("2020/01/01", "2020/01/10"): (3, ["A", "B", "C"]),
        }
        with (
            patch("pubmed_rag.ingest.requests.get", side_effect=_date_range_responder(range_data)),
            patch("pubmed_rag.ingest.time.sleep"),
        ):
            pmids = _bisect_date_range({}, mindate, maxdate)

        assert pmids == ["A", "B", "C"]

    def test_bisects_when_over_cap_and_merges_left_then_right(self, monkeypatch):
        monkeypatch.setattr(ingest_module, "PUBMED_HARD_RESULT_CAP", 2)
        mindate = datetime.date(2020, 1, 1)
        maxdate = datetime.date(2020, 1, 4)
        # midpoint = Jan 1 + (4-1)//2 days = Jan 2 -> left [1,2], right [3,4].
        # Both halves fit under the cap so this isolates a single bisection
        # step; multi-level recursion is covered separately below.
        range_data = {
            ("2020/01/01", "2020/01/04"): (5, []),  # over cap, triggers bisection
            ("2020/01/01", "2020/01/02"): (2, ["A", "B"]),
            ("2020/01/03", "2020/01/04"): (2, ["C", "D"]),
        }

        with (
            patch("pubmed_rag.ingest.requests.get", side_effect=_date_range_responder(range_data)),
            patch("pubmed_rag.ingest.time.sleep"),
        ):
            pmids = _bisect_date_range({}, mindate, maxdate)

        assert pmids == ["A", "B", "C", "D"]

    def test_recurses_multiple_levels(self, monkeypatch):
        monkeypatch.setattr(ingest_module, "PUBMED_HARD_RESULT_CAP", 2)
        mindate = datetime.date(2020, 1, 1)
        maxdate = datetime.date(2020, 1, 4)
        # Level 0: [1,4] count=5 -> over cap -> bisect at Jan 2 -> [1,2], [3,4]
        # Level 1 left [1,2]: count=2 -> at cap, fetch directly
        # Level 1 right [3,4]: count=3 -> over cap -> bisect at Jan 3 -> [3,3], [4,4]
        # Level 2 [3,3]: count=1 -> fetch directly
        # Level 2 [4,4]: count=2 -> fetch directly
        range_data = {
            ("2020/01/01", "2020/01/04"): (5, []),
            ("2020/01/01", "2020/01/02"): (2, ["A", "B"]),
            ("2020/01/03", "2020/01/04"): (3, []),
            ("2020/01/03", "2020/01/03"): (1, ["C"]),
            ("2020/01/04", "2020/01/04"): (2, ["D", "E"]),
        }

        with (
            patch("pubmed_rag.ingest.requests.get", side_effect=_date_range_responder(range_data)),
            patch("pubmed_rag.ingest.time.sleep"),
        ):
            pmids = _bisect_date_range({}, mindate, maxdate)

        assert pmids == ["A", "B", "C", "D", "E"]

    def test_zero_count_leaf_returns_empty_without_fetch_call(self, monkeypatch):
        monkeypatch.setattr(ingest_module, "PUBMED_HARD_RESULT_CAP", 5)
        mindate = datetime.date(2020, 1, 1)
        maxdate = datetime.date(2020, 1, 10)
        range_data = {("2020/01/01", "2020/01/10"): (0, [])}

        with (
            patch(
                "pubmed_rag.ingest.requests.get", side_effect=_date_range_responder(range_data)
            ) as mock_get,
            patch("pubmed_rag.ingest.time.sleep"),
        ):
            pmids = _bisect_date_range({}, mindate, maxdate)

        assert pmids == []
        mock_get.assert_called_once()  # just the probe, no wasted fetch call

    def test_single_day_over_cap_truncates_instead_of_infinite_recursion(self, monkeypatch):
        # A day that alone exceeds the cap can't be bisected any finer --
        # must truncate rather than recurse forever (mindate == maxdate).
        monkeypatch.setattr(ingest_module, "PUBMED_HARD_RESULT_CAP", 2)
        single_day = datetime.date(2020, 1, 1)
        range_data = {
            ("2020/01/01", "2020/01/01"): (5, ["A", "B"]),  # truncated fetch result
        }

        with (
            patch(
                "pubmed_rag.ingest.requests.get", side_effect=_date_range_responder(range_data)
            ) as mock_get,
            patch("pubmed_rag.ingest.time.sleep"),
        ):
            pmids = _bisect_date_range({}, single_day, single_day)

        assert pmids == ["A", "B"]
        # One probe + one truncated fetch -- no recursive calls attempted.
        assert mock_get.call_count == 2


class TestSearchPubmedDatePartitionIntegration:
    """search_pubmed()'s own decision about whether to trigger the
    date-partition fallback at all, exercised end-to-end."""

    def test_large_target_triggers_date_partition_not_retstart_paging(self, monkeypatch):
        monkeypatch.setattr(ingest_module, "PUBMED_HARD_RESULT_CAP", 3)
        today = datetime.date.today()

        # First call: count exceeds the (monkeypatched) cap via max_results.
        first_page = _esearch_response(idlist=["x"] * 3, count=100, webenv="WE1", querykey="1")

        range_data = {
            (today.strftime("%Y/%m/%d"), today.strftime("%Y/%m/%d")): (2, ["P1", "P2"]),
        }
        responder = _date_range_responder(range_data)

        calls = {"n": 0}

        def _combined_side_effect(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return first_page
            return responder(*args, **kwargs)

        with (
            patch("pubmed_rag.ingest.requests.get", side_effect=_combined_side_effect),
            patch("pubmed_rag.ingest.time.sleep"),
        ):
            # reldate=0 -> mindate == maxdate == today, single-leaf fetch.
            pmids = search_pubmed("huge query", max_results=100, reldate=0)

        assert pmids == ["P1", "P2"]

    def test_small_max_results_against_huge_total_count_skips_partition(self, monkeypatch):
        # total_count is enormous, but max_results is small enough that
        # ordinary retstart pagination never needs to exceed the cap --
        # must NOT trigger the (expensive) date-partition fallback.
        monkeypatch.setattr(ingest_module, "PUBMED_HARD_RESULT_CAP", 9999)
        monkeypatch.setattr(ingest_module, "ESEARCH_MAX_PAGE_SIZE", 10000)

        page1 = _esearch_response(idlist=["1", "2", "3"], count=700_000, webenv="WE1", querykey="1")

        with patch("pubmed_rag.ingest.requests.get") as mock_get:
            mock_get.return_value = page1
            pmids = search_pubmed("huge query", max_results=3)

        assert pmids == ["1", "2", "3"]
        # Only the one call -- confirms no mindate/maxdate probe was ever sent.
        mock_get.assert_called_once()
        assert "mindate" not in mock_get.call_args.kwargs["params"]

    def test_date_partition_result_deduped_and_truncated_to_max_results(self, monkeypatch):
        monkeypatch.setattr(ingest_module, "PUBMED_HARD_RESULT_CAP", 3)
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)

        first_page = _esearch_response(idlist=["x"] * 3, count=100, webenv="WE1", querykey="1")

        # midpoint of [yesterday, today] (a 2-day span) is yesterday itself,
        # so bisection splits into [yesterday, yesterday] and [today, today].
        range_data = {
            (yesterday.strftime("%Y/%m/%d"), today.strftime("%Y/%m/%d")): (6, []),
            (yesterday.strftime("%Y/%m/%d"), yesterday.strftime("%Y/%m/%d")): (
                3,
                ["A", "B", "DUP"],
            ),
            (today.strftime("%Y/%m/%d"), today.strftime("%Y/%m/%d")): (3, ["DUP", "C", "D"]),
        }
        responder = _date_range_responder(range_data)

        calls = {"n": 0}

        def _combined_side_effect(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return first_page
            return responder(*args, **kwargs)

        with (
            patch("pubmed_rag.ingest.requests.get", side_effect=_combined_side_effect),
            patch("pubmed_rag.ingest.time.sleep"),
        ):
            pmids = search_pubmed("huge query", max_results=4, reldate=1)

        # Merged: A, B, DUP, DUP, C, D -> deduped: A, B, DUP, C, D -> truncated to 4.
        assert pmids == ["A", "B", "DUP", "C"]
