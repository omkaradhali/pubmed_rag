"""
ingest.py — Fetch PubMed abstracts via NCBI E-utilities.

NCBI's esearch caps each individual call at 10,000 PMIDs, and search_pubmed()
pages past that internally via the Entrez history server (usehistory=y +
WebEnv/query_key). But PubMed specifically has a second, harder limit on top
of that: no single query can ever retrieve more than its first 9,999 matches,
however it's paginated (confirmed directly from NCBI's own error response,
not just the general E-utilities docs -- see PUBMED_HARD_RESULT_CAP). Once a
query's requested results exceed that, search_pubmed() falls back to
bisecting the date range into sub-queries that each fit under the cap, and
merges the results -- the standard workaround for this PubMed-specific
limit.

Public API:
    search_pubmed(query, max_results) -> list[str]             # returns PMIDs
    fetch_abstracts(pmids)            -> list[dict[str, str]]  # returns parsed records
    ingest(query, max_results)        -> list[dict[str, str]]  # search + fetch combined
    save_to_jsonl(records, path)      -> int                   # appends records to .jsonl file

Each parsed record contains:
    pmid              (str)       — PubMed unique ID
    title             (str)       — article title
    abstract          (str)       — full abstract text
    year              (str)       — 4-digit publication year, or "" if not found
    doi               (str)       — raw DOI string, e.g. "10.1038/s41591-024-01234-5", or ""
    doi_url           (str)       — clickable DOI link, e.g. "https://doi.org/10.1038/...", or ""
    pmc_id            (str)       — PubMed Central ID, e.g. "PMC11234567", or ""
    pmc_url           (str)       — clickable PMC link, or "" if no free full text
    authors           (list[str]) — author names as "LastName Initials",
                                    e.g. ["Smith JA", "Jones B"]
    journal           (str)       — full journal title, e.g. "Nature Medicine", or ""
    publication_types (list[str]) — e.g. ["Journal Article", "Randomized Controlled Trial"]
    mesh_terms        (list[str]) — NLM-assigned MeSH descriptor names, e.g. ["Breast Neoplasms"]
                                    Empty list for recently published articles not yet indexed.

doi_url and pmc_url are derived from doi/pmc_id at ingest time so downstream modules
(chunk, retrieve, generate, API) never need to reconstruct them.
"""

import datetime
import json
import logging
import math
import os
import time
import xml.etree.ElementTree as ET

import requests
from dotenv import load_dotenv
from requests.models import HTTPError

load_dotenv()

_logger = logging.getLogger(__name__)

# Constants

# Multi-specialty corpus support (docs/decisions/multi-specialty-corpus.md).
# Maps a short specialty name to its Entrez search string. Both entries use a
# MeSH disease/subject-matter heading
# (not a "field of study" heading like "Oncology[MeSH]" or "Neurosciences[MeSH]")
# deliberately: disease-category headings roll up thousands of subheadings
# hierarchically and index actual research output, while field-of-study
# headings are narrow administrative tags that return a tiny fraction of the
# relevant literature. Verified against a live NCBI query 2026-09-11: dropping
# to "Neurosciences[MeSH]" would have returned ~3K results over 5 years versus
# ~493K for "Nervous System Diseases[MeSH]" over the same window.
SPECIALTY_QUERIES: dict[str, str] = {
    "oncology": "Neoplasms[MeSH]",
    "neuroscience": "Nervous System Diseases[MeSH]",
}

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Batch size for efetch calls — NCBI can handle up to 200 PMIDs per request
EFETCH_BATCH_SIZE = 200

# Max PMIDs esearch will return per call, regardless of usehistory — pagination
# via retstart is what lets search_pubmed() walk past this per-call ceiling.
ESEARCH_MAX_PAGE_SIZE = 10000

# PubMed-specific hard ceiling on total results retrievable from a single
# query, however it's paginated: NCBI's esearch backend rejects any
# retstart > 9998 for the pubmed database with "'retstart' cannot be larger
# than 9998. For PubMed, ESearch can only retrieve the first 9,999 records
# matching the query." (confirmed live, not documented in the general
# E-utilities reference). Distinct from ESEARCH_MAX_PAGE_SIZE, which is the
# per-call retmax ceiling — this is the per-query total ceiling, and it's the
# one that actually binds for any PubMed query with more than ~10K matches.
# search_pubmed() falls back to date-range partitioning once this is exceeded.
PUBMED_HARD_RESULT_CAP = 9999

# Polite delay between batches — with API key: 10 req/sec, without: 3 req/sec
REQUEST_DELAY_WITH_API_KEY = 0.11  # seconds
REQUEST_DELAY_WITHOUT_API_KEY = 0.34  # seconds

# API Timeout set to 15 seconds.
SEARCH_MPIDS_API_TIMEOUT = 15
FETCH_ABSTRACT_TIMEOUT = 30


def _request_delay() -> float:
    """
    Return the appropriate request delay based on whether an API key is configured.
    With key: 0.11s (≤10 req/sec). Without: 0.34s (≤3 req/sec).
    """
    return (
        REQUEST_DELAY_WITH_API_KEY if os.getenv("NCBI_API_KEY") else REQUEST_DELAY_WITHOUT_API_KEY
    )


def _api_params() -> dict[str, str]:
    """
    Return a dict of common query params.
    If NCBI_API_KEY is set in .env, include it so we get 10 req/sec instead of 3.
    """
    params: dict[str, str] = {}

    api_key = os.getenv("NCBI_API_KEY", "")
    if api_key:
        params["api_key"] = api_key

    return params


def _esearch_request(params: dict) -> dict:
    """
    Perform one esearch GET request and return its "esearchresult" dict.

    Shared by every page of search_pubmed() so the request + error-handling
    logic exists in exactly one place, regardless of how many pages a search
    needs.

    Raises:
        requests.HTTPError: on a non-2xx response.
        ValueError: if the response JSON is missing expected fields.
    """
    try:
        response = requests.get(ESEARCH_URL, params=params, timeout=SEARCH_MPIDS_API_TIMEOUT)

        response.raise_for_status()

        data = response.json()

        return data.get("esearchresult", {})

    except HTTPError as err:
        raise HTTPError(f"API did not return a 2xx response: {err}") from err

    except ValueError as err:
        raise ValueError(f"Unexpected esearch response structure: {err}") from err


# Step 1: search
def search_pubmed(
    query: str,
    max_results: int = 10,
    reldate: int | None = None,
) -> list[str]:
    """
    Search PubMed and return a list of PMIDs matching query.

    esearch caps every individual response at ESEARCH_MAX_PAGE_SIZE (10,000)
    PMIDs. To return more than that, this opens an Entrez history-server
    session on the first call (usehistory=y) and pages through the rest with
    retstart, reusing the session's WebEnv/query_key instead of re-sending
    the query term — NCBI slices the same cached result set on each page
    rather than re-running the search.

    That pagination alone still isn't enough once max_results exceeds
    PUBMED_HARD_RESULT_CAP (9,999): PubMed rejects any retstart past 9998
    outright, so no amount of paging against a single query can retrieve
    more than its first 9,999 matches. Past that point, this transparently
    falls back to bisecting the date range into sub-queries that each fit
    under the cap (see _search_pubmed_by_date_partition) — same return type,
    just a different fetch strategy under the hood.

    Args:
        query:       Entrez search string, e.g. "colorectal cancer[Title/Abstract]"
        max_results: How many PMIDs to return, no longer capped at 10,000 —
                     paginates internally to satisfy any value up to the
                     query's total result count. Values needing more than
                     PUBMED_HARD_RESULT_CAP (9,999) results trigger the
                     date-partitioned fallback instead of retstart pagination.
        reldate:     If set, restrict results to articles indexed in the last N days.
                     Also the window the date-partitioned fallback bisects,
                     when max_results is large enough to need it.

    Returns:
        List of PMID strings, most-relevant first.

    Raises:
        requests.HTTPError: on a non-2xx response.
        ValueError: if the response JSON is missing expected fields.

    Request format example (first page, opens the history session):
        GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
            ?db=pubmed&term=<query>&retmax=<n>&retstart=0&usehistory=y
            &retmode=json&api_key=<key>

    Later pages reuse the session instead of the query term:
        GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
            ?db=pubmed&WebEnv=<webenv>&query_key=<querykey>&retmax=<n>
            &retstart=<offset>&retmode=json&api_key=<key>

    Sample Response:
        {
          "esearchresult": {
            "count": "698401",
            "webenv": "NCID_1_...",
            "querykey": "1",
            "idlist": ["12345678", "87654321", ...]
          }
        }
    """
    base_params = _api_params() | {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
    }

    if reldate is not None:
        base_params["reldate"] = reldate
        base_params["datetype"] = "edat"

    first_page_size = min(max_results, ESEARCH_MAX_PAGE_SIZE)
    first_page_params = base_params | {
        "retmax": first_page_size,
        "retstart": 0,
        "usehistory": "y",
    }
    result = _esearch_request(first_page_params)

    total_count = int(result.get("count", 0))
    target = min(max_results, total_count)

    if target > PUBMED_HARD_RESULT_CAP:
        # What we actually need exceeds PubMed's per-query ceiling -- no
        # amount of retstart pagination against this one query can reach it.
        # (Checking `target`, not `total_count`, matters: a query can have
        # far more than 9,999 total matches while max_results itself asks
        # for fewer -- that case still works fine via ordinary pagination
        # below, since we'd never actually page past the cap.)
        _logger.warning(
            "%r needs %d results, exceeding PubMed's %d-per-query ESearch "
            "retrieval cap — partitioning by date range instead of paging.",
            query,
            target,
            PUBMED_HARD_RESULT_CAP,
        )
        return _search_pubmed_by_date_partition(base_params, max_results, reldate)

    pmids: list[str] = list(result.get("idlist", []))
    web_env = result.get("webenv", "")
    query_key = result.get("querykey", "")

    # Nothing to page through: either the whole result fit in the first page,
    # or NCBI didn't hand back a history session (e.g. a zero-result search).
    retstart = len(pmids)

    while retstart < target and web_env and query_key:
        time.sleep(_request_delay())

        page_size = min(ESEARCH_MAX_PAGE_SIZE, target - retstart)
        page_params = base_params | {
            "retmax": page_size,
            "retstart": retstart,
            "WebEnv": web_env,
            "query_key": query_key,
        }
        page_result = _esearch_request(page_params)

        page_pmids = page_result.get("idlist", [])
        if not page_pmids:
            # Defensive: stop rather than loop forever if NCBI ever returns an
            # empty page before we've reached `target`.
            break

        pmids.extend(page_pmids)
        retstart += len(page_pmids)

    return pmids


def _resolve_date_bounds(reldate: int | None) -> tuple[datetime.date, datetime.date]:
    """
    Resolve the [mindate, maxdate] window to bisect when a query exceeds
    PubMed's per-query retrieval cap.

    Mirrors reldate's own semantics (last N days, ending today) so the
    partitioned fetch covers exactly the window the caller originally asked
    for. With no reldate (an unbounded, all-time query), falls back to a
    wide bound comfortably predating PubMed's earliest indexed records —
    bisection will still narrow it down quickly since only sub-ranges that
    actually contain matches recurse deeper.
    """
    today = datetime.date.today()
    if reldate is not None:
        return today - datetime.timedelta(days=reldate), today
    return datetime.date(1900, 1, 1), today


def _search_pubmed_by_date_partition(
    base_params: dict,
    max_results: int,
    reldate: int | None,
) -> list[str]:
    """
    Fetch PMIDs for a query whose result count exceeds PUBMED_HARD_RESULT_CAP.

    No single ESearch query can ever return more than its first 9,999
    matches for the pubmed database, however it's paginated — see
    PUBMED_HARD_RESULT_CAP. The workaround is to bisect the date range into
    sub-queries that each fit under the cap, fetch each directly, and merge.
    mindate/maxdate replace reldate for this path since bisection needs an
    addressable range to split, not a rolling "last N days" window.

    Does not honor max_results as an early-stopping budget mid-recursion —
    it always walks the full date range, then truncates once at the end.
    Bisection depth is driven by the query's actual result density, not by
    how many results the caller wants, so a small max_results against a
    huge corpus would still recurse the same way; that tradeoff is fine for
    this function's real use case (a full corpus build wants everything, so
    max_results is set correspondingly large) but means this path is not the
    fast one for "give me 50 of these 672,000 results" — ordinary pagination
    in search_pubmed() already handles that case without ever calling here.
    """
    mindate, maxdate = _resolve_date_bounds(reldate)

    # Bisection needs an addressable range: drop reldate (relative, not
    # addressable) in favor of the resolved mindate/maxdate below.
    date_params = {k: v for k, v in base_params.items() if k not in ("reldate", "datetype")}
    date_params["datetype"] = "edat"

    pmids = _bisect_date_range(date_params, mindate, maxdate)

    # Sub-ranges never overlap by construction, so duplicates shouldn't occur
    # in practice — dedupe defensively anyway (order-preserving) since a
    # correctness bug here would otherwise silently corrupt the corpus.
    seen: set[str] = set()
    deduped: list[str] = []
    for pmid in pmids:
        if pmid not in seen:
            seen.add(pmid)
            deduped.append(pmid)

    return deduped[:max_results]


def _bisect_date_range(
    date_params: dict,
    mindate: datetime.date,
    maxdate: datetime.date,
) -> list[str]:
    """
    Recursively bisect [mindate, maxdate] until each leaf's result count
    fits under PUBMED_HARD_RESULT_CAP, then fetch each leaf in one call.

    Each node costs one cheap retmax=0 probe call to get the sub-range's
    count; only sub-ranges over the cap recurse further, so total calls
    scale with how many leaves the corpus actually needs, not with the
    full date span.
    """
    if mindate > maxdate:
        return []

    time.sleep(_request_delay())
    probe_params = date_params | {
        "mindate": mindate.strftime("%Y/%m/%d"),
        "maxdate": maxdate.strftime("%Y/%m/%d"),
        "retmax": 0,
    }
    count = int(_esearch_request(probe_params).get("count", 0))

    if count == 0:
        return []

    if count <= PUBMED_HARD_RESULT_CAP:
        time.sleep(_request_delay())
        fetch_params = date_params | {
            "mindate": mindate.strftime("%Y/%m/%d"),
            "maxdate": maxdate.strftime("%Y/%m/%d"),
            "retmax": count,
            "retstart": 0,
        }
        return list(_esearch_request(fetch_params).get("idlist", []))

    if mindate == maxdate:
        # Can't bisect a single day any further -- NCBI's date filter has no
        # finer granularity than a day. Best-effort: take the first
        # PUBMED_HARD_RESULT_CAP for this day; the rest are genuinely
        # unreachable via ESearch, not a bug in this function.
        _logger.warning(
            "%s alone has %d results, exceeding the %d-per-query cap with no "
            "finer date range to split — truncating to the first %d.",
            mindate.isoformat(),
            count,
            PUBMED_HARD_RESULT_CAP,
            PUBMED_HARD_RESULT_CAP,
        )
        time.sleep(_request_delay())
        fetch_params = date_params | {
            "mindate": mindate.strftime("%Y/%m/%d"),
            "maxdate": maxdate.strftime("%Y/%m/%d"),
            "retmax": PUBMED_HARD_RESULT_CAP,
            "retstart": 0,
        }
        return list(_esearch_request(fetch_params).get("idlist", []))

    midpoint = mindate + (maxdate - mindate) // 2
    left = _bisect_date_range(date_params, mindate, midpoint)
    right = _bisect_date_range(date_params, midpoint + datetime.timedelta(days=1), maxdate)
    return left + right


# Step 2: fetch
def _parse_pubmed_xml(xml_text: str) -> list[dict]:
    """
    Parse a PubmedArticleSet XML blob into a list of record dicts.

    Each dict has:
        pmid              (str)       — PubMed unique ID
        title             (str)       — article title
        abstract          (str)       — full abstract (may be "" for articles without one)
        year              (str)       — 4-digit publication year, or "" if not found
        doi               (str)       — raw DOI string, or ""
        doi_url           (str)       — https://doi.org/{doi}, or ""
        pmc_id            (str)       — PubMed Central ID, e.g. "PMC11234567", or ""
        pmc_url           (str)       — https://...ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/, or ""
        authors           (list[str]) — ["LastName Initials", ...], or [] if no author list
        journal           (str)       — full journal title, or ""
        publication_types (list[str]) — ["Journal Article", "Randomized Controlled Trial", ...]
        mesh_terms        (list[str]) — NLM MeSH descriptor names, or [] if not yet indexed

    XPath selectors used:
        PMID:              .//PMID
        Title:             .//ArticleTitle
        Abstract:          .//AbstractText   (multiple for structured abstracts)
        Pub year:          .//PubDate/Year   (fallback: .//PubDate/MedlineDate)
        Article IDs:       .//ArticleIdList/ArticleId  (IdType="doi" and "pmc")
        Authors:           .//AuthorList/Author
        Journal:           .//Journal/Title
        Pub types:         .//PublicationTypeList/PublicationType
        MeSH:              .//MeshHeadingList/MeshHeading/DescriptorName
    """
    root = ET.fromstring(xml_text)

    records = []

    for article in root.findall(".//PubmedArticle"):
        # NOTE: always use `el is not None` when checking ElementTree find() results.
        # `if el` uses the element's child count as its truth value — an element with
        # no children (like <PMID>) is falsy even when .text is populated. This is a
        # known ET gotcha; future Python versions will raise an exception for `if el`.

        # PMID
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text.strip() if pmid_el is not None else ""

        # Title
        title_el = article.find(".//ArticleTitle")

        # Structured abstracts have multiple <AbstractText> sections (e.g. Background,
        # Methods, Results). Using itertext() to handle nested tags, joined with " ".
        title = " ".join(title_el.itertext()).strip() if title_el is not None else ""

        # Abstract — may have multiple <AbstractText> sections (structured abstracts)
        abstract_parts = [
            " ".join(el.itertext()).strip() for el in article.findall(".//AbstractText")
        ]
        abstract = " ".join(part for part in abstract_parts if part)

        # Publication year — try MedlineDate fallback for older records
        # Older articles use <MedlineDate> instead of <Year>. Its text looks like
        # "2003 Jan-Feb" — take the first 4 characters.
        year_el = article.find(".//PubDate/Year")
        if year_el is not None:
            year = year_el.text.strip()
        else:
            medline_el = article.find(".//PubDate/MedlineDate")
            year = medline_el.text[:4] if medline_el is not None else ""

        # Link-out identifiers
        # <ArticleIdList> holds identifiers assigned by different systems.
        # We extract two that are useful for direct linking:
        #
        #   IdType="doi"  — Digital Object Identifier, present on ~90% of modern
        #                   articles. doi.org is the canonical resolver.
        #   IdType="pmc"  — PubMed Central ID, present only when free full text
        #                   is available in PMC (~40-50% of PubMed records).
        #                   This is the "LinkOut — More Resources" link on PubMed.
        #
        # Sample XML structure:
        #   <ArticleIdList>
        #     <ArticleId IdType="pubmed">41980200</ArticleId>
        #     <ArticleId IdType="doi">10.1038/s41591-024-01234-5</ArticleId>
        #     <ArticleId IdType="pmc">PMC11234567</ArticleId>  ← absent if no free text
        #   </ArticleIdList>
        doi = ""
        pmc_id = ""
        for id_el in article.findall(".//ArticleIdList/ArticleId"):
            id_type = id_el.get("IdType", "")
            if id_type == "doi" and id_el.text:
                doi = id_el.text.strip()
            elif id_type == "pmc" and id_el.text:
                pmc_id = id_el.text.strip()

        # Derive full URLs here at ingest time so no downstream module ever needs
        # to reconstruct them. Empty string signals "not available" to the UI —
        # the frontend should hide the chip when the value is "".
        doi_url = f"https://doi.org/{doi}" if doi else ""
        pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/" if pmc_id else ""

        # Authors
        # <AuthorList> contains <Author> elements — either individual people with
        # <LastName>/<Initials>, or collective bodies with <CollectiveName>
        # (e.g. consortium groups like "TCGA Research Network").
        # We store "LastName Initials" for people and the collective name as-is.
        # Downstream citation rendering can apply "et al." logic from this list.
        authors: list[str] = []
        for author_el in article.findall(".//AuthorList/Author"):
            last_el = author_el.find("LastName")
            initials_el = author_el.find("Initials")
            collective_el = author_el.find("CollectiveName")
            if last_el is not None and last_el.text:
                initials = initials_el.text.strip() if initials_el is not None else ""
                name = f"{last_el.text.strip()} {initials}".strip()
                authors.append(name)
            elif collective_el is not None and collective_el.text:
                authors.append(collective_el.text.strip())

        # Journal
        # Full journal title (e.g. "Nature Medicine") from <Journal><Title>.
        # Prefer full title over <ISOAbbreviation> for readability in citations.
        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text.strip() if journal_el is not None else ""

        # Publication types
        # NLM assigns one or more publication type tags per article.
        # "Journal Article" is nearly always present; clinically useful values
        # include "Randomized Controlled Trial", "Review", "Meta-Analysis",
        # "Clinical Trial", "Case Reports", "Systematic Review".
        # Stored as a list so the UI and retrieval layer can filter by type.
        publication_types: list[str] = [
            el.text.strip()
            for el in article.findall(".//PublicationTypeList/PublicationType")
            if el.text
        ]

        # MeSH terms
        # Medical Subject Headings assigned by NLM curators after indexing.
        # Only <DescriptorName> is captured (not sub-qualifiers like "diagnosis"
        # or "drug therapy") to keep the list concise and query-friendly.
        # Important: recently published articles may have an empty list here —
        # NLM indexing typically lags publication by days to weeks.
        mesh_terms: list[str] = [
            el.text.strip()
            for el in article.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
            if el.text
        ]

        if pmid:  # skip malformed records with no PMID
            records.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "doi": doi,
                    "doi_url": doi_url,
                    "pmc_id": pmc_id,
                    "pmc_url": pmc_url,
                    "authors": authors,
                    "journal": journal,
                    "publication_types": publication_types,
                    "mesh_terms": mesh_terms,
                }
            )

    return records


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    """
    Fetch full abstract records for a list of PMIDs.

    Batches requests in groups of EFETCH_BATCH_SIZE to stay within NCBI rate limits.

    Args:
        pmids: List of PubMed ID strings.

    Returns:
        List of dicts: [{pmid, title, abstract, year, doi, doi_url, pmc_id, pmc_url,
                         authors, journal, publication_types, mesh_terms}]

    efetch Request looks like:
        GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
            ?db=pubmed&id=12345,67890&rettype=abstract&retmode=xml&api_key=<key>
        The id param is a comma-joined string of PMIDs.
    """
    if not pmids:
        return []

    all_records: list[dict[str, str]] = []
    # math.ceil gives the number of batches, including any partial final batch
    total_batches = math.ceil(len(pmids) / EFETCH_BATCH_SIZE)

    for i in range(0, len(pmids), EFETCH_BATCH_SIZE):
        batch = pmids[i : i + EFETCH_BATCH_SIZE]
        batch_num = i // EFETCH_BATCH_SIZE + 1
        _logger.info("Fetching batch %d/%d (%d PMIDs)...", batch_num, total_batches, len(batch))

        params = _api_params() | {
            "db": "pubmed",
            "id": ",".join(batch),
            "rettype": "abstract",
            "retmode": "xml",
        }
        try:
            response = requests.get(EFETCH_URL, params=params, timeout=FETCH_ABSTRACT_TIMEOUT)

            response.raise_for_status()

            records = _parse_pubmed_xml(response.text)

            all_records.extend(records)
        except HTTPError as err:
            raise HTTPError(f"Error calling API: {err}") from err

        # Polite delay between batches
        if i + EFETCH_BATCH_SIZE < len(pmids):
            time.sleep(_request_delay())

    return all_records


# Combined entry point
def ingest(
    query: str,
    max_results: int = 10,
    reldate: int | None = None,
    specialty: str | None = None,
) -> list[dict]:
    """
    Search PubMed and return parsed abstract records.

    Convenience wrapper: search_pubmed → fetch_abstracts.

    Args:
        query:       Entrez search string.
        max_results: Number of abstracts to fetch.
        reldate:     If set, restrict to articles indexed in the last N days.
        specialty:   If set, tag every returned record with this value (see
                     docs/decisions/multi-specialty-corpus.md). Stamped here
                     rather than left for the caller to
                     add, so a record's specialty is always set at the moment
                     it's known to have come from that specialty's query —
                     chunk.py, vectorstore.py, and retrieve.py all propagate
                     whatever is on the record without re-deriving it.

    Returns:
        List of dicts: {pmid, title, abstract, year, doi, doi_url, pmc_id, pmc_url,
                        authors, journal, publication_types, mesh_terms,
                        specialty (only present when the specialty arg is set)}.
    """
    pmids = search_pubmed(query, max_results=max_results, reldate=reldate)

    if not pmids:
        return []

    records = fetch_abstracts(pmids)

    if specialty is not None:
        for record in records:
            record["specialty"] = specialty

    return records


# Persistence
def save_to_jsonl(records: list[dict], path: str | os.PathLike) -> int:
    """
    Append records to a JSONL file — one JSON object per line.

    Opens in append mode so it is safe to call after a partial write or to
    accumulate records across multiple runs. Creates the file if it doesn't exist.

    ensure_ascii=False preserves non-ASCII characters (accented author names,
    special symbols) rather than escaping them to \\uXXXX sequences.

    Args:
        records: List of dicts to serialize (e.g. from fetch_abstracts).
        path:    Destination file path — created if absent.

    Returns:
        Number of records written in this call.
    """
    written = 0
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


# CLI entrypoint
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts and save to a JSONL file.")
    parser.add_argument(
        "--query",
        required=True,
        help='Entrez search string, e.g. "oncology[Title/Abstract]"',
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=500,
        help="Number of abstracts to fetch (default: 500)",
    )
    parser.add_argument(
        "--output",
        default="data/abstracts.jsonl",
        help="Output JSONL file path (default: data/abstracts.jsonl)",
    )
    args = parser.parse_args()

    _logger.info("Searching PubMed for: %r", args.query)
    pmids = search_pubmed(args.query, max_results=args.max_results)
    _logger.info("Found %d PMIDs", len(pmids))

    if not pmids:
        _logger.info("No results — exiting.")
        raise SystemExit(0)

    records = fetch_abstracts(pmids)
    _logger.info("Fetched %d records", len(records))

    saved = save_to_jsonl(records, args.output)
    _logger.info("Saved %d records → %s", saved, args.output)
