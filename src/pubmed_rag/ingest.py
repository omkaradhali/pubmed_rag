"""
ingest.py — Fetch PubMed abstracts via NCBI E-utilities.

Public API:
    search_pubmed(query, max_results) -> list[str]             # returns PMIDs
    fetch_abstracts(pmids)            -> list[dict[str, str]]  # returns parsed records
    ingest(query, max_results)        -> list[dict[str, str]]  # search + fetch combined
"""

import os
import time
import xml.etree.ElementTree as ET

import requests
from dotenv import load_dotenv
from requests.models import HTTPError

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Batch size for efetch calls — NCBI can handle up to 200 PMIDs per request
EFETCH_BATCH_SIZE = 200

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


# ── Step 1: search ─────────────────────────────────────────────────────────────


def search_pubmed(query: str, max_results: int = 10) -> list[str]:
    """
    Search PubMed and return a list of PMIDs matching query.

    Args:
        query: Entrez search string, e.g. "colorectal cancer[Title/Abstract]"
        max_results: How many PMIDs to return (NCBI cap: 10,000).

    Returns:
        List of PMID strings, most-relevant first.

    Raises:
        requests.HTTPError: on a non-2xx response.
        ValueError: if the response JSON is missing expected fields.

    Request format example:
        GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
            ?db=pubmed&term=<query>&retmax=<n>&retmode=json&api_key=<key>

    Sample Response:
        {
          "esearchresult": {
            "idlist": ["12345678", "87654321", ...]
          }
        }
    """
    pmids: list[str] = []

    params = _api_params() | {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }

    try:
        response = requests.get(ESEARCH_URL, params=params, timeout=SEARCH_MPIDS_API_TIMEOUT)

        response.raise_for_status()

        data = response.json()

        pmids = data.get("esearchresult", {}).get("idlist", [])

    except HTTPError as err:
        raise HTTPError(f"API did not return a 2xx response: {err}") from err

    except ValueError as err:
        raise ValueError(f"Unexpected esearch response structure: {err}") from err

    return pmids


# ── Step 2: fetch ──────────────────────────────────────────────────────────────


def _parse_pubmed_xml(xml_text: str) -> list[dict[str, str]]:
    """
    Parse a PubmedArticleSet XML blob into a list of record dicts.

    Each dict has:
        pmid     (str) — PubMed unique ID
        title    (str) — article title
        abstract (str) — full abstract (may be "" for articles without one)
        year     (str) — 4-digit publication year, or "" if not found

    XPath selectors in the MEDLINE XML schema:
        PMID:            .//PMID
        Title:           .//ArticleTitle
        Abstract parts:  .//AbstractText   (multiple for structured abstracts)
        Pub year:        .//PubDate/Year   (fallback: .//PubDate/MedlineDate)
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

        if pmid:  # skip malformed records with no PMID
            records.append({"pmid": pmid, "title": title, "abstract": abstract, "year": year})

    return records


def fetch_abstracts(pmids: list[str]) -> list[dict[str, str]]:
    """
    Fetch full abstract records for a list of PMIDs.

    Batches requests in groups of EFETCH_BATCH_SIZE to stay within NCBI rate limits.

    Args:
        pmids: List of PubMed ID strings.

    Returns:
        List of dicts: [{pmid, title, abstract, year}]

    efetch Request looks like:
        GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
            ?db=pubmed&id=12345,67890&rettype=abstract&retmode=xml&api_key=<key>
        The id param is a comma-joined string of PMIDs.
    """
    if not pmids:
        return []

    all_records: list[dict[str, str]] = []

    for i in range(0, len(pmids), EFETCH_BATCH_SIZE):
        batch = pmids[i : i + EFETCH_BATCH_SIZE]

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


# ── Combined entry point ───────────────────────────────────────────────────────


def ingest(query: str, max_results: int = 10) -> list[dict[str, str]]:
    """
    Search PubMed and return parsed abstract records.

    Convenience wrapper: search_pubmed → fetch_abstracts.

    Args:
        query:       Entrez search string.
        max_results: Number of abstracts to fetch.

    Returns:
        List of dicts: {pmid, title, abstract, year}.
    """
    pmids = search_pubmed(query, max_results=max_results)

    if not pmids:
        return []

    return fetch_abstracts(pmids)
