"""
ingest.py — Fetch PubMed abstracts via NCBI E-utilities.

NCBI's esearch will return up to 10,000 PMIDs in a single call.

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


def search_pubmed(
    query: str,
    max_results: int = 10,
    reldate: int | None = None,
) -> list[str]:
    """
    Search PubMed and return a list of PMIDs matching query.

    Args:
        query:       Entrez search string, e.g. "colorectal cancer[Title/Abstract]"
        max_results: How many PMIDs to return (NCBI cap: 10,000).
        reldate:     If set, restrict results to articles indexed in the last N days.

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

    if reldate is not None:
        params["reldate"] = reldate
        params["datetype"] = "edat"

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

        # ── Link-out identifiers ───────────────────────────────────────────────
        # <ArticleIdList> holds identifiers assigned by different systems.
        # We extract two that are useful for direct linking:
        #
        #   IdType="doi"  — Digital Object Identifier, present on ~90% of modern
        #                   articles. doi.org is the canonical resolver.
        #   IdType="pmc"  — PubMed Central ID, present only when free full text
        #                   is available in PMC (~40–50% of PubMed records).
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

        # ── Authors ───────────────────────────────────────────────────────────
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

        # ── Journal ───────────────────────────────────────────────────────────
        # Full journal title (e.g. "Nature Medicine") from <Journal><Title>.
        # Prefer full title over <ISOAbbreviation> for readability in citations.
        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text.strip() if journal_el is not None else ""

        # ── Publication types ─────────────────────────────────────────────────
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

        # ── MeSH terms ────────────────────────────────────────────────────────
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


# ── Combined entry point ───────────────────────────────────────────────────────


def ingest(
    query: str,
    max_results: int = 10,
    reldate: int | None = None,
) -> list[dict]:
    """
    Search PubMed and return parsed abstract records.

    Convenience wrapper: search_pubmed → fetch_abstracts.

    Args:
        query:       Entrez search string.
        max_results: Number of abstracts to fetch.
        reldate:     If set, restrict to articles indexed in the last N days.

    Returns:
        List of dicts: {pmid, title, abstract, year, doi, doi_url, pmc_id, pmc_url,
                        authors, journal, publication_types, mesh_terms}.
    """
    pmids = search_pubmed(query, max_results=max_results, reldate=reldate)

    if not pmids:
        return []

    return fetch_abstracts(pmids)


# ── Persistence ────────────────────────────────────────────────────────────────


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


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

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
