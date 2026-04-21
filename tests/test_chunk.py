"""
Unit tests for chunk.py.

Uses small synthetic records — no disk I/O, no JSONL files.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from pubmed_rag.chunk import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_record,
    chunk_records,
)

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
)


def _record(
    abstract: str = "This is a test abstract.",
    pmid: str = "12345",
    title: str = "Test Paper Title",
    year: str = "2023",
) -> dict:
    return {"pmid": pmid, "title": title, "abstract": abstract, "year": year}


# ── chunk_record ───────────────────────────────────────────────────────────────


class TestChunkRecord:
    def test_empty_abstract_returns_empty_list(self):
        assert chunk_record(_record(abstract=""), _SPLITTER) == []

    def test_required_keys_present(self):
        chunks = chunk_record(_record(), _SPLITTER)
        assert len(chunks) >= 1
        expected = {"pmid", "title", "year", "text", "chunk_index", "chunk_total"}
        for chunk in chunks:
            assert set(chunk.keys()) == expected

    def test_metadata_preserved_across_all_chunks(self):
        chunks = chunk_record(_record(pmid="99999", title="My Paper", year="2024"), _SPLITTER)
        for chunk in chunks:
            assert chunk["pmid"] == "99999"
            assert chunk["title"] == "My Paper"
            assert chunk["year"] == "2024"

    def test_chunk_index_sequential_zero_based(self):
        long_abstract = "Word " * 300
        chunks = chunk_record(_record(abstract=long_abstract), _SPLITTER)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    def test_chunk_total_matches_actual_count(self):
        long_abstract = "Word " * 300
        chunks = chunk_record(_record(abstract=long_abstract), _SPLITTER)
        for chunk in chunks:
            assert chunk["chunk_total"] == len(chunks)

    def test_short_abstract_produces_single_chunk(self):
        chunks = chunk_record(_record(abstract="Short."), _SPLITTER)
        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["chunk_total"] == 1

    def test_title_prepended_in_first_chunk(self):
        chunks = chunk_record(
            _record(title="Important Title", abstract="Abstract body."), _SPLITTER
        )
        assert "Important Title" in chunks[0]["text"]

    def test_long_abstract_produces_multiple_chunks(self):
        long_abstract = "Word " * 300  # ~1500 chars — exceeds DEFAULT_CHUNK_SIZE=1000
        chunks = chunk_record(_record(abstract=long_abstract), _SPLITTER)
        assert len(chunks) > 1


# ── chunk_records ──────────────────────────────────────────────────────────────


class TestChunkRecords:
    def test_empty_input_returns_empty_list(self):
        assert chunk_records([]) == []

    def test_multiple_records_produce_flat_list(self):
        records = [_record(pmid="1"), _record(pmid="2")]
        chunks = chunk_records(records)
        assert len(chunks) >= 2

    def test_records_with_empty_abstracts_are_skipped(self):
        records = [_record(abstract=""), _record(abstract="Real abstract.")]
        chunks = chunk_records(records)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk["text"]

    def test_all_chunk_keys_present_in_output(self):
        chunks = chunk_records([_record()])
        expected = {"pmid", "title", "year", "text", "chunk_index", "chunk_total"}
        for chunk in chunks:
            assert set(chunk.keys()) == expected
