"""
Unit tests for chunk.py — v0.2 parent-child schema (D-042).

Uses small synthetic records — no disk I/O, no JSONL files.

Tests cover:
  * shape — every chunk has the v0.2 keys (chunk_id, chunk_role, parent_id)
  * always-emit-parent invariant (D-042 sub-decision 2)
  * parent_id linkage — every child's parent_id matches an emitted parent's chunk_id
  * ID format — {pmid}_p{i} for parents, {pmid}_p{i}_c{j} for children
  * role invariants — parents have parent_id=None, children have parent_id set
  * chunk_index / chunk_total semantics within each role group
"""

from pubmed_rag.chunk import (
    DEFAULT_CHILD_CHUNK_OVERLAP,
    DEFAULT_CHILD_CHUNK_SIZE,
    DEFAULT_PARENT_CHUNK_OVERLAP,
    DEFAULT_PARENT_CHUNK_SIZE,
    _build_child_splitter,
    _build_parent_splitter,
    chunk_record,
    chunk_records,
    split_parents_children,
)

_PARENT_SPLITTER = _build_parent_splitter(DEFAULT_PARENT_CHUNK_SIZE, DEFAULT_PARENT_CHUNK_OVERLAP)
_CHILD_SPLITTER = _build_child_splitter(DEFAULT_CHILD_CHUNK_SIZE, DEFAULT_CHILD_CHUNK_OVERLAP)


def _record(
    abstract: str = "This is a test abstract.",
    pmid: str = "12345",
    title: str = "Test Paper Title",
    year: str = "2023",
    doi: str = "10.1234/test.2023",
    pmc_id: str = "PMC9999999",
    authors: list | None = None,
    journal: str = "Journal of Testing",
    publication_types: list | None = None,
    mesh_terms: list | None = None,
) -> dict:
    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "year": year,
        "doi": doi,
        "doi_url": f"https://doi.org/{doi}" if doi else "",
        "pmc_id": pmc_id,
        "pmc_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/" if pmc_id else "",
        "authors": authors if authors is not None else ["Smith JA", "Jones B"],
        "journal": journal,
        "publication_types": publication_types
        if publication_types is not None
        else ["Journal Article"],
        "mesh_terms": mesh_terms if mesh_terms is not None else ["Neoplasms", "Humans"],
    }


_EXPECTED_KEYS = {
    "pmid",
    "title",
    "year",
    "doi",
    "doi_url",
    "pmc_id",
    "pmc_url",
    "authors",
    "journal",
    "publication_types",
    "mesh_terms",
    "text",
    "chunk_index",
    "chunk_total",
    # v0.2 (D-042)
    "chunk_id",
    "chunk_role",
    "parent_id",
}


# chunk_record
class TestChunkRecord:
    def test_empty_abstract_returns_empty_list(self):
        assert chunk_record(_record(abstract=""), _PARENT_SPLITTER, _CHILD_SPLITTER) == []

    def test_short_abstract_emits_one_parent_and_one_child(self):
        chunks = chunk_record(
            _record(abstract="Short single-sentence body."), _PARENT_SPLITTER, _CHILD_SPLITTER
        )
        roles = [c["chunk_role"] for c in chunks]
        assert roles.count("parent") == 1
        assert roles.count("child") == 1

    def test_required_keys_present_on_every_chunk(self):
        chunks = chunk_record(_record(), _PARENT_SPLITTER, _CHILD_SPLITTER)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert set(chunk.keys()) == _EXPECTED_KEYS

    def test_parent_emitted_before_its_children(self):
        chunks = chunk_record(_record(), _PARENT_SPLITTER, _CHILD_SPLITTER)
        # First chunk for any record must be a parent.
        assert chunks[0]["chunk_role"] == "parent"

    def test_parent_has_no_parent_id(self):
        chunks = chunk_record(_record(), _PARENT_SPLITTER, _CHILD_SPLITTER)
        parents = [c for c in chunks if c["chunk_role"] == "parent"]
        assert parents, "no parents emitted"
        for parent in parents:
            assert parent["parent_id"] is None

    def test_child_points_to_an_emitted_parent(self):
        chunks = chunk_record(_record(), _PARENT_SPLITTER, _CHILD_SPLITTER)
        parent_ids = {c["chunk_id"] for c in chunks if c["chunk_role"] == "parent"}
        children = [c for c in chunks if c["chunk_role"] == "child"]
        assert children, "no children emitted"
        for child in children:
            assert child["parent_id"] in parent_ids

    def test_id_format_parents(self):
        chunks = chunk_record(_record(pmid="99999"), _PARENT_SPLITTER, _CHILD_SPLITTER)
        parents = [c for c in chunks if c["chunk_role"] == "parent"]
        for i, parent in enumerate(parents):
            assert parent["chunk_id"] == f"99999_p{i}"

    def test_id_format_children_inside_their_parent(self):
        chunks = chunk_record(_record(pmid="99999"), _PARENT_SPLITTER, _CHILD_SPLITTER)
        # For each parent, its children's chunk_id must start with parent_id + "_c".
        parents = [c for c in chunks if c["chunk_role"] == "parent"]
        for parent in parents:
            siblings = [
                c
                for c in chunks
                if c["chunk_role"] == "child" and c["parent_id"] == parent["chunk_id"]
            ]
            for i, child in enumerate(siblings):
                assert child["chunk_id"] == f"{parent['chunk_id']}_c{i}"

    def test_metadata_preserved_across_all_chunks(self):
        chunks = chunk_record(
            _record(pmid="99999", title="My Paper", year="2024"),
            _PARENT_SPLITTER,
            _CHILD_SPLITTER,
        )
        for chunk in chunks:
            assert chunk["pmid"] == "99999"
            assert chunk["title"] == "My Paper"
            assert chunk["year"] == "2024"

    def test_title_prepended_in_first_parent(self):
        chunks = chunk_record(
            _record(title="Important Title", abstract="Abstract body."),
            _PARENT_SPLITTER,
            _CHILD_SPLITTER,
        )
        first_parent = next(c for c in chunks if c["chunk_role"] == "parent")
        assert "Important Title" in first_parent["text"]

    def test_long_abstract_produces_multiple_children(self):
        long_abstract = "Word " * 400  # ~2000 chars — exceeds DEFAULT_CHILD_CHUNK_SIZE=1500
        chunks = chunk_record(_record(abstract=long_abstract), _PARENT_SPLITTER, _CHILD_SPLITTER)
        children = [c for c in chunks if c["chunk_role"] == "child"]
        assert len(children) > 1

    def test_child_indices_are_zero_based_and_sequential_per_parent(self):
        long_abstract = "Word " * 400
        chunks = chunk_record(_record(abstract=long_abstract), _PARENT_SPLITTER, _CHILD_SPLITTER)
        for parent in (c for c in chunks if c["chunk_role"] == "parent"):
            siblings = [
                c
                for c in chunks
                if c["chunk_role"] == "child" and c["parent_id"] == parent["chunk_id"]
            ]
            for i, child in enumerate(siblings):
                assert child["chunk_index"] == i
                assert child["chunk_total"] == len(siblings)


# chunk_records
class TestChunkRecords:
    def test_empty_input_returns_empty_list(self):
        assert chunk_records([]) == []

    def test_multiple_records_produce_flat_list_with_both_roles(self):
        records = [_record(pmid="1"), _record(pmid="2")]
        chunks = chunk_records(records)
        roles = {c["chunk_role"] for c in chunks}
        assert roles == {"parent", "child"}

    def test_records_with_empty_abstracts_are_skipped(self):
        records = [_record(abstract=""), _record(abstract="Real abstract body.")]
        chunks = chunk_records(records)
        # Only the non-empty record contributes.
        pmids = {c["pmid"] for c in chunks}
        assert pmids == {"12345"}

    def test_all_chunk_keys_present_in_output(self):
        chunks = chunk_records([_record()])
        for chunk in chunks:
            assert set(chunk.keys()) == _EXPECTED_KEYS

    def test_every_pmid_has_at_least_one_parent_and_one_child(self):
        records = [_record(pmid="a"), _record(pmid="b")]
        chunks = chunk_records(records)
        for pmid in ("a", "b"):
            sub = [c for c in chunks if c["pmid"] == pmid]
            roles = {c["chunk_role"] for c in sub}
            assert roles == {"parent", "child"}


# split_parents_children
class TestSplitParentsChildren:
    def test_split_preserves_total_count(self):
        chunks = chunk_records([_record()])
        parents, children = split_parents_children(chunks)
        assert len(parents) + len(children) == len(chunks)

    def test_split_partitions_by_role(self):
        chunks = chunk_records([_record()])
        parents, children = split_parents_children(chunks)
        assert all(c["chunk_role"] == "parent" for c in parents)
        assert all(c["chunk_role"] == "child" for c in children)

    def test_empty_input_returns_two_empty_lists(self):
        parents, children = split_parents_children([])
        assert parents == []
        assert children == []
