"""Unit tests for the curriculum/chunk loader.

These tests don't hit OpenAI — they verify that the parser turns raw
spreadsheet rows and JSON into the expected schemas.
"""
import pytest

from app.data_loader import (
    _parse_lo_text,
    _parse_domain_label,
    load_chunks,
    load_learning_outcomes,
)


class TestLOTextParser:
    def test_pulls_id_from_learning_outcome_prefix(self):
        lo_id, text = _parse_lo_text(
            "Learning Outcome 6.5.3.1.1: Describing the moon's phases."
        )
        assert lo_id == "6.5.3.1.1"
        assert text == "Describing the moon's phases."

    def test_pulls_id_without_prefix(self):
        lo_id, text = _parse_lo_text("6.5.2.1.1 Exploring matter.")
        assert lo_id == "6.5.2.1.1"
        assert "matter" in text

    def test_handles_no_colon(self):
        lo_id, text = _parse_lo_text("6.5.2.1.3 Understanding chemical reactions")
        assert lo_id == "6.5.2.1.3"
        assert text == "Understanding chemical reactions"


class TestDomainParser:
    def test_splits_domain_and_subdomain(self):
        d, s = _parse_domain_label(
            "Domain 2: Physical Sciences. Subdomain 2.1: Matter and its Interactions"
        )
        assert d == "Domain 2: Physical Sciences"
        assert s == "Subdomain 2.1: Matter and its Interactions"

    def test_handles_typo_in_source(self):
        # Real data has "Ecosystemsand" (no space) — should still split.
        d, s = _parse_domain_label(
            "Domain 1: Life Sciences. Subdomain 1.3: Ecosystemsand Their Interactions"
        )
        assert d == "Domain 1: Life Sciences"
        assert "Ecosystemsand" in s


class TestLoadLearningOutcomes:
    def test_loads_all_33_los(self):
        los = load_learning_outcomes("data/LO.xlsx")
        assert len(los) == 33

    def test_every_lo_has_id_and_text(self):
        los = load_learning_outcomes("data/LO.xlsx")
        for lo in los:
            assert lo.lo_id
            assert lo.text
            assert lo.domain
            assert lo.subdomain

    def test_lo_ids_are_dotted_codes(self):
        los = load_learning_outcomes("data/LO.xlsx")
        for lo in los:
            parts = lo.lo_id.split(".")
            assert len(parts) >= 4
            assert all(p.isdigit() for p in parts)


class TestLoadChunks:
    def test_loads_all_chunks(self):
        chunks = load_chunks("data/chunks.json")
        assert len(chunks) == 105

    def test_chunks_have_content(self):
        chunks = load_chunks("data/chunks.json")
        for c in chunks:
            assert c.chunk_id
            assert c.content
            # Most chunks should have page info
        with_pages = [c for c in chunks if c.page_start is not None]
        assert len(with_pages) > 0
