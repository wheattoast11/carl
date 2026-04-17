"""Tests for the research paper retrieval pipeline."""

from __future__ import annotations

import builtins
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from carl_studio.research.arxiv_client import (
    PaperResult,
    format_paper_summary,
    get_paper,
    search_papers,
)


# ---------------------------------------------------------------------------
# PaperResult dataclass
# ---------------------------------------------------------------------------


class TestPaperResult:
    def test_creation_minimal(self) -> None:
        paper = PaperResult(
            title="Test Paper",
            authors=["Alice"],
            abstract="An abstract.",
            arxiv_id="2301.00001",
            published="2023-01-01T00:00:00+00:00",
            updated="2023-01-02T00:00:00+00:00",
            pdf_url="https://arxiv.org/pdf/2301.00001",
        )
        assert paper.title == "Test Paper"
        assert paper.authors == ["Alice"]
        assert paper.categories == []

    def test_creation_with_categories(self) -> None:
        paper = PaperResult(
            title="Categorized",
            authors=["Bob"],
            abstract="Abstract.",
            arxiv_id="2301.00002",
            published="2023-01-01T00:00:00+00:00",
            updated="2023-01-02T00:00:00+00:00",
            pdf_url="https://arxiv.org/pdf/2301.00002",
            categories=["cs.AI", "cs.LG"],
        )
        assert paper.categories == ["cs.AI", "cs.LG"]

    def test_categories_default_not_shared(self) -> None:
        p1 = PaperResult(
            title="A", authors=[], abstract="", arxiv_id="1",
            published="", updated="", pdf_url="",
        )
        p2 = PaperResult(
            title="B", authors=[], abstract="", arxiv_id="2",
            published="", updated="", pdf_url="",
        )
        p1.categories.append("cs.AI")
        assert p2.categories == []


# ---------------------------------------------------------------------------
# format_paper_summary
# ---------------------------------------------------------------------------


class TestFormatPaperSummary:
    def _make_paper(self, *, num_authors: int = 2) -> PaperResult:
        authors = [f"Author{i}" for i in range(num_authors)]
        return PaperResult(
            title="Deep Learning for Coherence",
            authors=authors,
            abstract="We study coherence.",
            arxiv_id="2301.12345",
            published="2023-01-15T00:00:00+00:00",
            updated="2023-02-01T00:00:00+00:00",
            pdf_url="https://arxiv.org/pdf/2301.12345",
            categories=["cs.AI", "cs.LG"],
        )

    def test_contains_title(self) -> None:
        text = format_paper_summary(self._make_paper())
        assert "**Deep Learning for Coherence**" in text

    def test_contains_arxiv_id(self) -> None:
        text = format_paper_summary(self._make_paper())
        assert "2301.12345" in text

    def test_contains_categories(self) -> None:
        text = format_paper_summary(self._make_paper())
        assert "cs.AI" in text
        assert "cs.LG" in text

    def test_truncated_date(self) -> None:
        text = format_paper_summary(self._make_paper())
        assert "2023-01-15" in text

    def test_few_authors_listed(self) -> None:
        text = format_paper_summary(self._make_paper(num_authors=2))
        assert "Author0, Author1" in text
        assert "et al." not in text

    def test_many_authors_truncated(self) -> None:
        text = format_paper_summary(self._make_paper(num_authors=5))
        assert "et al." in text
        assert "5 authors" in text

    def test_contains_abstract(self) -> None:
        text = format_paper_summary(self._make_paper())
        assert "We study coherence." in text


# ---------------------------------------------------------------------------
# search_papers — import error
# ---------------------------------------------------------------------------


class TestSearchPapersImportError:
    def test_raises_helpful_import_error(self) -> None:
        real_import = builtins.__import__

        def _block_arxiv(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "arxiv":
                raise ImportError("No module named 'arxiv'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_arxiv):
            with pytest.raises(ImportError, match="carl-studio\\[research\\]"):
                search_papers("coherence")


# ---------------------------------------------------------------------------
# search_papers — mocked arxiv
# ---------------------------------------------------------------------------


def _fake_arxiv_paper(
    *,
    title: str = "Fake Paper",
    arxiv_id: str = "2301.99999",
) -> SimpleNamespace:
    """Build a fake arxiv paper result matching the arxiv library API."""
    return SimpleNamespace(
        title=title,
        authors=[SimpleNamespace(name="Alice"), SimpleNamespace(name="Bob")],
        summary="Fake abstract.",
        entry_id=f"http://arxiv.org/abs/{arxiv_id}",
        published=datetime(2023, 1, 1, tzinfo=timezone.utc),
        updated=datetime(2023, 2, 1, tzinfo=timezone.utc),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        categories=["cs.AI"],
    )


class TestSearchPapersMocked:
    def test_returns_paper_results(self) -> None:
        fake_paper = _fake_arxiv_paper()
        mock_arxiv = MagicMock()
        mock_arxiv.Client.return_value.results.return_value = [fake_paper]
        mock_arxiv.Search = MagicMock()
        mock_arxiv.SortCriterion.Relevance = "relevance"

        with patch.dict(sys.modules, {"arxiv": mock_arxiv}):
            papers = search_papers("test query", max_results=1)

        assert len(papers) == 1
        assert papers[0].title == "Fake Paper"
        assert papers[0].authors == ["Alice", "Bob"]
        assert papers[0].arxiv_id == "2301.99999"
        assert papers[0].categories == ["cs.AI"]

    def test_returns_empty_list(self) -> None:
        mock_arxiv = MagicMock()
        mock_arxiv.Client.return_value.results.return_value = []
        mock_arxiv.Search = MagicMock()
        mock_arxiv.SortCriterion.Relevance = "relevance"

        with patch.dict(sys.modules, {"arxiv": mock_arxiv}):
            papers = search_papers("nonexistent", max_results=1)

        assert papers == []


# ---------------------------------------------------------------------------
# get_paper — mocked arxiv
# ---------------------------------------------------------------------------


class TestGetPaperMocked:
    def test_returns_paper(self) -> None:
        fake_paper = _fake_arxiv_paper(arxiv_id="2301.11111")
        mock_arxiv = MagicMock()
        mock_arxiv.Client.return_value.results.return_value = [fake_paper]
        mock_arxiv.Search = MagicMock()

        with patch.dict(sys.modules, {"arxiv": mock_arxiv}):
            paper = get_paper("2301.11111")

        assert paper is not None
        assert paper.arxiv_id == "2301.11111"
        assert paper.title == "Fake Paper"

    def test_returns_none_when_not_found(self) -> None:
        mock_arxiv = MagicMock()
        mock_arxiv.Client.return_value.results.return_value = []
        mock_arxiv.Search = MagicMock()

        with patch.dict(sys.modules, {"arxiv": mock_arxiv}):
            paper = get_paper("0000.00000")

        assert paper is None

    def test_raises_helpful_import_error(self) -> None:
        real_import = builtins.__import__

        def _block_arxiv(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "arxiv":
                raise ImportError("No module named 'arxiv'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_block_arxiv):
            with pytest.raises(ImportError, match="carl-studio\\[research\\]"):
                get_paper("2301.00001")
