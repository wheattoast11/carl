"""Arxiv paper search and retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PaperResult:
    """A single paper from arxiv search."""

    title: str
    authors: list[str]
    abstract: str
    arxiv_id: str
    published: str
    updated: str
    pdf_url: str
    categories: list[str] = field(default_factory=list)


def _require_arxiv():  # type: ignore[no-untyped-def]
    """Import and return the arxiv module, raising a helpful error if missing."""
    try:
        import arxiv
        return arxiv
    except ImportError:
        raise ImportError(
            "This feature requires the arxiv package. "
            "Install: pip install 'carl-studio[research]'"
        ) from None


def search_papers(query: str, max_results: int = 5) -> list[PaperResult]:
    """Search arxiv for papers matching query.

    Requires: pip install arxiv
    """
    arxiv = _require_arxiv()

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results = []
    for paper in client.results(search):
        results.append(
            PaperResult(
                title=paper.title,
                authors=[a.name for a in paper.authors],
                abstract=paper.summary,
                arxiv_id=paper.entry_id.split("/")[-1],
                published=paper.published.isoformat() if paper.published else "",
                updated=paper.updated.isoformat() if paper.updated else "",
                pdf_url=paper.pdf_url or "",
                categories=list(paper.categories) if paper.categories else [],
            )
        )

    return results


def get_paper(arxiv_id: str) -> PaperResult | None:
    """Get a specific paper by arxiv ID (e.g., '2301.12345')."""
    arxiv = _require_arxiv()

    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])

    for paper in client.results(search):
        return PaperResult(
            title=paper.title,
            authors=[a.name for a in paper.authors],
            abstract=paper.summary,
            arxiv_id=arxiv_id,
            published=paper.published.isoformat() if paper.published else "",
            updated=paper.updated.isoformat() if paper.updated else "",
            pdf_url=paper.pdf_url or "",
            categories=list(paper.categories) if paper.categories else [],
        )
    return None


def format_paper_summary(paper: PaperResult) -> str:
    """Format a paper for display or LLM context."""
    authors_str = ", ".join(paper.authors[:3])
    if len(paper.authors) > 3:
        authors_str += f" et al. ({len(paper.authors)} authors)"
    return (
        f"**{paper.title}**\n"
        f"Authors: {authors_str}\n"
        f"Published: {paper.published[:10]}\n"
        f"ArXiv: {paper.arxiv_id}\n"
        f"Categories: {', '.join(paper.categories)}\n\n"
        f"{paper.abstract}"
    )
