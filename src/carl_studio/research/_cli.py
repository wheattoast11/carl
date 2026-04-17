"""Research paper CLI."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

research_app = typer.Typer(help="Search and retrieve research papers.")


@research_app.command(name="search")
def search_cmd(
    query: str = typer.Argument(..., help="Search query"),
    max_results: int = typer.Option(5, "--max", "-n", help="Max results"),
) -> None:
    """Search arxiv for papers."""
    from carl_studio.research.arxiv_client import format_paper_summary, search_papers

    console = Console()
    try:
        papers = search_papers(query, max_results=max_results)
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    if not papers:
        console.print("[dim]No papers found.[/dim]")
        return

    for paper in papers:
        console.print(
            Panel(
                format_paper_summary(paper),
                title=f"[bold]{paper.arxiv_id}[/bold]",
                border_style="dim",
            )
        )


@research_app.command(name="get")
def get_cmd(
    arxiv_id: str = typer.Argument(..., help="ArXiv paper ID (e.g., 2301.12345)"),
) -> None:
    """Get a specific paper by arxiv ID."""
    from carl_studio.research.arxiv_client import format_paper_summary, get_paper

    console = Console()
    try:
        paper = get_paper(arxiv_id)
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    if not paper:
        console.print(f"[red]Paper {arxiv_id} not found.[/red]")
        raise typer.Exit(1)

    console.print(Panel(format_paper_summary(paper), border_style="cyan"))
