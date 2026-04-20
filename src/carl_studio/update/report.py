"""UpdateReport — the structured output of `carl update`.

All fields typed (Pydantic) so the report serializes cleanly to JSON
for machine-readable consumption and renders with CampConsole for
humans.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GitCommitDelta(BaseModel):
    """One commit in the delta since last update."""

    sha: str
    date: str
    subject: str
    author: str


class DependencyDelta(BaseModel):
    """Current-vs-latest version delta for one dependency."""

    package: str
    current: str
    latest: str
    severity: str = Field(
        default="info",
        description="'info' | 'minor' | 'patch' | 'major' | 'security'",
    )
    summary: str | None = None


class BlastRadiusEntry(BaseModel):
    """Positive-framed impact summary for one change item."""

    category: str = Field(
        description="'commit' | 'dependency' | 'security' | 'convention'"
    )
    target: str = Field(description="What the change affects (file, pkg, etc.)")
    direction: str = Field(
        default="positive",
        description="'positive' | 'breaking' | 'neutral' — framing for the user",
    )
    impact: str = Field(description="One-line explanation of the unlock/improvement")


class UpdateReport(BaseModel):
    """Full report surfaced by ``carl update`` (and checked for staleness)."""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    carl_version: str = Field(description="Current carl-studio __version__")
    git_head: str | None = Field(default=None, description="Short SHA of local HEAD")
    last_check_at: datetime | None = Field(
        default=None, description="Previous `carl update` completion timestamp"
    )
    git_commits: list[GitCommitDelta] = Field(default_factory=list)
    dep_deltas: list[DependencyDelta] = Field(default_factory=list)
    security_findings: list[dict[str, Any]] = Field(default_factory=list)
    blast_radius: list[BlastRadiusEntry] = Field(default_factory=list)
    errors: list[str] = Field(
        default_factory=list,
        description="Non-fatal error strings collected during scanning.",
    )

    @property
    def any_deltas(self) -> bool:
        """True when there's anything worth surfacing in the report."""
        return bool(
            self.git_commits
            or self.dep_deltas
            or self.security_findings
            or self.blast_radius
        )


def _build_blast_radius(
    commits: list[GitCommitDelta],
    deps: list[DependencyDelta],
) -> list[BlastRadiusEntry]:
    """Derive positive-framed blast-radius entries from the scanned deltas.

    Heuristic but deterministic — e.g., a commit mentioning "fix" maps to
    direction=positive + "resolved" phrasing; a major dep bump maps to
    "new capability available". No LLM required; zero network.
    """

    entries: list[BlastRadiusEntry] = []
    for c in commits[:5]:  # cap: surface ~top-5 recent, full list in git_commits
        subject_lower = c.subject.lower()
        if any(k in subject_lower for k in ("fix", "resolve")):
            impact = f"Resolves: {c.subject}"
        elif any(k in subject_lower for k in ("feat", "add", "enable")):
            impact = f"Unlocks: {c.subject}"
        else:
            impact = f"Updates: {c.subject}"
        entries.append(
            BlastRadiusEntry(
                category="commit", target=c.sha[:7], direction="positive", impact=impact
            )
        )

    for d in deps[:5]:
        if d.severity == "major":
            direction = "breaking"
            impact = f"{d.package} {d.current} → {d.latest}: major — review changelog"
        else:
            direction = "positive"
            impact = (
                f"{d.package} {d.current} → {d.latest}: {d.summary or d.severity}"
            )
        entries.append(
            BlastRadiusEntry(
                category="dependency",
                target=d.package,
                direction=direction,
                impact=impact,
            )
        )

    return entries


def build_update_report(
    *,
    carl_version: str,
    git_head: str | None = None,
    last_check_at: datetime | None = None,
    git_commits: list[GitCommitDelta] | None = None,
    dep_deltas: list[DependencyDelta] | None = None,
    security_findings: list[dict[str, Any]] | None = None,
    errors: list[str] | None = None,
) -> UpdateReport:
    """Assemble the full report from scanned deltas + derive blast radius."""

    commits = git_commits or []
    deps = dep_deltas or []
    report = UpdateReport(
        carl_version=carl_version,
        git_head=git_head,
        last_check_at=last_check_at,
        git_commits=commits,
        dep_deltas=deps,
        security_findings=security_findings or [],
        blast_radius=_build_blast_radius(commits, deps),
        errors=errors or [],
    )
    return report
