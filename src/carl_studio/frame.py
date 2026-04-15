"""WorkFrame — the lens through which CARL structures any analytical problem.

Axioms:

1. Structure precedes meaning.  Imposing {domain, function, role, objective}
   on an unstructured space is the first act of intelligence.

2. Attention is dimensionality reduction.  The frame determines what to attend
   to — {pharma × drug_rollout × manager} attends to different entities than
   {SaaS × quota_setting × analyst}, even on identical underlying data.

3. Frameworks are composable lenses.  MECE decomposition ensures completeness
   without overlap.  Any business problem decomposes into independent lanes
   along the frame's dimensions.

4. Resolution is observer-dependent.  An analyst sees account-level signals;
   a VP sees portfolio-level trends.  The frame encodes this.

5. Isomorphism across domains.  Territory design for SaaS is structurally
   equivalent to vaccine distribution logistics.  The decomposition —
   partition, assign, gate, iterate — is invariant.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_CARL_HOME = Path.home() / ".carl"
_DEFAULT_FRAME_PATH = _CARL_HOME / "frame.yaml"


class WorkFrame(BaseModel):
    """Declarative specification of how to see a problem.

    Four orthogonal MECE dimensions:
      domain    — subject matter (vocabulary, entities, relationships)
      function  — analytical lens (metrics, KPIs, success criteria)
      role      — decision scope (actions, authority, resolution)
      objectives — optimization targets (what "good" means)
    """

    domain: str = ""
    function: str = ""
    role: str = ""
    objectives: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    context: str = ""

    @property
    def active(self) -> bool:
        """True if any dimension is set."""
        return bool(self.domain or self.function or self.role or self.objectives)

    def attention_query(self) -> str:
        """Structured prompt prefix that shapes extraction toward this frame.

        Used as a QKV-style query: determines what to attend to in source
        data and what values to extract.
        """
        parts: list[str] = []
        if self.domain:
            parts.append(f"Domain: {self.domain}")
        if self.function:
            parts.append(f"Function: {self.function}")
        if self.role:
            parts.append(f"Role perspective: {self.role}")
        if self.objectives:
            parts.append(f"Objectives: {'; '.join(self.objectives)}")
        if self.entities:
            parts.append(f"Key entities: {', '.join(self.entities)}")
        if self.metrics:
            parts.append(f"Metrics: {', '.join(self.metrics)}")
        if self.constraints:
            parts.append(f"Constraints: {'; '.join(self.constraints)}")
        if self.context:
            parts.append(f"Context: {self.context}")
        return "\n".join(parts)

    def decompose(self) -> list[str]:
        """MECE decomposition: independent analytical lanes implied by this frame.

        Each lane is a concern that can be investigated in parallel without
        overlap.  The union of lanes covers the full problem space.
        """
        lanes: list[str] = []
        if self.domain:
            lanes.append(f"domain:{self.domain}")
        if self.function:
            lanes.append(f"function:{self.function}")
        for obj in self.objectives:
            lanes.append(f"objective:{obj}")
        for metric in self.metrics:
            lanes.append(f"metric:{metric}")
        return lanes or ["general"]

    def entity_patterns(self) -> list[str]:
        """Return entity names as case-insensitive patterns for extraction."""
        return [e.lower() for e in self.entities if e.strip()]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """Persist to YAML."""
        dest = Path(path) if path else _DEFAULT_FRAME_PATH
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in self.model_dump().items() if v}
        dest.write_text(yaml.dump({"frame": data}, default_flow_style=False, sort_keys=False))
        return dest

    @classmethod
    def load(cls, path: str | Path | None = None) -> WorkFrame:
        """Load from YAML.  Returns empty frame if file doesn't exist."""
        src = Path(path) if path else _DEFAULT_FRAME_PATH
        if not src.is_file():
            return cls()
        try:
            raw = yaml.safe_load(src.read_text())
            data: dict[str, Any] = raw.get("frame", raw) if isinstance(raw, dict) else {}
            return cls.model_validate(data)
        except Exception:
            logger.warning("Failed to parse frame from %s, returning empty frame", src)
            return cls()

    @classmethod
    def from_project(cls, config_path: str = "carl.yaml") -> WorkFrame:
        """Read frame from carl.yaml project config if present."""
        p = Path(config_path)
        if not p.is_file():
            return cls.load()  # fall back to global
        try:
            raw = yaml.safe_load(p.read_text())
            if isinstance(raw, dict) and "frame" in raw:
                return cls.model_validate(raw["frame"])
        except Exception:
            pass
        return cls.load()

    def clear(self) -> None:
        """Remove persisted frame."""
        if _DEFAULT_FRAME_PATH.is_file():
            _DEFAULT_FRAME_PATH.unlink()
