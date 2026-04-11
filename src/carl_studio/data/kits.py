"""Kit registry — named reward + template configurations.

A Kit is a named bundle of ingredients (rewards) with weights,
molds (templates), and gates (validation thresholds).

Built-in kits ship with carl-studio. Users can add custom kits
in ~/.carl/kits/ as YAML files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Kit", "KitRegistry", "IngredientRef", "GateRef", "BUILTIN_KITS"]


@dataclass
class IngredientRef:
    """Reference to a reward function with weight."""

    id: str
    weight: float
    stage: str = "all"  # "A", "B", "all"


@dataclass
class GateRef:
    """Gate configuration for stage transitions."""

    metric: str
    threshold: float
    window: int = 5


@dataclass
class Kit:
    """Named reward + template configuration."""

    id: str
    name: str
    description: str
    ingredients: list[IngredientRef]
    molds: list[str] = field(default_factory=list)  # template IDs, empty = all
    gates: list[GateRef] = field(default_factory=list)

    def reward_weights(self) -> dict[str, float]:
        """Return {ingredient_id: weight} mapping."""
        return {i.id: i.weight for i in self.ingredients}


# ---------------------------------------------------------------------------
# Built-in kits
# ---------------------------------------------------------------------------

BUILTIN_KITS: dict[str, Kit] = {
    "coding-agent": Kit(
        id="coding-agent",
        name="Coding Agent",
        description=(
            "General-purpose coding agent. Balanced tool use + task completion + quality."
        ),
        ingredients=[
            IngredientRef("task-completion", 3.0),
            IngredientRef("tool-engagement", 1.0),
            IngredientRef("gated-carl", 1.0, stage="B"),
            IngredientRef("tool-format", 1.5),
            IngredientRef("gr3-length", 0.5),
        ],
    ),
    "tool-specialist": Kit(
        id="tool-specialist",
        name="Tool Specialist",
        description="Heavy tool use focus. For models that need to learn tool calling.",
        ingredients=[
            IngredientRef("task-completion", 3.0),
            IngredientRef("tool-engagement", 2.0),
            IngredientRef("tool-selection", 1.5),
            IngredientRef("tool-format", 2.0),
        ],
        molds=[
            "diag-wrong-file",
            "diag-import-chain",
            "recover-edge-case-trap",
        ],
    ),
    "reasoning": Kit(
        id="reasoning",
        name="Reasoning",
        description=(
            "Thinking quality over tool volume. Coherence + crystallization focus."
        ),
        ingredients=[
            IngredientRef("task-completion", 3.0),
            IngredientRef("coherence", 1.0),
            IngredientRef("crystallization", 1.0),
            IngredientRef("conciseness", 0.5),
        ],
        molds=[
            "reason-profile-optimize",
            "reason-analyze-decide-act",
            "plan-dependency-order",
        ],
    ),
    "mcp-agent": Kit(
        id="mcp-agent",
        name="MCP Agent",
        description=(
            "Multi-tool chain specialist. For models working with MCP servers."
        ),
        ingredients=[
            IngredientRef("task-completion", 3.0),
            IngredientRef("tool-engagement", 1.0),
            IngredientRef("chain-completion", 2.0),
        ],
        molds=[
            "plan-dependency-order",
            "cohere-standardize",
            "plan-monkey-patch-trap",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class KitRegistry:
    """Load built-in + user kits.

    User kits live in ~/.carl/kits/*.yaml. Each file must have at minimum an
    ``id`` key; all other fields follow the same schema as BUILTIN_KITS.
    Malformed YAML files are skipped silently (missing keys raise KeyError at
    construction time, which is also swallowed so one bad file can't block the
    rest).
    """

    def __init__(self) -> None:
        self._kits: dict[str, Kit] = dict(BUILTIN_KITS)
        self._load_user_kits()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_user_kits(self) -> None:
        user_dir = Path.home() / ".carl" / "kits"
        if not user_dir.is_dir():
            return
        for yml in sorted(user_dir.glob("*.yaml")):
            try:
                with open(yml, encoding="utf-8") as f:
                    data: Any = yaml.safe_load(f)
                if not isinstance(data, dict) or "id" not in data:
                    continue
                kit = Kit(
                    id=data["id"],
                    name=data.get("name", data["id"]),
                    description=data.get("description", ""),
                    ingredients=[
                        IngredientRef(**i) for i in data.get("ingredients", [])
                    ],
                    molds=data.get("molds", []),
                    gates=[GateRef(**g) for g in data.get("gates", [])],
                )
                self._kits[kit.id] = kit
            except Exception:
                pass  # skip malformed kit files

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, kit_id: str) -> Kit:
        """Return a Kit by ID.

        Raises:
            KeyError: If kit_id is not registered.
        """
        if kit_id not in self._kits:
            available = ", ".join(sorted(self._kits))
            raise KeyError(f"Kit '{kit_id}' not found. Available: {available}")
        return self._kits[kit_id]

    def list_kits(self) -> list[Kit]:
        """Return all registered kits (built-in + user)."""
        return list(self._kits.values())

    def register(self, kit: Kit) -> None:
        """Register a kit at runtime. Overwrites if id already exists."""
        if not isinstance(kit, Kit):
            raise TypeError(f"Expected Kit, got {type(kit).__name__}")
        self._kits[kit.id] = kit
