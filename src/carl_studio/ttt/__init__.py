"""
TTT (Test-Time Training) — Adaptive inference mechanisms.

Provides:
  - SLOTOptimizer: Per-sample hidden delta optimization (non-parametric, ephemeral)
  - SLOTResult: Dataclass holding SLOT optimization output
  - LoRAMicroUpdate: Rank-1 per-interaction weight adaptation (parametric, persistent)
  - EMLHead: Opaque handle to an EML tree head (fitter lives in terminals-runtime)

Most members here require torch and (for LoRA) peft at runtime. We lazy-load
them via ``__getattr__`` so that ``from carl_studio.ttt.eml_head import EMLHead``
does not drag transformers / terminals-runtime into sys.modules.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from carl_studio.ttt.eml_head import EMLHead
    from carl_studio.ttt.grpo_integration import (
        GRPOReflectionCallback,
        TTTStepResult,
    )
    from carl_studio.ttt.slot import LoRAMicroUpdate, SLOTOptimizer, SLOTResult


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "SLOTOptimizer": ("carl_studio.ttt.slot", "SLOTOptimizer"),
    "SLOTResult": ("carl_studio.ttt.slot", "SLOTResult"),
    "LoRAMicroUpdate": ("carl_studio.ttt.slot", "LoRAMicroUpdate"),
    "GRPOReflectionCallback": (
        "carl_studio.ttt.grpo_integration",
        "GRPOReflectionCallback",
    ),
    "TTTStepResult": ("carl_studio.ttt.grpo_integration", "TTTStepResult"),
    "EMLHead": ("carl_studio.ttt.eml_head", "EMLHead"),
}


def __getattr__(name: str) -> Any:
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module 'carl_studio.ttt' has no attribute {name!r}")
    mod_name, attr = spec
    import importlib

    mod = importlib.import_module(mod_name)
    value = getattr(mod, attr)
    globals()[name] = value
    return value


__all__ = [
    "SLOTOptimizer", "SLOTResult", "LoRAMicroUpdate",
    "GRPOReflectionCallback", "TTTStepResult",
    "EMLHead",
]
