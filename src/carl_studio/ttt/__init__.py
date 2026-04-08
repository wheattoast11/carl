"""
TTT (Test-Time Training) — Adaptive inference mechanisms.

Provides:
  - SLOTOptimizer: Per-sample hidden delta optimization (non-parametric, ephemeral)
  - SLOTResult: Dataclass holding SLOT optimization output
  - LoRAMicroUpdate: Rank-1 per-interaction weight adaptation (parametric, persistent)

These require torch and (for LoRA) peft at runtime.
"""
from carl_studio.ttt.slot import SLOTOptimizer, SLOTResult, LoRAMicroUpdate

__all__ = ["SLOTOptimizer", "SLOTResult", "LoRAMicroUpdate"]
