from __future__ import annotations

# Phase boundary constants (from BITC paper)
TAU_FLUID_CRYSTAL = 0.3
TAU_GAS_FLUID = 0.7

BLOCKS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


def sparkline(values: list[float], width: int = 20) -> str:
    """Render values as unicode sparkline. Values should be in [0, 1]."""
    if not values:
        return ""
    recent = values[-width:]
    return "".join(BLOCKS[min(int(v * 7), 7)] for v in recent)


def phase_char(tau: float) -> str:
    """Single character phase indicator from tau (disorder parameter)."""
    if tau < TAU_FLUID_CRYSTAL:
        return "C"  # crystalline
    if tau <= TAU_GAS_FLUID:
        return "F"  # fluid
    return "G"  # gaseous


def status_line(phi_values: list[float], tau: float, step: int, gate_fired: bool = False) -> str:
    """Render the always-on status line for carl observe."""
    spark = sparkline(phi_values)
    phase = phase_char(tau)
    gate = "STAGED" if gate_fired else "GATING"
    return f"\r[{phase}] Phi:{spark} step:{step} {gate}"
