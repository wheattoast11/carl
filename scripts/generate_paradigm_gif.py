#!/usr/bin/env python3
"""
Generate an animated GIF visualizing the CARL phase transition paradigm.

Produces:
  assets/carl-paradigm.gif  — animated 5-frame loop (~12s)
  assets/carl-paradigm.svg  — static fallback (crystal frame)

Dependencies: matplotlib, numpy, pillow (all standard scientific Python).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

GIF_PATH = ASSETS_DIR / "carl-paradigm.gif"
SVG_PATH = ASSETS_DIR / "carl-paradigm.svg"

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
BG_COLOR = "#0d1117"
TEXT_COLOR = "#e6edf3"
GRID_ROWS = 12
GRID_COLS = 24
DOT_RADIUS = 0.32
FIG_W, FIG_H = 8.0, 4.0  # inches (800x400 at 100 dpi)
DPI = 100

# Blue -> purple -> gold palette (8 stops)
PALETTE_HEX = [
    "#1f6feb",  # blue
    "#388bfd",  # light blue
    "#8957e5",  # purple
    "#a371f7",  # light purple
    "#bc8cff",  # lavender
    "#d2a8ff",  # pale purple
    "#e3b341",  # gold
    "#f0c75e",  # light gold
]
PALETTE_RGB = np.array([mcolors.to_rgb(c) for c in PALETTE_HEX])
N_COLORS = len(PALETTE_RGB)

# Crystal color (dominant) and accent
CRYSTAL_COLOR = np.array(mcolors.to_rgb("#a371f7"))  # light purple
CRYSTAL_ACCENT = np.array(mcolors.to_rgb("#e3b341"))  # gold
GRPO_ARROW_COLOR = "#f0c75e"

# Frame durations in milliseconds (total ~12s at 10 fps interpolation)
# We render sub-frames for smooth transitions.
FRAME_SPEC = [
    # (label, subtitle, duration_seconds)
    (
        "Before Training",
        "Phi ~ 0, Entropy = max",
        2.0,
    ),
    (
        "SFT Step 15",
        "Entropy Spike (1.0 -> 9.3)",
        2.0,
    ),
    (
        "Step 25 -- Crystallization",
        "Acc: 3% -> 65% in 5 steps",
        3.0,
    ),
    (
        "Step 46 -- Converged",
        "Acc: 99.3%, Phi ~ 1.0",
        2.0,
    ),
    (
        "GRPO",
        "Vision activates spatial grounding",
        3.0,
    ),
]

# RNG for reproducibility
RNG = np.random.RandomState(42)

# ---------------------------------------------------------------------------
# Grid state generators
# ---------------------------------------------------------------------------

def _grid_positions() -> np.ndarray:
    """Return (GRID_ROWS*GRID_COLS, 2) array of grid center positions."""
    xs = np.linspace(0.5, GRID_COLS - 0.5, GRID_COLS)
    ys = np.linspace(0.5, GRID_ROWS - 0.5, GRID_ROWS)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _random_colors(n: int) -> np.ndarray:
    """Fully random palette indices."""
    return RNG.randint(0, N_COLORS, size=n)


def _melting_colors(n: int) -> np.ndarray:
    """More chaotic than random -- bimodal clusters with noise."""
    colors = np.empty(n, dtype=int)
    # Create unstable clusters that partially form then break
    for i in range(n):
        r = RNG.random()
        if r < 0.3:
            colors[i] = 0  # blue cluster
        elif r < 0.5:
            colors[i] = 2  # purple cluster
        else:
            colors[i] = RNG.randint(0, N_COLORS)  # noise
    # Add entropy spikes -- random swaps
    swap_count = n // 3
    for _ in range(swap_count):
        i, j = RNG.randint(0, n, size=2)
        colors[i], colors[j] = colors[j], colors[i]
    return colors


def _crystal_colors(n: int, rows: int, cols: int) -> np.ndarray:
    """Nearly uniform crystal with structured variation in specific columns."""
    # Dominant color index = 3 (light purple)
    colors = np.full(n, 3, dtype=int)
    # Columns 5, 11, 17 have structured variation (the "digits that vary")
    vary_cols = {5, 11, 17}
    for idx in range(n):
        col = idx % cols
        row = idx // cols
        if col in vary_cols:
            # Structured: cycle through gold/blue based on row
            colors[idx] = 6 if (row % 3 == 0) else (0 if row % 3 == 1 else 7)
    return colors


def _transition_colors(
    src: np.ndarray,
    dst: np.ndarray,
    t: float,
) -> np.ndarray:
    """Interpolate between two color-index arrays.

    t=0 -> src, t=1 -> dst. Uses a snap function (sigmoid) so the
    transition is dramatic around t=0.5.
    """
    # Sharpen with sigmoid
    k = 12.0  # steepness
    s = 1.0 / (1.0 + np.exp(-k * (t - 0.5)))
    mask = RNG.random(len(src)) < s
    out = src.copy()
    out[mask] = dst[mask]
    return out


def _grpo_colors(n: int, rows: int, cols: int) -> tuple[np.ndarray, list]:
    """Crystal lattice with spatial grounding arrows overlaid."""
    colors = _crystal_colors(n, rows, cols)
    # Generate arrow data: (start_col, start_row, dx, dy)
    arrows = []
    arrow_starts = [(3, 2), (10, 5), (18, 8), (7, 9), (15, 3), (21, 6)]
    for sx, sy in arrow_starts:
        dx = RNG.uniform(-1.5, 1.5)
        dy = RNG.uniform(-1.0, 1.0)
        arrows.append((sx + 0.5, sy + 0.5, dx, dy))
    return colors, arrows


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _draw_dots(
    ax: plt.Axes,
    positions: np.ndarray,
    color_indices: np.ndarray,
    jitter: float = 0.0,
) -> None:
    """Draw filled circles at grid positions with palette colors."""
    n = len(positions)
    rgbs = PALETTE_RGB[color_indices % N_COLORS]

    if jitter > 0:
        offsets = RNG.uniform(-jitter, jitter, size=(n, 2))
        pos = positions + offsets
    else:
        pos = positions

    # Use scatter for performance
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=rgbs,
        s=(DOT_RADIUS * 72 / DPI * 30) ** 2,  # approximate marker area
        edgecolors="none",
        zorder=2,
    )


def _draw_arrows(ax: plt.Axes, arrows: list) -> None:
    """Draw spatial grounding arrows."""
    for sx, sy, dx, dy in arrows:
        ax.annotate(
            "",
            xy=(sx + dx, sy + dy),
            xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="->",
                color=GRPO_ARROW_COLOR,
                lw=2.0,
                connectionstyle="arc3,rad=0.15",
            ),
            zorder=3,
        )


def _setup_axes(ax: plt.Axes) -> None:
    """Configure axes for the dot grid."""
    ax.set_xlim(-0.5, GRID_COLS + 0.5)
    ax.set_ylim(-0.5, GRID_ROWS + 0.5)
    ax.set_aspect("equal")
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_label(
    ax: plt.Axes,
    title: str,
    subtitle: str,
) -> None:
    """Draw frame label text."""
    ax.text(
        GRID_COLS / 2,
        GRID_ROWS + 0.3,
        title,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        fontfamily="monospace",
        color=TEXT_COLOR,
        zorder=5,
    )
    ax.text(
        GRID_COLS / 2,
        -0.6,
        subtitle,
        ha="center",
        va="top",
        fontsize=9,
        fontfamily="monospace",
        color="#8b949e",
        zorder=5,
    )


def _draw_entropy_arrow(ax: plt.Axes, level: float, rising: bool) -> None:
    """Draw an entropy indicator bar on the right edge.

    level: 0..1 normalized entropy.
    rising: if True, draw an upward arrow tip.
    """
    bar_x = GRID_COLS + 0.1
    bar_w = 0.4
    bar_h = level * GRID_ROWS
    bar_bottom = 0.0

    # Gradient bar
    gradient = np.linspace(0, 1, 50).reshape(-1, 1)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "entropy", ["#1f6feb", "#e3b341"]
    )
    ax.imshow(
        gradient,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        extent=[bar_x, bar_x + bar_w, bar_bottom, bar_bottom + bar_h],
        zorder=4,
    )

    # Arrow tip
    if rising:
        ax.annotate(
            "",
            xy=(bar_x + bar_w / 2, bar_bottom + bar_h + 0.4),
            xytext=(bar_x + bar_w / 2, bar_bottom + bar_h),
            arrowprops=dict(arrowstyle="->", color="#e3b341", lw=1.5),
            zorder=5,
        )

    # Label
    ax.text(
        bar_x + bar_w / 2,
        -0.2,
        "H",
        ha="center",
        va="top",
        fontsize=7,
        fontfamily="monospace",
        color="#8b949e",
        zorder=5,
    )


# ---------------------------------------------------------------------------
# Pre-compute frame states
# ---------------------------------------------------------------------------

POSITIONS = _grid_positions()
N_DOTS = len(POSITIONS)

STATE_RANDOM = _random_colors(N_DOTS)
STATE_MELTING = _melting_colors(N_DOTS)
STATE_CRYSTAL = _crystal_colors(N_DOTS, GRID_ROWS, GRID_COLS)
STATE_GRPO_COLORS, STATE_GRPO_ARROWS = _grpo_colors(N_DOTS, GRID_ROWS, GRID_COLS)

# Sub-frame breakdown (10 fps):
# Frame 0 (Random):       20 sub-frames = 2s
# Frame 1 (Melting):      20 sub-frames = 2s
# Frame 2 (Transition):   30 sub-frames = 3s
# Frame 3 (Crystal):      20 sub-frames = 2s
# Frame 4 (GRPO):         30 sub-frames = 3s
# Total: 120 sub-frames = 12s

FPS = 10
SUB_FRAMES = []
for i, (title, subtitle, duration) in enumerate(FRAME_SPEC):
    n_sub = int(duration * FPS)
    for sf in range(n_sub):
        SUB_FRAMES.append((i, sf, n_sub, title, subtitle))

TOTAL_FRAMES = len(SUB_FRAMES)


def _render_frame(ax: plt.Axes, frame_idx: int) -> None:
    """Render a single sub-frame onto the given axes."""
    ax.clear()
    _setup_axes(ax)

    stage, sf, n_sub, title, subtitle = SUB_FRAMES[frame_idx]
    t = sf / max(n_sub - 1, 1)  # 0..1 within this stage

    if stage == 0:
        # Random state, stable
        _draw_dots(ax, POSITIONS, STATE_RANDOM, jitter=0.05)
        _draw_entropy_arrow(ax, level=0.85, rising=False)
        _draw_label(ax, title, subtitle)

    elif stage == 1:
        # Melting: transition from random toward melting state
        colors = _transition_colors(STATE_RANDOM, STATE_MELTING, t)
        jitter = 0.05 + 0.15 * t  # increasing chaos
        _draw_dots(ax, POSITIONS, colors, jitter=jitter)
        _draw_entropy_arrow(ax, level=0.85 + 0.15 * t, rising=True)
        _draw_label(ax, title, subtitle)

    elif stage == 2:
        # Phase transition: dramatic snap from melting to crystal
        # Use sharper sigmoid for more dramatic snap
        colors = _transition_colors(STATE_MELTING, STATE_CRYSTAL, t)
        jitter = 0.20 * (1.0 - t ** 0.5)  # chaos -> order
        _draw_dots(ax, POSITIONS, colors, jitter=jitter)
        entropy_level = 1.0 - 0.85 * (1.0 / (1.0 + np.exp(-15 * (t - 0.4))))
        _draw_entropy_arrow(ax, level=max(0.05, entropy_level), rising=False)
        _draw_label(ax, title, subtitle)

    elif stage == 3:
        # Crystal: stable, clean
        _draw_dots(ax, POSITIONS, STATE_CRYSTAL, jitter=0.0)
        _draw_entropy_arrow(ax, level=0.08, rising=False)
        _draw_label(ax, title, subtitle)

    elif stage == 4:
        # GRPO: crystal + arrows fade in
        _draw_dots(ax, POSITIONS, STATE_GRPO_COLORS, jitter=0.0)
        _draw_entropy_arrow(ax, level=0.08, rising=False)
        if t > 0.2:
            # Fade arrows in
            alpha = min(1.0, (t - 0.2) / 0.3)
            # Draw arrows with alpha
            for sx, sy, dx, dy in STATE_GRPO_ARROWS:
                ax.annotate(
                    "",
                    xy=(sx + dx, sy + dy),
                    xytext=(sx, sy),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=GRPO_ARROW_COLOR,
                        lw=2.5,
                        alpha=alpha,
                        connectionstyle="arc3,rad=0.15",
                    ),
                    zorder=3,
                )
            # Spatial grounding labels near arrow tips
            if t > 0.5:
                label_alpha = min(1.0, (t - 0.5) / 0.3)
                for j, (sx, sy, dx, dy) in enumerate(STATE_GRPO_ARROWS):
                    ax.text(
                        sx + dx,
                        sy + dy + 0.4,
                        f"({int(sx*40+dx*20)},{int(sy*40+dy*20)})",
                        ha="center",
                        va="bottom",
                        fontsize=5.5,
                        fontfamily="monospace",
                        color=GRPO_ARROW_COLOR,
                        alpha=label_alpha,
                        zorder=5,
                    )
        _draw_label(ax, title, subtitle)


# ---------------------------------------------------------------------------
# Generate GIF
# ---------------------------------------------------------------------------

def generate_gif() -> None:
    """Render and save the animated GIF."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    fig.subplots_adjust(left=0.02, right=0.95, top=0.88, bottom=0.10)

    def update(frame_idx: int) -> None:
        _render_frame(ax, frame_idx)

    anim = FuncAnimation(
        fig,
        update,
        frames=TOTAL_FRAMES,
        interval=1000 // FPS,  # ms per sub-frame
        repeat=True,
    )

    print(f"Rendering {TOTAL_FRAMES} frames at {FPS} fps...")
    anim.save(
        str(GIF_PATH),
        writer=PillowWriter(fps=FPS),
        dpi=DPI,
    )
    plt.close(fig)

    size_kb = GIF_PATH.stat().st_size / 1024
    print(f"Saved: {GIF_PATH} ({size_kb:.0f} KB, {TOTAL_FRAMES} frames, {FPS} fps)")


# ---------------------------------------------------------------------------
# Generate static SVG fallback (crystal frame)
# ---------------------------------------------------------------------------

def generate_svg() -> None:
    """Render the crystal frame as a static SVG."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG_COLOR)
    fig.subplots_adjust(left=0.02, right=0.95, top=0.88, bottom=0.10)

    _setup_axes(ax)
    _draw_dots(ax, POSITIONS, STATE_CRYSTAL, jitter=0.0)
    _draw_entropy_arrow(ax, level=0.08, rising=False)
    _draw_label(
        ax,
        "Step 46 -- Converged",
        "Acc: 99.3%, Phi ~ 1.0",
    )

    # Add CARL watermark
    ax.text(
        GRID_COLS - 0.2,
        0.2,
        "CARL",
        ha="right",
        va="bottom",
        fontsize=7,
        fontfamily="monospace",
        color="#484f58",
        zorder=5,
    )

    fig.savefig(str(SVG_PATH), format="svg", facecolor=BG_COLOR, dpi=DPI)
    plt.close(fig)

    size_kb = SVG_PATH.stat().st_size / 1024
    print(f"Saved: {SVG_PATH} ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"Output directory: {ASSETS_DIR}")
    generate_gif()
    generate_svg()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
