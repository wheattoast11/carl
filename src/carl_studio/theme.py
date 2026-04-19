"""Camp CARL theme system.

Two personas (CARL/CARLI) with full color palette, iconography, and voice.
Users pick on first run. Customizable via ~/.carl/theme.yaml.

The theme is the single primitive that drives all CLI rendering.
Every output goes through the theme. Change the theme, change everything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Persona(str, Enum):
    CARL = "carl"
    CARLI = "carli"


@dataclass
class Palette:
    """6-color system. Everything renders from these."""

    primary: str = "#2D5F2D"      # Forest green
    secondary: str = "#1A2744"    # Night sky navy
    accent: str = "#E8722A"       # Campfire orange
    success: str = "#D4A847"      # Merit badge gold
    warning: str = "#C44B2F"      # Ember red
    muted: str = "#8B7E6A"        # Trail dust


@dataclass
class Icons:
    """Status symbols. Swap these for a completely different feel."""

    ok: str = "◆"
    warn: str = "▲"
    fail: str = "✕"
    info: str = "●"
    progress: str = "◇"
    badge: str = "★"
    fire: str = "🔥"
    crystal: str = "◈"
    arrow: str = "→"


@dataclass
class Voice:
    """Camp personality. Every status message goes through here."""

    greeting: str = ""
    farewell: str = ""
    training_start: str = ""
    training_done: str = ""
    eval_pass: str = ""
    eval_fail: str = ""
    phase_transition: str = ""
    send_it: str = ""
    error: str = ""
    idle: str = ""


# ---------------------------------------------------------------------------
# Persona defaults
# ---------------------------------------------------------------------------

CARL_PALETTE = Palette(
    primary="#2D5F2D",
    secondary="#1A2744",
    accent="#E8722A",
    success="#D4A847",
    warning="#C44B2F",
    muted="#8B7E6A",
)

CARLI_PALETTE = Palette(
    primary="#3A6B5C",
    secondary="#2E1A44",
    accent="#E85D8A",
    success="#7BD48A",
    warning="#E8A22A",
    muted="#9B8EA6",
)

CARL_VOICE = Voice(
    greeting="Welcome to Camp CARL.",
    farewell="Lights out. See you tomorrow.",
    training_start="Drills started. Your carlito is warming up.",
    training_done="Training complete. Merit badge earned.",
    eval_pass="Skills test: PASS. Ready for graduation.",
    eval_fail="Skills test: FAIL. Back to practice.",
    phase_transition="Phase transition detected. Your carlito just had a breakthrough.",
    send_it="Full send. Carl's got it from here.",
    error="Something went sideways. Check the counselor's notes.",
    idle="Quiet time. Your carlito is resting.",
)

CARLI_VOICE = Voice(
    greeting="Hey! Welcome to camp!",
    farewell="Great session! Can't wait for tomorrow.",
    training_start="Let's gooo! Your carlito is getting started!",
    training_done="YES! Training done! Look at those numbers!",
    eval_pass="Passed! Your carlito absolutely crushed it!",
    eval_fail="Not quite there yet — but we'll get it next time.",
    phase_transition="Whoa!! Phase transition! The crystal just formed!",
    send_it="SEND IT! I'll handle everything. Go grab a snack.",
    error="Oops — something broke. Let me take a look.",
    idle="Taking a breather. Carlito's recharging.",
)

CARL_ICONS = Icons()  # Default geometric

CARLI_ICONS = Icons(
    ok="✓",
    warn="⚡",
    fail="✗",
    info="◉",
    progress="…",
    badge="⭐",
    fire="🔥",
    crystal="💎",
    arrow="→",
)


@dataclass
class CampTheme:
    """The one primitive. Everything renders through this."""

    persona: Persona = Persona.CARL
    palette: Palette = field(default_factory=lambda: Palette())
    icons: Icons = field(default_factory=lambda: Icons())
    voice: Voice = field(default_factory=lambda: Voice())
    density: str = "normal"  # "chill" (spacious) or "normal" or "focused" (compact)
    ascii_art: bool = True   # Show camp banner on startup
    badges: bool = True      # Show merit badges on completion
    sound: bool = False      # Terminal bell on events (future)

    @classmethod
    def carl(cls) -> CampTheme:
        return cls(
            persona=Persona.CARL,
            palette=CARL_PALETTE,
            icons=CARL_ICONS,
            voice=CARL_VOICE,
        )

    @classmethod
    def carli(cls) -> CampTheme:
        return cls(
            persona=Persona.CARLI,
            palette=CARLI_PALETTE,
            icons=CARLI_ICONS,
            voice=CARLI_VOICE,
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

THEME_DIR = Path.home() / ".carl"
THEME_FILE = THEME_DIR / "theme.yaml"


def load_theme() -> CampTheme:
    """Load theme from ~/.carl/theme.yaml, falling back to defaults."""
    if THEME_FILE.exists():
        try:
            import yaml
            with open(THEME_FILE) as f:
                raw = yaml.safe_load(f) or {}
            persona = Persona(raw.get("persona", "carl"))
            base = CampTheme.carl() if persona == Persona.CARL else CampTheme.carli()

            # Override fields from yaml
            if "density" in raw:
                base.density = raw["density"]
            if "ascii_art" in raw:
                base.ascii_art = raw["ascii_art"]
            if "badges" in raw:
                base.badges = raw["badges"]
            if "palette" in raw and isinstance(raw["palette"], dict):
                for k, v in raw["palette"].items():
                    if hasattr(base.palette, k):
                        setattr(base.palette, k, v)
            if "icons" in raw and isinstance(raw["icons"], dict):
                for k, v in raw["icons"].items():
                    if hasattr(base.icons, k):
                        setattr(base.icons, k, v)
            return base
        except Exception:
            pass

    # Check env var
    persona_env = os.environ.get("CARL_PERSONA", "").lower()
    if persona_env == "carli":
        return CampTheme.carli()
    return CampTheme.carl()


def save_theme(theme: CampTheme) -> None:
    """Save theme to ~/.carl/theme.yaml."""
    import yaml
    THEME_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "persona": theme.persona.value,
        "density": theme.density,
        "ascii_art": theme.ascii_art,
        "badges": theme.badges,
    }
    with open(THEME_FILE, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


# ---------------------------------------------------------------------------
# Rendering helpers (Rich-based)
# ---------------------------------------------------------------------------

def get_console():
    """Get a CampConsole configured with the current theme.

    Returns (CampConsole, CampTheme) for backward compatibility.
    Prefer ``from carl_studio.console import get_console`` directly.
    """
    from carl_studio.console import CampConsole
    theme = load_theme()
    cc = CampConsole(theme)
    return cc, theme


def banner(version: str = "") -> str:
    """Camp CARL ASCII banner."""
    theme = load_theme()
    if not theme.ascii_art:
        return f"CARL Studio {version}" if version else "CARL Studio"

    if theme.persona == Persona.CARL:
        return f"""
   ╔═══════════════════════════════╗
   ║     ◈  CAMP CARL  ◈          ║
   ║   Coherence-Aware RL {version:>8s} ║
   ╚═══════════════════════════════╝
   {theme.voice.greeting}
"""
    else:
        return f"""
   ┌───────────────────────────────┐
   │     💎  CAMP CARL  💎          │
   │   Coherence-Aware RL {version:>8s} │
   └───────────────────────────────┘
   {theme.voice.greeting}
"""


def badge(name: str, detail: str = "") -> str:
    """Format a merit badge award."""
    theme = load_theme()
    if not theme.badges:
        return f"  {name}: {detail}" if detail else f"  {name}"
    icon = theme.icons.badge
    return f"  {icon} {name}" + (f" — {detail}" if detail else "")


def status_line(icon_key: str, message: str) -> str:
    """Format a status line with themed icon."""
    theme = load_theme()
    icon = getattr(theme.icons, icon_key, "●")
    return f"  [{icon}] {message}"
