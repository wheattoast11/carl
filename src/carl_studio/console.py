"""Camp CARL Rich console.

Single rendering surface for all CLI output. Every command uses this.
Change the theme, change everything.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme as RichTheme

from carl_studio.theme import CampTheme, Persona, load_theme


def _build_rich_theme(theme: CampTheme) -> RichTheme:
    """Map camp palette to Rich style names."""
    p = theme.palette
    return RichTheme({
        "camp.primary": f"bold {p.primary}",
        "camp.secondary": f"{p.secondary}",
        "camp.accent": f"bold {p.accent}",
        "camp.success": f"bold {p.success}",
        "camp.warning": f"bold {p.warning}",
        "camp.muted": p.muted,
        "camp.pass": f"bold {p.success}",
        "camp.fail": f"bold {p.warning}",
        "camp.header": f"bold {p.primary}",
        "camp.key": p.muted,
        "camp.value": "bold white",
        "camp.dim": "dim",
    })


class CampConsole:
    """Themed Rich Console for all CLI output.

    Usage::

        from carl_studio.console import console
        console.header("CARL Train", "v0.2.0")
        console.kv("Model", "Tesslate/OmniCoder-9B")
        console.gate("PASS", detail="94.6% click accuracy")
    """

    def __init__(self, theme: CampTheme | None = None) -> None:
        self.theme = theme or load_theme()
        self._console = Console(theme=_build_rich_theme(self.theme))

    @property
    def raw(self) -> Console:
        """Access underlying Rich Console for advanced usage."""
        return self._console

    # -- Output primitives --------------------------------------------------

    def print(self, *args: Any, **kwargs: Any) -> None:
        self._console.print(*args, **kwargs)

    def blank(self) -> None:
        self._console.print()

    def rule(self, title: str = "") -> None:
        self._console.rule(title, style="camp.muted")

    # -- Structured output --------------------------------------------------

    def header(self, title: str, subtitle: str = "") -> None:
        """Command header panel."""
        t = self.theme
        icon = t.icons.crystal
        body = f"{icon}  {title}"
        if subtitle:
            body += f"  {subtitle}"
        border = "green" if t.persona == Persona.CARL else "magenta"
        self._console.print(Panel(
            Text(body, justify="center"),
            border_style=border,
            padding=(0, 2),
        ))

    def voice(self, key: str) -> None:
        """Print a themed voice line (greeting, training_start, etc.)."""
        msg = getattr(self.theme.voice, key, None)
        if msg:
            self._console.print(f"  {msg}", style="camp.muted")

    def kv(self, key: str, value: Any, key_width: int = 12) -> None:
        """Key-value line."""
        self._console.print(f"  {key + ':':<{key_width + 1}} [camp.value]{value}[/]", style="camp.key")

    def metric(self, key: str, value: float, fmt: str = ".4f") -> None:
        """Numeric metric line."""
        self._console.print(f"  {key + ':':<31} [camp.value]{value:{fmt}}[/]", style="camp.key")

    def info(self, message: str) -> None:
        """Info line with icon."""
        icon = self.theme.icons.info
        self._console.print(f"  [{icon}] {message}", style="camp.muted")

    def ok(self, message: str) -> None:
        """Success line with icon."""
        icon = self.theme.icons.ok
        self._console.print(f"  [{icon}] {message}", style="camp.success")

    def warn(self, message: str) -> None:
        """Warning line with icon."""
        icon = self.theme.icons.warn
        self._console.print(f"  [{icon}] {message}", style="camp.warning")

    def error(self, message: str) -> None:
        """Error line with icon."""
        icon = self.theme.icons.fail
        self._console.print(f"  [{icon}] {message}", style="camp.warning")

    def gate(self, passed: bool, detail: str = "") -> None:
        """PASS/FAIL gate result."""
        if passed:
            label = Text("PASS", style="camp.pass")
            icon = self.theme.icons.ok
        else:
            label = Text("FAIL", style="camp.fail")
            icon = self.theme.icons.fail
        line = Text(f"  {icon} Gate: ")
        line.append(label)
        if detail:
            line.append(f"  {detail}", style="camp.muted")
        self._console.print(line)

    def badge_award(self, name: str, detail: str = "") -> None:
        """Merit badge earned."""
        icon = self.theme.icons.badge
        line = f"  {icon} {name}"
        if detail:
            line += f" -- {detail}"
        self._console.print(line, style="camp.success")

    # -- Tables -------------------------------------------------------------

    def make_table(
        self,
        *columns: str,
        title: str = "",
        show_header: bool = True,
        pad_edge: bool = True,
    ) -> Table:
        """Create a themed Table. Caller adds rows, then passes to print()."""
        t = Table(
            title=title,
            show_header=show_header,
            header_style="camp.header",
            border_style="camp.muted",
            pad_edge=pad_edge,
            padding=(0, 1),
        )
        for col in columns:
            t.add_column(col)
        return t

    # -- Config summary (reused by train, project show, etc.) ---------------

    def config_block(self, pairs: list[tuple[str, Any]], title: str = "") -> None:
        """Render a key-value config block as a panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("key", style="camp.key", min_width=10)
        table.add_column("value", style="camp.value")
        for k, v in pairs:
            table.add_row(k, str(v))
        if title:
            self._console.print(Panel(table, title=title, border_style="camp.muted", padding=(0, 1)))
        else:
            self._console.print(table)

    # -- Banner -------------------------------------------------------------

    def banner(self, version: str = "") -> None:
        """Print the Camp CARL startup banner."""
        t = self.theme
        if not t.ascii_art:
            self._console.print(f"CARL Studio {version}" if version else "CARL Studio")
            return
        self.header(f"CAMP CARL  --  Coherence-Aware RL {version}", "carl.camp")
        self.voice("greeting")

    # -- Constants line (kappa, sigma) --------------------------------------

    def constants(self) -> None:
        """Print the CARL conservation constants."""
        self._console.print(
            f"  [camp.muted]\u03ba={64/3:.4f}  \u03c3={3/16:.4f}  \u03ba\u03c3=4.0[/]"
        )


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_console: CampConsole | None = None


def get_console() -> CampConsole:
    """Get or create the module-level CampConsole singleton."""
    global _console
    if _console is None:
        _console = CampConsole()
    return _console


