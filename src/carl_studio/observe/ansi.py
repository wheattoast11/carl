from __future__ import annotations


def move_cursor(row: int, col: int) -> str:
    """Move terminal cursor to row, col (1-indexed)."""
    return f"\033[{row};{col}H"


def clear_screen() -> str:
    """Clear terminal screen and move cursor to top-left."""
    return "\033[2J\033[H"


def color_rgb(text: str, r: int, g: int, b: int) -> str:
    """Apply 24-bit foreground color to text."""
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def bg_rgb(text: str, r: int, g: int, b: int) -> str:
    """Apply 24-bit background color to text."""
    return f"\033[48;2;{r};{g};{b}m{text}\033[0m"


def bold(text: str) -> str:
    """Apply bold formatting to text."""
    return f"\033[1m{text}\033[0m"


def dim(text: str) -> str:
    """Apply dim formatting to text."""
    return f"\033[2m{text}\033[0m"


def bar(value: float, width: int = 20, fill: str = "\u2588", empty: str = "\u2591") -> str:
    """Render a horizontal bar chart segment. Value clamped to [0, 1]."""
    value = max(0.0, min(1.0, value))
    filled = int(value * width)
    return fill * filled + empty * (width - filled)
