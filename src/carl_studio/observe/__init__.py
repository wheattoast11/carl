"""carl observe — live training dashboard.

Two modes:
  - One-shot: ``carl observe`` → Claude-powered CoherenceObserver assessment
  - Live TUI: ``carl observe --live`` → Textual dashboard with real-time metrics

The TUI requires ``carl-studio[tui]`` (textual + textual-plotext).
"""

from . import data_source
from .data_source import FileSource, ObserveFrame, TraceFileSource, TrackioSource
from .sparkline import sparkline, phase_char, status_line
from .crystal_panel import CrystalPanel
from .crystal_viz import crystal_structure

__all__ = [
    "FileSource", "ObserveFrame", "TraceFileSource", "TrackioSource",
    "sparkline", "phase_char", "status_line",
    "CrystalPanel", "crystal_structure",
]
