"""carl observe — live training dashboard.

Two modes:
  - One-shot: ``carl observe`` → Claude-powered CoherenceObserver assessment
  - Live TUI: ``carl observe --live`` → Textual dashboard with real-time metrics

The TUI requires ``carl-studio[tui]`` (textual + textual-plotext).
"""

from . import data_source
from .data_source import FileSource, ObserveFrame, TraceFileSource, TrackioSource

__all__ = ["FileSource", "ObserveFrame", "TraceFileSource", "TrackioSource"]
