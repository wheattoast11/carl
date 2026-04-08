"""carl observe — live training dashboard.

Two modes:
  - One-shot: ``carl observe`` → Claude-powered CoherenceObserver assessment
  - Live TUI: ``carl observe --live`` → Textual dashboard with real-time metrics

The TUI requires ``carl-studio[tui]`` (textual + textual-plotext).
"""

from carl_studio.observe.data_source import FileSource, TrackioSource

__all__ = ["FileSource", "TrackioSource"]
