"""Back-compat shim — FrameBuffer lives in ``carl_core.frame_buffer``."""
from __future__ import annotations

from carl_core.frame_buffer import FrameBuffer, FrameRecord

__all__ = ["FrameBuffer", "FrameRecord"]
