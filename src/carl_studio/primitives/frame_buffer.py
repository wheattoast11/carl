"""Compat shim: `carl_studio.primitives.frame_buffer` -> `carl_core.frame_buffer`."""
from carl_core.frame_buffer import *  # noqa: F401, F403
from carl_core.frame_buffer import FrameBuffer, FrameRecord

__all__ = ["FrameBuffer", "FrameRecord"]
