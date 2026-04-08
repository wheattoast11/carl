"""VLM reward functions for GUI grounding tasks."""
from __future__ import annotations
import math
import re
from carl_studio.training.rewards.base import extract_text


def _parse_coordinates(text: str) -> tuple[int, int] | None:
    """Extract (x, y) coordinates from text."""
    m = re.search(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def coordinate_format_reward(completions: list, **kwargs) -> list[float]:
    """Does output match (x, y) coordinate format? Weight: 1.5"""
    rewards = []
    for c in completions:
        text = extract_text(c)
        coords = _parse_coordinates(text)
        if coords is not None:
            stripped = re.sub(r"\(?\s*\d+\s*,\s*\d+\s*\)?", "", text).strip()
            rewards.append(1.0 if len(stripped) < 10 else 0.6)
        else:
            rewards.append(0.0)
    return rewards


def click_accuracy_reward(
    completions: list, bbox: list | None = None, max_dist: float = 1000.0, **kwargs,
) -> list[float]:
    """Is predicted (x,y) inside the ground-truth bbox? Weight: 3.0

    Screen-relative partial credit: provides gradient signal even at 500px+ distance.
    bbox: list of [x1, y1, x2, y2] per sample (xyxy format, pixel coords).
    max_dist: screen-scale distance for partial credit falloff (default 1000px).
    """
    if bbox is None:
        return [0.0] * len(completions)
    rewards = []
    for c, box in zip(completions, bbox):
        text = extract_text(c)
        coords = _parse_coordinates(text)
        if coords is None:
            rewards.append(0.0)
            continue
        x, y = coords
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            rewards.append(0.0)
            continue
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        if x1 <= x <= x2 and y1 <= y <= y2:
            rewards.append(1.0)
        else:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            score = max(0.0, 0.5 * (1.0 - dist / max_dist))
            rewards.append(round(score, 4))
    return rewards


def precision_reward(
    completions: list, bbox: list | None = None, max_dist: float = 1000.0, **kwargs,
) -> list[float]:
    """Distance from bbox center -- closer = higher reward. Weight: 2.0

    Uses screen-scale distance (default 1000px) to provide gradient signal
    at large distances. At 504px: score = 0.496 (vs 0.0 with old 200px scale).
    """
    if bbox is None:
        return [0.0] * len(completions)
    rewards = []
    for c, box in zip(completions, bbox):
        text = extract_text(c)
        coords = _parse_coordinates(text)
        if coords is None:
            rewards.append(0.0)
            continue
        x, y = coords
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            rewards.append(0.0)
            continue
        cx = (float(box[0]) + float(box[2])) / 2
        cy = (float(box[1]) + float(box[3])) / 2
        dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        score = max(0.0, 1.0 - dist / max_dist)
        rewards.append(round(score, 4))
    return rewards
