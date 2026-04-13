from __future__ import annotations
from carl_studio.observe.ansi import color_rgb, bold


def _fill_char(value: float) -> str:
    """Map reward value [0,1] to fill density character.

    Args:
        value: Reward value in [0, 1].
    """
    if value < 0.25:
        return "\u2591"  # light shade
    if value < 0.50:
        return "\u2592"  # medium shade
    if value < 0.75:
        return "\u2593"  # dark shade
    return "\u2588"  # full block


def crystal_structure(rewards: dict[str, float]) -> str:
    """Render 6-face isometric crystal. Each face maps to a reward function.

    Face mapping:
      top    = task_completion (the peak -- what we're building toward)
      front  = tool_engagement (the visible face -- what users see)
      right  = gated_carl (quality -- activated after gate)
      left   = tool_format (structure -- parseability)
      back   = gr3_length (constraint -- anti-hack)
      bottom = diversity (foundation -- anti-collapse)

    Args:
        rewards: Dict mapping reward names to values in [0, 1].
    """
    faces = {
        "top": rewards.get("task_completion", 0),
        "front": rewards.get("tool_engagement", 0),
        "right": rewards.get("gated_carl", 0),
        "left": rewards.get("tool_format", 0),
        "back": rewards.get("gr3_length", 0),
        "bottom": rewards.get("diversity", 0),
    }

    top = _fill_char(faces["top"])
    front = _fill_char(faces["front"])
    right = _fill_char(faces["right"])
    left = _fill_char(faces["left"])
    bottom = _fill_char(faces["bottom"])

    _cyan = lambda s: color_rgb(s, 0, 255, 200)       # task completion
    _blue = lambda s: color_rgb(s, 100, 200, 255)      # tool engagement
    _purple = lambda s: color_rgb(s, 200, 100, 255)    # gated carl
    _amber = lambda s: color_rgb(s, 255, 200, 100)     # tool format
    _grey = lambda s: color_rgb(s, 100, 100, 100)      # bottom/diversity

    crystal = [
        f"      {_cyan(top * 6)}",
        f"     {_cyan('/' + top * 6 + chr(92))}",
        f"    {_cyan('/' + top * 8 + chr(92))}",
        f"   {_amber(left * 4)}{_blue(front * 6)}{_purple(right * 4)}",
        f"   {_amber(left * 4)}{_blue(front * 6)}{_purple(right * 4)}",
        f"   {_amber(left * 4)}{_blue(front * 6)}{_purple(right * 4)}",
        f"   {_amber(left * 4)}{_blue(front * 6)}{_purple(right * 4)}",
        f"    {_grey(chr(92) + bottom * 8 + '/')}",
        f"     {_grey(chr(92) + bottom * 6 + '/')}",
        f"      {_grey(bottom * 6)}",
    ]

    # Labels
    labels = [
        "",
        bold("  CARL Crystal Structure"),
        "",
        f"  top:    task_completion  {faces['top']:.2f}",
        f"  front:  tool_engagement  {faces['front']:.2f}",
        f"  right:  gated_carl       {faces['right']:.2f}",
        f"  left:   tool_format      {faces['left']:.2f}",
        f"  back:   gr3_length       {faces['back']:.2f}",
        f"  bottom: diversity        {faces['bottom']:.2f}",
        "",
    ]

    # Side by side
    result: list[str] = []
    max_lines = max(len(crystal), len(labels))
    for i in range(max_lines):
        left = crystal[i] if i < len(crystal) else ""
        right = labels[i] if i < len(labels) else ""
        result.append(f"{left}  {right}")

    return "\n".join(result)
