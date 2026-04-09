"""Adapter for CARL tool-calling datasets (wheattoast11/zero-rl-tool-calling-data)."""

from __future__ import annotations

from typing import Any, Iterator

from carl_studio.data.adapters.base import DataAdapter
from carl_studio.data.types import Domain, Modality, UnifiedSample


class ToolCallingAdapter(DataAdapter):
    """Adapt CARL tool-calling format: {messages: [{role, content}]}."""

    def adapt(self, raw: list[dict[str, Any]]) -> Iterator[UnifiedSample]:
        for i, row in enumerate(raw):
            messages = row.get("messages", [])
            user_msg = ""
            for m in messages:
                if m.get("role") == "user":
                    user_msg = m.get("content", "")
                    break
            if not user_msg:
                continue

            prompt = [
                {"role": "system", "content": "You are a coding agent. Solve tasks by writing and executing code."},
                {"role": "user", "content": user_msg},
            ]

            yield UnifiedSample(
                id=f"tool_calling_{i}",
                prompt=prompt,
                problem_statement=user_msg,
                domain=Domain.TOOL_CALLING,
                modality=Modality.CODE,
                source=self.source.name,
                metadata={"original_messages": messages},
            )
