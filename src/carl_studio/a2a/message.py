"""A2AMessage — communication between agents during task execution."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class A2AMessage(BaseModel):
    """Progress update or notification between agents."""

    id: str
    task_id: str
    type: Literal["progress", "artifact", "error", "log", "heartbeat"]
    payload: dict[str, Any] = Field(default_factory=dict)
    sender: str = "carl-studio"
    timestamp: str = Field(default_factory=_utcnow)

    @classmethod
    def progress(
        cls,
        task_id: str,
        step: int,
        total: int,
        detail: str = "",
        sender: str = "carl-studio",
    ) -> "A2AMessage":
        """Emit a progress update message."""
        return cls(
            id=str(uuid.uuid4()),
            task_id=task_id,
            type="progress",
            sender=sender,
            payload={"step": step, "total": total, "detail": detail},
        )

    @classmethod
    def artifact(
        cls,
        task_id: str,
        name: str,
        data: dict[str, Any],
        sender: str = "carl-studio",
    ) -> "A2AMessage":
        """Emit an artifact (output file, model checkpoint reference, etc.)."""
        return cls(
            id=str(uuid.uuid4()),
            task_id=task_id,
            type="artifact",
            sender=sender,
            payload={"name": name, "data": data},
        )

    @classmethod
    def log(
        cls,
        task_id: str,
        message: str,
        level: str = "info",
        sender: str = "carl-studio",
    ) -> "A2AMessage":
        """Emit a log line."""
        return cls(
            id=str(uuid.uuid4()),
            task_id=task_id,
            type="log",
            sender=sender,
            payload={"message": message, "level": level},
        )

    @classmethod
    def error(
        cls,
        task_id: str,
        message: str,
        detail: str = "",
        sender: str = "carl-studio",
    ) -> "A2AMessage":
        """Emit an error message (not terminal — task may still recover)."""
        return cls(
            id=str(uuid.uuid4()),
            task_id=task_id,
            type="error",
            sender=sender,
            payload={"message": message, "detail": detail},
        )

    @classmethod
    def heartbeat(
        cls,
        task_id: str,
        sender: str = "carl-studio",
    ) -> "A2AMessage":
        """Emit a heartbeat to signal the worker is still alive."""
        return cls(
            id=str(uuid.uuid4()),
            task_id=task_id,
            type="heartbeat",
            sender=sender,
            payload={},
        )
