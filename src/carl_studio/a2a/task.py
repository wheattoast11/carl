"""A2ATask — a unit of work dispatched between agents."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class A2ATaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class A2ATask(BaseModel):
    """A task dispatched to a CARL agent."""

    id: str
    skill: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    status: A2ATaskStatus = A2ATaskStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    sender: str = ""
    receiver: str = "carl-studio"
    created_at: str = Field(default_factory=_utcnow)
    updated_at: str = Field(default_factory=_utcnow)
    completed_at: str | None = None
    priority: int = 0

    def mark_running(self) -> "A2ATask":
        """Transition to RUNNING. No-op if already terminal."""
        if self.is_terminal():
            return self
        return self.model_copy(update={
            "status": A2ATaskStatus.RUNNING,
            "updated_at": _utcnow(),
        })

    def mark_done(self, result: dict[str, Any]) -> "A2ATask":
        """Transition to DONE with result payload."""
        if self.is_terminal():
            return self
        now = _utcnow()
        return self.model_copy(update={
            "status": A2ATaskStatus.DONE,
            "result": result,
            "completed_at": now,
            "updated_at": now,
        })

    def mark_failed(self, error: str) -> "A2ATask":
        """Transition to FAILED with error message."""
        if self.is_terminal():
            return self
        now = _utcnow()
        return self.model_copy(update={
            "status": A2ATaskStatus.FAILED,
            "error": error,
            "completed_at": now,
            "updated_at": now,
        })

    def mark_cancelled(self) -> "A2ATask":
        """Transition to CANCELLED."""
        if self.is_terminal():
            return self
        now = _utcnow()
        return self.model_copy(update={
            "status": A2ATaskStatus.CANCELLED,
            "completed_at": now,
            "updated_at": now,
        })

    def is_terminal(self) -> bool:
        """True when no further transitions are valid."""
        return self.status in (
            A2ATaskStatus.DONE,
            A2ATaskStatus.FAILED,
            A2ATaskStatus.CANCELLED,
        )
