"""Typed requests and caller-bound permissions for local agent delegation."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from carl_core.errors import CARLError
from carl_core.safepath import safe_resolve
from pydantic import BaseModel, ConfigDict, Field


class Host(StrEnum):
    """Native agent protocols supported by CARL."""

    CODEX = "codex"
    CLAUDE = "claude"
    OPENCODE = "opencode"


class DelegationRequest(BaseModel):
    """A bounded task request; caller context supplies its authority."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    host: Host
    instruction: str = Field(min_length=1, max_length=262144)
    workdir: Path = Field(default_factory=Path.cwd)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=128)
    timeout_s: float = Field(default=600, gt=0, le=3600)
    write: bool = False
    model: str | None = None
    native_session: str | None = None


@dataclass(frozen=True)
class DelegationContext:
    """Trusted entrypoint bounds, never taken from model-supplied task fields."""

    owner: str
    root: Path
    allow_write: bool = False
    depth: int = 0

    @classmethod
    def local(cls, *, owner: str, root: Path, allow_write: bool = False) -> DelegationContext:
        """Bind an operator-selected root to the current CARL session."""
        depth = int(os.environ.get("CARL_DELEGATION_DEPTH", "0"))
        return cls(owner=owner, root=root.resolve(), allow_write=allow_write, depth=depth)

    def admit(self, request: DelegationRequest) -> Path:
        """Validate the task before creating a process or consuming a slot."""
        if self.depth >= 1:
            raise CARLError("Child delegation is disabled", code="carl.agent.depth")
        if request.write and not self.allow_write:
            raise CARLError("Write scope was not granted", code="carl.agent.permission")
        if request.native_session:
            raise CARLError(
                "Use the original live task for continuation; foreign session resume is refused",
                code="carl.agent.session",
            )
        directory = safe_resolve(request.workdir, self.root, must_exist=True)
        if not directory.is_dir():
            raise CARLError("Workdir must be a directory", code="carl.agent.workdir")
        return directory


DELEGATION_TOOLS = frozenset(
    {
        "delegate_agent",
        "tasks_reply",
        "tasks_get",
        "tasks_cancel",
        "read_agent_result",
        "read_agent_input",
    }
)


def trace_projection(name: str, arguments: dict[str, Any], result: Any) -> tuple[Any, Any]:
    """Keep delegation payloads out of traces, including denied calls."""
    if name in {"encode_data", "interpret", "interpretation_feedback"}:
        from carl_core.hashing import content_hash
        references = {key: arguments[key] for key in
                      ("interpretation_id", "feedback_ref", "artifact_ref", "confirmed") if key in arguments}
        references["input_sha256"] = content_hash(arguments)
        return references, {"content_retained": False, "output_sha256": content_hash(result)}
    if name not in DELEGATION_TOOLS:
        return arguments, result
    safe = {key: arguments[key] for key in ("host", "task_id", "request_id") if key in arguments}
    return safe, {"content_retained": False}
