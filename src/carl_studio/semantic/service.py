"""Session-owned encoding, recall, explicit feedback and consolidation."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

from carl_core.data_handles import DataRef
from carl_core.errors import CARLError
from carl_core.hashing import content_hash
from carl_core.interaction import ActionType

from .types import Carrier, EncoderBinding, ExecutionBinding, Interpretation, SemanticInput

if TYPE_CHECKING:
    from carl_studio.session import Session


class SemanticService:
    """One implementation shared by every surface of a Session."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self._lock = RLock()
        self.binding: EncoderBinding | None = None
        self.execution: ExecutionBinding | None = None
        self.encoder: Callable[[SemanticInput], Carrier] | None = None
        self.head_weights: tuple[float, ...] | dict[int, tuple[float, ...]] | None = None
        self.validate_binding: Callable[[], None] | None = None
        self.generation: str | None = None
        self._cache: dict[str, tuple[Carrier, dict[str, Any]]] = {}
        self.interpretations: dict[str, Interpretation] = {}
        self.refinements = 0
        self.event_refinements: dict[str, int] = {}

    def bind(
        self,
        binding: EncoderBinding,
        execution: ExecutionBinding,
        encoder: Callable[[SemanticInput], Carrier],
        *,
        validator: Callable[[], None] | None = None,
        generation: str | None = None,
    ) -> None:
        """Atomically bind one encoder, source validator and retrieval generation."""
        if execution.processor_sha256 != binding.processor_sha256:
            raise ValueError("Processor identity mismatch")
        with self._lock:
            self.head_weights = None
            self.validate_binding = validator
            self.binding, self.execution, self.encoder = binding, execution, encoder
            self.generation = generation or binding.space_id
            self._cache.clear()

    @property
    def configured(self) -> bool:
        """Recognize a workspace activation before its first lazy model call."""
        if self.encoder is not None or (
            os.environ.get("CARL_ENCODER_MODEL") and os.environ.get("CARL_ENCODER_PYTHON")
        ):
            return True
        if self.session.workspace:
            from carl_studio.db import LocalDB

            key = "encoder-workspace:" + content_hash(str(Path(self.session.workspace).resolve()))
            return LocalDB().get_config(key) is not None
        return False

    def begin_turn(self) -> None:
        self.refinements = 0

    def _trace(self, operation: str, references: dict[str, Any], *, success: bool) -> None:
        step = self.session.chain.record(
            ActionType.DATA_TRANSFORM,
            "semantic." + operation,
            input=references,
            output={"success": success},
            success=success,
        )
        step.semantic_refs = references

    def _artifact(self, raw: bytes, key: str) -> dict[str, Any]:
        from carl_studio.settings import carl_home

        root = carl_home() / "artifacts" / "semantic"
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination = root / (key + ".json")
        if destination.exists() and destination.read_bytes() != raw:
            raise ValueError("Derived artifact collision")
        if not destination.exists():
            fd, name = tempfile.mkstemp(dir=root)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(name, destination)
            finally:
                Path(name).unlink(missing_ok=True)
        ref = self.session.data_vault.open_file(destination, content_type="application/json")
        ref = ref.model_copy(update={"sha256": hashlib.sha256(raw).hexdigest()})
        self.session.data_vault.restore_file(ref)
        return ref.describe()

    def restore_artifact(self, descriptor: dict[str, Any]) -> DataRef:
        """Validate a durable descriptor before restoring its original vault ID."""
        ref = DataRef.model_validate({k: v for k, v in descriptor.items() if k != "expires_at"})
        if ref.created_at.tzinfo is None:
            raise CARLError(
                "Semantic source time requires a timezone", code="carl.semantic.invalid_descriptor"
            )
        if ref.is_expired():
            raise CARLError("Semantic source expired", code="carl.semantic.expired")
        parsed = urlparse(ref.uri)
        if ref.kind != "file" or parsed.scheme != "file" or parsed.netloc:
            raise CARLError("Semantic source unavailable", code="carl.semantic.unavailable")
        path = Path(unquote(parsed.path))
        if not path.is_file() or not ref.sha256:
            raise CARLError("Semantic artifact unavailable", code="carl.semantic.unavailable")
        if hashlib.sha256(path.read_bytes()).hexdigest() != ref.sha256:
            raise CARLError("Semantic source changed", code="carl.semantic.stale")
        return self.session.data_vault.restore_file(ref)

    def encode_data(self, data: dict[str, Any], *, expose_sources: bool = True) -> dict[str, Any]:
        """Serialize full-carrier computation against concurrent retries."""
        with self._lock:
            return self._encode_data(data, expose_sources=expose_sources)

    def _encode_data(self, data: dict[str, Any], *, expose_sources: bool = True) -> dict[str, Any]:
        """Compute one full carrier; all resolution views reuse it."""
        try:
            request = SemanticInput.model_validate(data)
        except ValueError:
            self._trace("encode", {"input_sha256": content_hash(data)}, success=False)
            raise
        refs = {"event_id": request.event_id, "content_id": request.content_id}
        try:
            if self.encoder is None and self.session.workspace:
                from carl_studio.db import LocalDB

                key = "encoder-workspace:" + content_hash(
                    str(Path(self.session.workspace).resolve())
                )
                active = LocalDB().get_config(key)
                if active:
                    import json

                    from .local import bind_local, source_identity, source_identity_checkpoint

                    generation = json.loads(active)["current"]
                    checkpoint = Path(generation["checkpoint"])
                    model_identity, processor_identity = source_identity(Path(generation["model"]))
                    if (
                        model_identity != generation["model_artifact_sha256"]
                        or processor_identity != generation["processor"]
                    ):
                        raise ValueError("Active model or processor source bytes changed")
                    if source_identity_checkpoint(checkpoint) != generation["checkpoint_sha256"]:
                        raise ValueError("Active checkpoint source bytes changed")
                    bind_local(
                        self.session,
                        Path(generation["model"]),
                        Path(generation["interpreter"]),
                        checkpoint=checkpoint,
                        expected_execution=ExecutionBinding.model_validate(generation["execution"]),
                    )
            if (
                self.encoder is None
                and os.environ.get("CARL_ENCODER_MODEL")
                and os.environ.get("CARL_ENCODER_PYTHON")
            ):
                from .local import bind_local

                bind_local(
                    self.session,
                    Path(os.environ["CARL_ENCODER_MODEL"]),
                    Path(os.environ["CARL_ENCODER_PYTHON"]),
                )
            if self.binding is None or self.execution is None or self.encoder is None:
                raise CARLError("Encoder is unavailable", code="carl.semantic.unavailable")
            if request.recipe not in self.binding.recipes:
                raise ValueError("Unsupported task recipe")
            for part in request.parts:
                if part.modality not in self.binding.modalities:
                    raise ValueError("Unsupported modality")
                if part.source_ref:
                    ref = next(
                        (
                            r
                            for r in self.session.data_vault.list_refs()
                            if str(r.ref_id) == part.source_ref
                        ),
                        None,
                    )
                    if ref is None or ref.is_expired():
                        raise CARLError(
                            "Source unavailable or expired", code="carl.semantic.unavailable"
                        )
                    raw = self.session.data_vault.read(ref)
                    if hashlib.sha256(raw).hexdigest() != part.source_sha256:
                        raise CARLError("Source bytes changed", code="carl.semantic.stale")
                    if ref.kind == "file" and ref.sha256 is None:
                        self.session.data_vault.restore_file(
                            ref.model_copy(update={"sha256": part.source_sha256})
                        )
            if self.validate_binding is not None:
                self.validate_binding()
            key = content_hash(
                {
                    "content": request.content_id,
                    "space": self.binding.space_id,
                    "execution": self.execution.execution_id,
                }
            )
            if key not in self._cache:
                carrier = Carrier.model_validate(self.encoder(request))
                if len(carrier.values) != self.binding.native_dimension or not set(
                    self.binding.dimensions
                ) <= set(carrier.dimensions):
                    raise ValueError("Encoder carrier disagrees with its dimension capabilities")
                if self.validate_binding is not None:
                    self.validate_binding()
                for part in request.parts:
                    if (
                        part.source_ref
                        and hashlib.sha256(
                            self.session.data_vault.read(self._reference(part.source_ref))
                        ).hexdigest()
                        != part.source_sha256
                    ):
                        raise CARLError(
                            "Source bytes changed during encoding", code="carl.semantic.stale"
                        )
                artifact = self._artifact(carrier.model_dump_json().encode(), key)
                self._cache[key] = (carrier, artifact)
            _, artifact = self._cache[key]
            result: dict[str, Any] = {
                **refs,
                "space_id": self.binding.space_id,
                "execution_id": self.execution.execution_id,
                "recipe": request.recipe,
                "generation": self.generation or self.binding.space_id,
                "dimensions": self.binding.native_dimension,
                "artifact": artifact,
            }
            if expose_sources:
                source_refs: list[dict[str, Any]] = []
                for index, part in enumerate(request.parts):
                    if part.source_ref:
                        source_refs.append(self._reference(part.source_ref).describe())
                    elif part.text is not None:
                        ref = self.session.data_vault.put_bytes(
                            part.text.encode("utf-8"),
                            uri=f"carl-data://semantic/{request.event_id}/part-{index}",
                            content_type="text/plain",
                        )
                        source_refs.append(ref.describe())
                result["source_refs"] = source_refs
            self._trace("encode", result, success=True)
            return result
        except (ValueError, OSError, CARLError, RuntimeError):
            self._trace("encode", refs, success=False)
            raise

    def carrier(self, data: dict[str, Any]) -> Carrier:
        result = self.encode_data(data, expose_sources=False)
        descriptor = result["artifact"]
        ref = self.restore_artifact(descriptor)
        return Carrier.model_validate_json(self.session.data_vault.read(ref))

    def ranking_score(self, query: Carrier, candidate: Carrier, dimension: int) -> float:
        """Keep learned candidate ranking separate from cosine similarity."""
        if self.head_weights is None:
            return query.score(candidate, dimension)
        weights = (
            self.head_weights[dimension]
            if isinstance(self.head_weights, dict)
            else self.head_weights[:dimension]
        )
        return sum(
            a * b * weight
            for a, b, weight in zip(
                query.scoring(dimension),
                candidate.scoring(dimension),
                weights,
            )
        )

    def recall(
        self, query: str, sources: list[tuple[str, str, float]], *, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Progressive scoring retains candidates with zero lexical overlap."""
        if not 1 <= limit <= 8:
            raise ValueError("Recall limit must be between one and eight")
        if not sources:
            return []
        try:
            q = self.carrier(
                {
                    "event_id": "query",
                    "recipe": "SearchQuery",
                    "parts": [{"modality": "text", "text": query}],
                }
            )
            candidates = [
                (
                    ref,
                    lexical,
                    self.carrier(
                        {
                            "event_id": ref,
                            "recipe": "Document",
                            "parts": [{"modality": "text", "text": text}],
                        }
                    ),
                )
                for ref, text, lexical in sources
            ]
            dimensions = self.binding.dimensions if self.binding is not None else q.dimensions
            initial = next((d for d in dimensions if d >= 128), dimensions[-1])
            middle = next((d for d in dimensions if d >= 512), dimensions[-1])
            full = dimensions[-1]
            candidates.sort(key=lambda row: self.ranking_score(q, row[2], initial), reverse=True)
            resolution = initial
            if self.refinements < 2:
                candidates = sorted(
                    candidates[:32],
                    key=lambda row: self.ranking_score(q, row[2], middle),
                    reverse=True,
                )
                self.refinements += 1
                resolution = middle
            if self.refinements < 2:
                candidates = sorted(
                    candidates[:8],
                    key=lambda row: self.ranking_score(q, row[2], full),
                    reverse=True,
                )
                self.refinements += 1
                resolution = full
            return [
                {
                    "source_ref": ref,
                    "lexical_score": lexical,
                    "semantic_score": q.score(vector, resolution),
                    "ranking_score": self.ranking_score(q, vector, resolution),
                    "resolution": resolution,
                    "mode": "semantic",
                }
                for ref, lexical, vector in candidates[:limit]
            ]
        except CARLError as exc:
            if exc.code != "carl.semantic.unavailable":
                raise
            return [
                {
                    "source_ref": ref,
                    "lexical_score": lexical,
                    "semantic_score": None,
                    "resolution": None,
                    "mode": "lexical",
                }
                for ref, _, lexical in sorted(sources, key=lambda row: row[2], reverse=True)
                if lexical > 0
            ][:limit]

    def _reference(self, ref_id: str) -> DataRef:
        ref = next(
            (r for r in self.session.data_vault.list_refs() if str(r.ref_id) == ref_id), None
        )
        if ref is None or ref.is_expired():
            raise CARLError("Semantic reference unavailable", code="carl.semantic.unavailable")
        raw = self.session.data_vault.read(ref)
        if ref.sha256 is not None and hashlib.sha256(raw).hexdigest() != ref.sha256:
            raise CARLError("Semantic source bytes changed", code="carl.semantic.stale")
        if ref.size_bytes is not None and len(raw) != ref.size_bytes:
            raise CARLError("Semantic source size changed", code="carl.semantic.stale")
        return ref

    def interpret(self, record: dict[str, Any]) -> dict[str, Any]:
        """Record one bounded, unconfirmed interpretation atomically."""
        with self._lock:
            return self._interpret(record)

    def _interpret(self, record: dict[str, Any]) -> dict[str, Any]:
        """Record a candidate separately from feedback; silence confirms nothing."""
        refs = {"input_sha256": content_hash(record)}
        try:
            interpretation = Interpretation.model_validate(record)
            if interpretation.feedback != "unconfirmed":
                raise ValueError("Use explicit feedback to confirm interpretations")
            prior = self.interpretations.get(interpretation.id)
            if prior is not None:
                if any(
                    getattr(prior, key) != getattr(interpretation, key)
                    for key in (
                        "event_id",
                        "utterance_ref",
                        "context_ref",
                        "goal_ref",
                        "proposal_ref",
                        "revision",
                        "supersedes",
                    )
                ):
                    raise ValueError("Interpretation identity collision")
                self._trace(
                    "interpret", {"interpretation_id": prior.id, "replayed": True}, success=True
                )
                return prior.model_dump(mode="json")
            if self.event_refinements.get(interpretation.event_id, 0) >= 2:
                raise CARLError("Turn refinement budget exceeded", code="carl.semantic.budget")
            for name in ("utterance_ref", "context_ref", "goal_ref", "proposal_ref"):
                self._reference(getattr(interpretation, name))
            self.event_refinements[interpretation.event_id] = (
                self.event_refinements.get(interpretation.event_id, 0) + 1
            )
            self.interpretations[interpretation.id] = interpretation
            self._trace("interpret", {"interpretation_id": interpretation.id}, success=True)
            return interpretation.model_dump(mode="json")
        except (ValueError, KeyError, OSError, CARLError, RuntimeError):
            self._trace("interpret", refs, success=False)
            raise

    def _retain_reference(self, ref_id: str) -> dict[str, Any]:
        ref = self._reference(ref_id)
        raw = self.session.data_vault.read(ref)
        descriptor = self._artifact(raw, hashlib.sha256(raw).hexdigest())
        durable = DataRef.model_validate({k: v for k, v in descriptor.items() if k != "expires_at"})
        durable = durable.model_copy(
            update={"ref_id": ref.ref_id, "created_at": ref.created_at, "ttl_s": ref.ttl_s}
        )
        self.session.data_vault.restore_file(durable)
        return durable.describe()

    def interpretation_feedback(
        self,
        interpretation_id: str,
        feedback_ref: str,
        confirmed: bool,
        artifact_ref: str | None = None,
        correction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply explicit feedback atomically against concurrent retries."""
        with self._lock:
            return self._interpretation_feedback(
                interpretation_id, feedback_ref, confirmed, artifact_ref, correction
            )

    def _interpretation_feedback(
        self,
        interpretation_id: str,
        feedback_ref: str,
        confirmed: bool,
        artifact_ref: str | None = None,
        correction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Confirm, reject or supersede a candidate from explicit speaker feedback."""
        refs: dict[str, Any] = {
            "interpretation_id": interpretation_id,
            "feedback_ref": feedback_ref,
        }
        try:
            original = self.interpretations[interpretation_id]
            self._reference(feedback_ref)
            if confirmed:
                if artifact_ref is None:
                    raise ValueError("Confirmed artifact required")
                self._reference(artifact_ref)
            proposed = original
            if correction is not None:
                proposed = Interpretation.model_validate(
                    {
                        **correction,
                        "event_id": original.event_id,
                        "context_ref": correction.get("context_ref", original.context_ref),
                        "goal_ref": correction.get("goal_ref", original.goal_ref),
                        "utterance_ref": original.utterance_ref,
                        "supersedes": original.id,
                        "revision": original.revision + 1,
                    }
                )
                if proposed.id == original.id:
                    raise ValueError("A correction must be a successor")
                prior = self.interpretations.get(proposed.id)
                if prior is not None and (
                    prior.supersedes != original.id or prior.proposal_ref != proposed.proposal_ref
                ):
                    raise ValueError("Successor identity collision")
                for name in ("proposal_ref", "context_ref", "goal_ref"):
                    self._reference(getattr(proposed, name))
            accepted = Interpretation.model_validate(
                {
                    **proposed.model_dump(),
                    "feedback": "confirmed" if confirmed else "rejected",
                    "feedback_ref": feedback_ref,
                    "confirmed_artifact": artifact_ref if confirmed else None,
                }
            )
            from carl_core.memory import MemoryStore

            from carl_studio.consent import ConsentManager

            captured = ConsentManager().capture_allowed(
                session_id=self.session.chain.chain_id, workspace=self.session.workspace
            )
            if accepted.feedback == "confirmed" and captured:
                from carl_studio.settings import carl_home

                predecessor = self.interpretations.get(accepted.supersedes or "")
                rejected_ref = predecessor.proposal_ref if predecessor is not None else None
                descriptors = [
                    self._retain_reference(ref_id)
                    for ref_id in (
                        accepted.utterance_ref,
                        accepted.context_ref,
                        accepted.goal_ref,
                        accepted.proposal_ref,
                        rejected_ref or original.proposal_ref,
                        feedback_ref,
                        artifact_ref or "",
                    )
                ]
                committed = MemoryStore(carl_home() / "memory").commit_once(
                    accepted.commit_key,
                    self.session.data_vault.read(self._reference(accepted.proposal_ref)).decode(
                        "utf-8"
                    ),
                    metadata={
                        "interpretation": accepted.model_dump(mode="json"),
                        "interpretation_id": accepted.id,
                        "source_artifacts": descriptors,
                        "capture_kind": "accepted_interpretation",
                        "rejected_proposal_ref": rejected_ref,
                    },
                )
                refs["memory_id"] = committed.id
            if accepted.feedback == "rejected" and captured:
                from carl_studio.settings import carl_home

                descriptor = self._retain_reference(feedback_ref)
                invalidation = MemoryStore(carl_home() / "memory").commit_once(
                    content_hash(
                        {
                            "interpretation_id": accepted.id,
                            "feedback_ref": feedback_ref,
                            "decision": "rejected",
                        }
                    ),
                    "",
                    metadata={
                        "capture_kind": "interpretation_feedback",
                        "interpretation_id": accepted.id,
                        "source_artifacts": [descriptor],
                    },
                )
                refs["feedback_memory_id"] = invalidation.id
            refs["captured"] = captured
            if correction is not None:
                self.interpretations[original.id] = Interpretation.model_validate(
                    {**original.model_dump(), "feedback": "rejected", "feedback_ref": feedback_ref}
                )
            self.interpretations[accepted.id] = accepted
            refs["interpretation_id"] = accepted.id
            self._trace("feedback", refs, success=True)
            return {**accepted.model_dump(mode="json"), **refs}
        except (ValueError, KeyError, OSError, CARLError, RuntimeError):
            self._trace("feedback", refs, success=False)
            raise

    @staticmethod
    def tool_schemas() -> list[dict[str, Any]]:
        return [
            {
                "name": "encode_data",
                "description": "Encode ordered semantic data into an artifact reference.",
                "input_schema": {
                    "type": "object",
                    "properties": {"data": SemanticInput.model_json_schema()},
                    "required": ["data"],
                },
            },
            {
                "name": "interpret",
                "description": "Record an unconfirmed candidate interpretation.",
                "input_schema": {
                    "type": "object",
                    "properties": {"record": Interpretation.model_json_schema()},
                    "required": ["record"],
                },
            },
            {
                "name": "interpretation_feedback",
                "description": "Record explicit confirmation or correction.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "interpretation_id": {"type": "string"},
                        "feedback_ref": {"type": "string"},
                        "confirmed": {"type": "boolean"},
                        "artifact_ref": {"type": ["string", "null"]},
                        "correction": {"type": ["object", "null"]},
                    },
                    "required": ["interpretation_id", "feedback_ref", "confirmed"],
                },
            },
        ]

    def register(self, dispatcher: Any) -> list[str]:
        from carl_studio.handles.bundle import make_handler

        names: list[str] = []
        for schema in self.tool_schemas():
            name = schema["name"]
            dispatcher.register(name, make_handler(getattr(self, name)))
            names.append(name)
        return names
