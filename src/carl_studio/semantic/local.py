"""Bind an isolated encoder to Session's subprocess and resource owners."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote, urlparse

from carl_core.hashing import content_hash

from .types import Carrier, EncoderBinding, ExecutionBinding, SemanticInput

if TYPE_CHECKING:
    from carl_studio.session import Session


def source_identity(model: Path) -> tuple[str, str]:
    from carl_studio.training.preparation import file_hash

    files = {
        str(p.relative_to(model)): file_hash(p) for p in sorted(model.rglob("*")) if p.is_file()
    }
    if not files or "model.safetensors" not in files:
        raise ValueError("Local encoder weights required")
    manifest_path = model / "artifact-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        for descriptor in manifest.get("files", []):
            name = descriptor["path"]
            if name in files and files[name] != descriptor["sha256"]:
                raise ValueError("Encoder artifact disagrees with its pinned manifest")
    processors = {k: v for k, v in files.items() if k != "model.safetensors"}
    return content_hash(files), content_hash(processors)


def invoke(
    session: Session,
    interpreter: Path,
    operation: str,
    request: dict[str, Any] | None = None,
    *,
    timeout: float = 120,
) -> dict[str, Any]:
    try:
        from carl_encoders import worker as encoder_worker
    except ImportError as exc:
        raise RuntimeError("Install carl-studio[encoders] to use the encoder worker") from exc

    worker = Path(encoder_worker.__file__).resolve()
    if operation == "fit_metadata":
        worker, operation = worker.with_name("fit_worker.py"), "metadata"
    temporary: Path | None = None
    argv = [str(interpreter), "-P", str(worker), operation]
    if request is not None:
        fd, name = tempfile.mkstemp(prefix="carl-encoder-", suffix=".json")
        temporary = Path(name)
        with os.fdopen(fd, "w") as stream:
            json.dump(request, stream)
        argv += ["--request", name]
    data = request or {}
    worker_settings: dict[str, Any] = data.get("settings") or {}
    execution_settings: dict[str, Any] = worker_settings.get("execution") or {}
    selected = data.get("device", execution_settings.get("device", "cpu"))
    environment = None
    if selected == "cpu":
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
        }
    resource = session.subprocess_toolkit.spawn(argv, ttl_s=int(timeout) + 10, env=environment)
    try:
        result = session.subprocess_toolkit.wait(resource["ref_id"], timeout_s=timeout)
        if result["exit_code"] != 0:
            raise RuntimeError("Isolated encoder failed; inspect worker stderr reference")
        output = session.data_toolkit.read_text(
            result["stdout_ref"]["ref_id"], max_bytes=16 * 1024 * 1024
        )["text"]
        return json.loads(output)
    finally:
        session.subprocess_toolkit.terminate(resource["ref_id"])
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def bind_local(
    session: Session,
    model: Path,
    interpreter: Path,
    *,
    device: str | None = None,
    checkpoint: Path | None = None,
    expected_execution: ExecutionBinding | None = None,
) -> None:
    """Use exact offline model and environment bytes without installing anything."""
    device = device or (expected_execution.device if expected_execution is not None else "cpu")
    artifact, processor = source_identity(model)
    checkpoint_identity = None
    state: dict[str, Any] = {}
    logical_checkpoint: dict[str, Any] = {}
    if checkpoint is not None:
        checkpoint_identity = source_identity_checkpoint(checkpoint)
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        logical_checkpoint = invoke(
            session, interpreter, "checkpoint_metadata", {"checkpoint": str(checkpoint)}
        )
        if state.get("settings", {}).get("mode") == "adapter":
            artifact = content_hash({"base": artifact, "adapter": logical_checkpoint["encoder_id"]})
    manifest = json.loads((model / "artifact-manifest.json").read_text())
    binding = EncoderBinding(
        model=manifest["model"],
        revision=manifest["revision"],
        artifact_sha256=artifact,
        processor_sha256=processor,
    )
    dtype = (
        expected_execution.dtype
        if expected_execution is not None
        else ("float32" if device == "cpu" else "bfloat16")
    )
    runtime = invoke(session, interpreter, "metadata", {"device": device, "dtype": dtype})
    if expected_execution is not None and any(
        name.startswith("carl.encoder.fit.") for name in expected_execution.dependencies
    ):
        runtime["dependencies"].update(invoke(session, interpreter, "fit_metadata"))
    runtime.setdefault("device", device)
    runtime.setdefault("dtype", dtype)
    execution = ExecutionBinding(
        **runtime,
        processor_sha256=processor,
    )

    if expected_execution is not None and any(
        getattr(execution, key) != getattr(expected_execution, key)
        for key in (
            "interpreter",
            "interpreter_sha256",
            "dependencies",
            "processor_sha256",
            "device",
            "dtype",
        )
    ):
        raise ValueError("Active worker execution identity changed")

    def validate_sources() -> None:
        base_artifact, current_processor = source_identity(model)
        expected_artifact = (
            content_hash({"base": base_artifact, "adapter": logical_checkpoint["encoder_id"]})
            if checkpoint is not None and state.get("settings", {}).get("mode") == "adapter"
            else base_artifact
        )
        if expected_artifact != artifact or current_processor != processor:
            raise ValueError("Encoder source bytes changed")
        if checkpoint is not None and source_identity_checkpoint(checkpoint) != checkpoint_identity:
            raise ValueError("Candidate checkpoint bytes changed")
        if expected_execution is not None:
            from carl_studio.training.preparation import file_hash

            if file_hash(Path(execution.interpreter)) != expected_execution.interpreter_sha256:
                raise ValueError("Worker interpreter source bytes changed")

    def encode(request: SemanticInput) -> Carrier:
        data = request.model_dump(mode="json")
        temporary_sources: list[Path] = []
        try:
            for part in data["parts"]:
                if part["source_ref"]:
                    ref = next(
                        r
                        for r in session.data_vault.list_refs()
                        if str(r.ref_id) == part["source_ref"]
                    )
                    if ref.kind == "file":
                        parsed = urlparse(ref.uri)
                        if parsed.scheme != "file" or parsed.netloc:
                            raise ValueError("Media file source is unavailable")
                        part["path"] = unquote(parsed.path)
                    else:
                        raw = session.data_vault.read(ref)
                        suffix = mimetypes.guess_extension(ref.content_type or "") or {
                            "image": ".png",
                            "audio": ".wav",
                            "video": ".mp4",
                        }.get(part["modality"], ".txt")
                        fd, name = tempfile.mkstemp(suffix=suffix, prefix="carl-media-")
                        with os.fdopen(fd, "wb") as stream:
                            stream.write(raw)
                        temporary_sources.append(Path(name))
                        part["path"] = name
                    if (
                        hashlib.sha256(Path(part["path"]).read_bytes()).hexdigest()
                        != part["source_sha256"]
                    ):
                        raise ValueError("Media source bytes changed")
            materialize_windows(session, data)
            validate_sources()
            response = invoke(
                session,
                interpreter,
                "encode",
                {
                    "model": str(model),
                    "device": device,
                    "dtype": dtype,
                    "input": data,
                    "checkpoint": str(checkpoint) if checkpoint else None,
                },
            )
            if response.get("head_weights") is not None:
                heads = response["head_weights"]
                session.semantic.head_weights = (
                    {
                        int(d): tuple(weights)
                        for d, weights in cast(dict[str, list[float]], heads).items()
                    }
                    if isinstance(heads, dict)
                    else tuple(heads)
                )
            return Carrier(values=tuple(response["values"]))
        finally:
            for path in temporary_sources:
                path.unlink(missing_ok=True)

    session.semantic.bind(
        binding,
        execution,
        encode,
        validator=validate_sources,
        generation=content_hash({"space": binding.space_id, "heads": checkpoint_identity}),
    )


def source_identity_checkpoint(checkpoint: Path) -> str:
    from carl_studio.training.preparation import file_hash

    return content_hash(
        {name: file_hash(checkpoint / name) for name in ("encoder_state.pt", "trainer_state.json")}
    )


def materialize_windows(
    session: Session, data: dict[str, Any], *, deadline: float | None = None
) -> None:
    """Derive explicit media windows through the existing subprocess owner."""
    import shutil
    import time

    from carl_studio.settings import carl_home
    from carl_studio.training.preparation import file_hash

    for part in data["parts"]:
        if part.get("start_s") is None:
            continue
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("Media windows require an existing local FFmpeg installation")
        source = Path(part["path"])
        if file_hash(source) != part["source_sha256"]:
            raise ValueError("Window source bytes changed")
        recipe = {
            "source_sha256": part["source_sha256"],
            "start_s": part["start_s"],
            "end_s": part["end_s"],
            "modality": part["modality"],
            "audio": "mono-16000",
            "video": "fps-1",
            "processor": file_hash(Path(executable)),
        }
        key = content_hash(recipe)
        root = carl_home() / "artifacts" / "semantic" / "media"
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        suffix = ".wav" if part["modality"] == "audio" else ".mp4"
        fd, name = tempfile.mkstemp(dir=root, suffix=suffix)
        os.close(fd)
        temporary = Path(name)
        destination = root / (key + suffix)
        arguments = [
            executable,
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-protocol_whitelist",
            "file,pipe",
            "-ss",
            str(part["start_s"]),
            "-i",
            str(source),
            "-t",
            str(part["end_s"] - part["start_s"]),
        ]
        arguments += (
            ["-ac", "1", "-ar", "16000"] if part["modality"] == "audio" else ["-vf", "fps=1", "-an"]
        )
        arguments.append(str(temporary))
        resource = session.subprocess_toolkit.spawn(arguments, ttl_s=70)
        try:
            remaining = min(60.0, deadline - time.monotonic()) if deadline is not None else 60.0
            if remaining <= 0:
                raise TimeoutError("Media preprocessing exceeded the encoder runtime budget")
            result = session.subprocess_toolkit.wait(resource["ref_id"], timeout_s=remaining)
            if result["exit_code"] != 0 or temporary.stat().st_size == 0:
                raise RuntimeError("Media segmentation failed")
            digest = file_hash(temporary)
            if destination.exists() and file_hash(destination) != digest:
                raise ValueError("Media derivative source bytes changed")
            if not destination.exists():
                os.replace(temporary, destination)
            ref = session.data_vault.open_file(destination)
            ref = ref.model_copy(update={"sha256": digest})
            session.data_vault.restore_file(ref)
            part["path"] = str(destination)
            part["segment_applied"] = True
            part["segment_ref"] = ref.describe()
            session.semantic._trace(
                "segment",
                {
                    "source_ref": part["source_ref"],
                    "artifact": ref.describe(),
                    "processing_recipe": recipe,
                },
                success=True,
            )
        finally:
            session.subprocess_toolkit.terminate(resource["ref_id"])
            temporary.unlink(missing_ok=True)
