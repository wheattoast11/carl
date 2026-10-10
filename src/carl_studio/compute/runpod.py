"""RunPod execution on an admitted allocation through bound provider and SSH ports."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import shlex
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Protocol, cast

from carl_studio.adapters._common import JobState, load_state, save_state, state_dir
from carl_studio.adapters.protocol import BackendStatus

_GPU_MAP = {
    "l4x1": "NVIDIA L4",
    "l40sx1": "NVIDIA L40S",
    "a10g-large": "NVIDIA A10G",
    "a100-large": "NVIDIA A100 80GB",
    "h100": "NVIDIA H100 80GB HBM3",
}


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class RunPodAllocation:
    """Exact allocation and runtime returned by the existing admission owner."""

    pod_id: str
    hardware: str
    gpu_count: int
    image: str
    python_executable: str
    runtime_lock: Mapping[str, str]
    runtime_ref: str
    qualification_ref: str
    authority_ref: str
    supervisor_ref: str
    remote_root: str
    deadline_epoch_s: int
    shutdown_seconds: int
    allocation_ref: str
    budget_ref: str


class RunPodProviderPort(Protocol):
    """Provider effects retain their existing admission and reservation owner."""

    async def provision(self, hardware: str, timeout: int) -> RunPodAllocation: ...
    async def get_pod(self, pod_id: str) -> Mapping[str, Any]: ...
    async def terminate_pod(self, pod_id: str) -> None: ...


class RunPodSSHRunner(Protocol):
    """Run through the existing SSH connection, preserving its host verification."""

    async def __call__(self, pod_id: str, command: str, payload: str, timeout: float) -> str: ...


class RunPodJobStatePort(Protocol):
    """Atomically claim and reopen JobState at the existing execution owner."""

    def claim(self, state: JobState) -> tuple[JobState, bool]: ...
    def load(self, job_id: str) -> JobState: ...
    def save(self, state: JobState) -> None: ...


class AtomicRunPodJobState:
    """Serialize claims through the existing CARL adapter state owner."""

    def _lock(self, job_id: str) -> int:
        if not re.fullmatch(r"runpod-[0-9a-f]{24}", job_id):
            raise ValueError("Invalid RunPod job identity")
        root = state_dir("runpod")
        if root.is_symlink() or root.stat().st_uid != os.getuid():
            raise ValueError("RunPod job-state directory must be owned")
        os.chmod(root, 0o700)
        descriptor = os.open(
            root / (job_id + ".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    def claim(self, state: JobState) -> tuple[JobState, bool]:
        descriptor = self._lock(state.run_id)
        try:
            path = state_dir("runpod") / (state.run_id + ".json")
            if path.exists():
                current = load_state("runpod", state.run_id)
                if current.backend != "runpod" or current.run_id != state.run_id:
                    raise ValueError("RunPod recorded job identity differs")
                if current.raw.get("binding") != state.raw.get("binding"):
                    raise ValueError("RunPod request conflicts with its recorded execution")
                return current, False
            save_state(state)
            return load_state("runpod", state.run_id), True
        finally:
            os.close(descriptor)

    def load(self, job_id: str) -> JobState:
        descriptor = self._lock(job_id)
        try:
            return load_state("runpod", job_id)
        finally:
            os.close(descriptor)

    def save(self, state: JobState) -> None:
        descriptor = self._lock(state.run_id)
        try:
            current = load_state("runpod", state.run_id)
            if current.backend != "runpod" or current.run_id != state.run_id:
                raise ValueError("RunPod recorded job identity differs")
            if current.raw.get("binding") != state.raw.get("binding"):
                raise ValueError("RunPod recorded binding changed")
            if BackendStatus.is_terminal(current.status) and (
                current.status != state.status
                or current.raw.get("observation") != state.raw.get("observation")
            ):
                raise ValueError("RunPod terminal observation is immutable")
            observation = state.raw.get("observation")
            if observation is not None:
                prior = current.raw.get("observation", {})
                if prior.get("pid") is not None and observation.get("pid") is None:
                    observation = {
                        **observation,
                        "pid": prior["pid"],
                        "start_ticks": prior["start_ticks"],
                    }
                    state.raw["observation"] = observation
                RunPodBackend.validate_worker_observation(current, observation)
                if RunPodBackend.worker_status(observation["phase"]) != state.status:
                    raise ValueError("RunPod status differs from its worker observation")
                state.pid = observation.get("pid")
            elif current.raw.get("observation") is not None:
                state.raw["observation"] = current.raw["observation"]
                state.pid, state.status = current.pid, current.status
            for flag in ("cancel_requested", "reconcile_required"):
                reconciled = (
                    flag == "reconcile_required"
                    and state.raw.get(flag) is False
                    and observation is not None
                    and observation["phase"] != "unknown"
                )
                if current.raw.get(flag) and not reconciled:
                    state.raw[flag] = True
            save_state(state)
        finally:
            os.close(descriptor)


class SSHCommandResult(Protocol):
    stdout: str | bytes


class SSHConnection(Protocol):
    async def run(
        self, command: str, *, input: str, check: bool, timeout: float
    ) -> SSHCommandResult: ...


class RunPodSSHConnection:
    """Bind an already verified asyncssh-compatible connection to one pod."""

    def __init__(self, pod_id: str, connection: SSHConnection) -> None:
        self._pod_id, self._connection = pod_id, connection

    async def __call__(self, pod_id: str, command: str, payload: str, timeout: float) -> str:
        if pod_id != self._pod_id:
            raise ValueError("RunPod SSH connection belongs to another pod")
        result = await self._connection.run(command, input=payload, check=True, timeout=timeout)
        return result.stdout.decode() if isinstance(result.stdout, bytes) else result.stdout


_COMMON = """
import hashlib, importlib.metadata, json, os, pathlib, platform, signal, subprocess, sys, time

def write(path, value):
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def identity(pid):
    try:
        stat = pathlib.Path("/proc", str(pid), "stat").read_text().rsplit(")", 1)[1].split()
        return int(stat[19]) if stat[0] != "Z" else None
    except FileNotFoundError:
        return None

def runtime(binding):
    lock = binding["runtime_lock"]
    digest = hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if binding["runtime_ref"] != "sha256:" + digest:
        raise ValueError("RunPod runtime lock digest differs")
    observed = {name: platform.python_version() if name == "python" else importlib.metadata.version(name) for name in lock}
    if observed != lock or os.path.realpath(sys.executable) != os.path.realpath(binding["python_executable"]):
        raise ValueError("RunPod runtime differs from its admitted lock")

def group_alive(pgid):
    for path in pathlib.Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if stat[0] != "Z" and int(stat[2]) == pgid:
                return True
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return False

def stop_group(child, binding):
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass
        deadline = min(time.time() + binding["shutdown_seconds"], binding["deadline_epoch_s"])
        while group_alive(child.pid) and time.time() < deadline:
            child.poll()
            time.sleep(0.02)
        if not group_alive(child.pid):
            child.wait(timeout=binding["shutdown_seconds"])
            return
    raise RuntimeError("RunPod child group remains unresolved")
"""

_WORKER = (
    _COMMON
    + """
root = pathlib.Path(sys.argv[1])
binding = json.loads((root / "request.json").read_text())
pid = os.getpid()
start = identity(pid)
launch = {"binding": binding, "pid": pid, "start_ticks": start}
child = None
cancelled = False
cleanup_attempted = False

def cancel(signum, frame):
    global cancelled
    cancelled = True

signal.signal(signal.SIGTERM, cancel)
write(root / "launch.json", launch)
try:
    runtime(binding)
    script = root / "training.py"
    if hashlib.sha256(script.read_bytes()).hexdigest() != binding["script_sha256"]:
        raise ValueError("RunPod script digest differs")
    if time.time() >= binding["work_deadline_epoch_s"]:
        raise TimeoutError("RunPod work deadline ended")
    if cancelled:
        raise InterruptedError("RunPod launch was canceled")
    child = subprocess.Popen([binding["python_executable"], str(script)], start_new_session=True)
    timed_out = False
    while child.poll() is None or group_alive(child.pid):
        timed_out = time.time() >= binding["work_deadline_epoch_s"]
        if cancelled or timed_out:
            cleanup_attempted = True
            stop_group(child, binding)
            break
        time.sleep(min(0.1, max(0, binding["work_deadline_epoch_s"] - time.time())))
    code = child.returncode
    phase = "canceled" if cancelled else "timed_out" if timed_out else "complete" if code == 0 else "failed"
except BaseException:
    if child is not None and (child.poll() is None or group_alive(child.pid)):
        if cleanup_attempted:
            raise
        stop_group(child, binding)
    phase, code = "canceled" if cancelled else "failed", child.returncode if child is not None else None
write(root / "terminal.json", {**launch, "phase": phase, "exit_code": code})
"""
)

_REMOTE = (
    _COMMON
    + """
payload = json.loads(sys.stdin.read())
binding = payload["binding"]
root = pathlib.Path(binding["remote_root"]) / binding["job_id"]
for path in [root, *root.parents]:
    if path.is_symlink():
        raise ValueError("RunPod job path is symlinked")
root.mkdir(mode=0o700, parents=True, exist_ok=True)
if root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
    raise ValueError("RunPod job directory must be private and owned")
request = root / "request.json"
launch_path, terminal_path = root / "launch.json", root / "terminal.json"

def observe():
    if request.exists() and json.loads(request.read_text()) != binding:
        raise ValueError("RunPod execution identity conflict")
    for path in (terminal_path, launch_path):
        if path.exists():
            value = json.loads(path.read_text())
            if value.get("binding") != binding:
                raise ValueError("RunPod worker binding differs")
            if path == terminal_path:
                return value
            return {**value, "phase": "running" if identity(value["pid"]) == value["start_ticks"] else "unknown"}
    return {"binding": binding, "phase": "unknown"}

if payload["action"] == "launch":
    runtime(binding)
    if hashlib.sha256(payload["script"].encode()).hexdigest() != binding["script_sha256"]:
        raise ValueError("RunPod submitted script digest differs")
    if hashlib.sha256(payload["worker"].encode()).hexdigest() != binding["worker_sha256"]:
        raise ValueError("RunPod worker source digest differs")
    try:
        descriptor = os.open(root / "launch.intent", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        result = observe()
    else:
        os.close(descriptor)
        write(request, binding)
        descriptor = os.open(root / "training.py", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(payload["script"])
            stream.flush()
            os.fsync(stream.fileno())
        with (root / "worker.log").open("xb") as log:
            os.chmod(root / "worker.log", 0o600)
            subprocess.Popen([binding["python_executable"], "-c", payload["worker"], str(root)], stdout=log, stderr=log, start_new_session=True)
        deadline = min(time.time() + 2, binding["work_deadline_epoch_s"])
        result = observe()
        while result["phase"] == "unknown" and time.time() < deadline:
            time.sleep(0.01)
            result = observe()
elif payload["action"] == "stop":
    result = observe()
    if result["phase"] == "running" and identity(result["pid"]) == result["start_ticks"]:
        os.kill(result["pid"], signal.SIGTERM)
        result = {**result, "phase": "cancel_requested"}
elif payload["action"] == "logs":
    result = observe()
    path = root / "worker.log"
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        result["logs"] = stream.read(65536).decode(errors="replace").splitlines()[-payload["tail"]:]
else:
    result = observe()
print(json.dumps(result))
"""
)


class RunPodBackend:
    """Execute admitted scripts; allocation status never proves worker completion."""

    def __init__(
        self,
        *,
        provider: RunPodProviderPort | None = None,
        ssh_runner: RunPodSSHRunner | None = None,
        job_state: RunPodJobStatePort | None = None,
        allocation: RunPodAllocation | None = None,
    ) -> None:
        self._provider = provider
        self._ssh_runner = ssh_runner
        self._job_state = job_state
        if allocation is not None:
            self._validate_allocation(allocation, allocation.hardware)
        self._allocation = self._snapshot(allocation) if allocation is not None else None

    @property
    def name(self) -> str:
        return "runpod"

    def _ports(self) -> tuple[RunPodProviderPort, RunPodSSHRunner, RunPodJobStatePort]:
        if self._provider is None or self._ssh_runner is None or self._job_state is None:
            raise RuntimeError(
                "RunPod requires admitted provider, SSH transport and job-state ports"
            )
        return self._provider, self._ssh_runner, self._job_state

    async def provision(self, hardware: str, timeout: int) -> str:
        provider, _, _ = self._ports()
        if type(timeout) is not int or timeout <= 0:
            raise ValueError("RunPod requires a finite positive timeout")
        allocation = self._allocation or await provider.provision(hardware, timeout)
        self._validate_allocation(allocation, hardware)
        if (
            not time.time() + 2 * allocation.shutdown_seconds
            < allocation.deadline_epoch_s
            <= time.time() + timeout
        ):
            raise ValueError("RunPod allocation differs from its admitted time budget")
        self._allocation = self._snapshot(allocation)
        return allocation.pod_id

    @staticmethod
    def _validate_allocation(allocation: RunPodAllocation, hardware: str) -> None:
        refs = (
            allocation.runtime_ref,
            allocation.qualification_ref,
            allocation.authority_ref,
            allocation.supervisor_ref,
            allocation.allocation_ref,
            allocation.budget_ref,
        )
        if (
            allocation.hardware not in {hardware, _GPU_MAP.get(hardware, hardware)}
            or type(allocation.gpu_count) is not int
            or allocation.gpu_count < 1
            or not allocation.pod_id
            or not re.fullmatch(r".+@sha256:[0-9a-f]{64}", allocation.image)
            or not all(re.fullmatch(r"sha256:[0-9a-f]{64}", ref) for ref in refs)
            or not allocation.python_executable.startswith("/")
            or not allocation.remote_root.startswith("/")
            or ".." in allocation.remote_root.split("/")
            or not allocation.runtime_lock.get("python")
            or allocation.runtime_ref != "sha256:" + _digest(dict(allocation.runtime_lock))
            or type(allocation.shutdown_seconds) is not int
            or allocation.shutdown_seconds < 1
            or type(allocation.deadline_epoch_s) is not int
            or allocation.deadline_epoch_s <= 2 * allocation.shutdown_seconds
        ):
            raise ValueError(
                "RunPod allocation differs from its admitted hardware, runtime or budget"
            )

    @staticmethod
    def _snapshot(allocation: RunPodAllocation) -> RunPodAllocation:
        if allocation.runtime_ref != "sha256:" + _digest(dict(allocation.runtime_lock)):
            raise ValueError("RunPod admitted runtime lock digest differs")
        return replace(allocation, runtime_lock=MappingProxyType(dict(allocation.runtime_lock)))

    async def _observe(self, state: JobState, action: str, **payload: object) -> Mapping[str, Any]:
        _, runner, owner = self._ports()
        binding = state.raw["binding"]
        request = {"action": action, "binding": binding, **payload}
        command = shlex.join([binding["python_executable"], "-c", _REMOTE])
        deadline = (
            binding["work_deadline_epoch_s"] if action == "launch" else binding["deadline_epoch_s"]
        )
        timeout = min(10.0, max(0.1, deadline - time.time()))
        response = await runner(binding["pod_id"], command, json.dumps(request), timeout)
        value = self.validate_worker_observation(state, json.loads(response))
        phase = value["phase"]
        state.raw["observation"] = {key: item for key, item in value.items() if key != "logs"}
        if phase != "unknown":
            state.raw["reconcile_required"] = False
        if action == "stop" and phase == "cancel_requested":
            state.raw["cancel_requested"] = True
        state.pid = value.get("pid")
        state.status = self.worker_status(phase)
        owner.save(state)
        return value

    @staticmethod
    def worker_status(phase: str) -> str:
        return {
            "complete": BackendStatus.COMPLETED,
            "failed": BackendStatus.FAILED,
            "timed_out": BackendStatus.FAILED,
            "canceled": BackendStatus.CANCELED,
        }.get(phase, BackendStatus.RUNNING if phase == "running" else BackendStatus.PENDING)

    @staticmethod
    def _load(owner: RunPodJobStatePort, job_id: str) -> JobState:
        if not re.fullmatch(r"runpod-[0-9a-f]{24}", job_id):
            raise ValueError("Invalid RunPod job identity")
        state = owner.load(job_id)
        if (
            state.backend != "runpod"
            or state.run_id != job_id
            or state.raw.get("binding", {}).get("job_id") != job_id
        ):
            raise ValueError("RunPod recorded job identity differs")
        return state

    def _cached_terminal(self, state: JobState) -> None:
        value = self.validate_worker_observation(state, state.raw.get("observation"))
        if self.worker_status(value["phase"]) != state.status:
            raise ValueError("RunPod terminal state lacks a matching worker terminal")

    @staticmethod
    def validate_worker_observation(state: JobState, candidate: object) -> dict[str, Any]:
        if not isinstance(candidate, dict):
            raise TypeError("RunPod worker observation does not bind its execution")
        value = cast(dict[str, Any], candidate)
        if value.get("binding") != state.raw["binding"]:
            raise ValueError("RunPod worker observation does not bind its execution")
        phase = value.get("phase")
        if phase not in {
            "unknown",
            "running",
            "complete",
            "failed",
            "canceled",
            "timed_out",
            "cancel_requested",
        }:
            raise ValueError("RunPod worker phase is invalid")
        if (
            phase != "unknown"
            or value.get("pid") is not None
            or value.get("start_ticks") is not None
        ):
            if (
                type(value.get("pid")) is not int
                or value["pid"] <= 0
                or type(value.get("start_ticks")) is not int
                or value["start_ticks"] <= 0
            ):
                raise ValueError("RunPod worker process identity is missing")
            prior = state.raw.get("observation", {})
            if prior.get("pid") is not None and (prior["pid"], prior["start_ticks"]) != (
                value["pid"],
                value["start_ticks"],
            ):
                raise ValueError("RunPod worker process identity changed")
        if phase == "complete" and (
            type(value.get("exit_code")) is not int or value["exit_code"] != 0
        ):
            raise ValueError("RunPod successful terminal requires a zero exit code")
        return value

    async def execute(self, script: str, **kwargs: object) -> str:
        _, _, owner = self._ports()
        allocation = self._allocation
        if allocation is None:
            raise RuntimeError("Must provision() before execute()")
        if (
            not script
            or set(kwargs) != {"request_id", "plan_id", "run_id"}
            or not all(isinstance(value, str) and value for value in kwargs.values())
        ):
            raise ValueError(
                "RunPod execution requires script and exact request, plan and run identities"
            )
        job_id = "runpod-" + _digest(kwargs["request_id"])[:24]
        binding = {
            **kwargs,
            "job_id": job_id,
            "pod_id": allocation.pod_id,
            "script_sha256": hashlib.sha256(script.encode()).hexdigest(),
            "runtime_ref": allocation.runtime_ref,
            "runtime_lock": dict(allocation.runtime_lock),
            "qualification_ref": allocation.qualification_ref,
            "authority_ref": allocation.authority_ref,
            "supervisor_ref": allocation.supervisor_ref,
            "allocation_ref": allocation.allocation_ref,
            "budget_ref": allocation.budget_ref,
            "image": allocation.image,
            "gpu_count": allocation.gpu_count,
            "hardware": allocation.hardware,
            "python_executable": allocation.python_executable,
            "remote_root": allocation.remote_root,
            "deadline_epoch_s": allocation.deadline_epoch_s,
            "work_deadline_epoch_s": allocation.deadline_epoch_s - 2 * allocation.shutdown_seconds,
            "shutdown_seconds": allocation.shutdown_seconds,
            "worker_sha256": hashlib.sha256(_WORKER.encode()).hexdigest(),
        }
        state, fresh = owner.claim(JobState(job_id, "runpod", raw={"binding": binding}))
        if state.backend != "runpod" or state.run_id != job_id:
            raise ValueError("RunPod recorded job identity differs")
        if state.raw.get("binding") != binding:
            raise ValueError("RunPod request conflicts with its recorded execution")
        if BackendStatus.is_terminal(state.status):
            self._cached_terminal(state)
            return job_id
        if fresh and time.time() >= allocation.deadline_epoch_s - 2 * allocation.shutdown_seconds:
            state.raw["reconcile_required"] = True
            owner.save(state)
            raise TimeoutError("RunPod work deadline ended")
        try:
            value = await self._observe(
                state, "launch" if fresh else "status", script=script, worker=_WORKER
            )
        except (Exception, asyncio.CancelledError):
            state.raw["reconcile_required"] = True
            owner.save(state)
            raise
        if value["phase"] == "unknown":
            state.raw["reconcile_required"] = True
            owner.save(state)
            raise RuntimeError(
                "RunPod submission requires reconciliation; the recorded attempt will not be launched again"
            )
        return job_id

    async def status(self, job_id: str) -> str:
        provider, _, owner = self._ports()
        state = self._load(owner, job_id)
        if BackendStatus.is_terminal(state.status):
            self._cached_terminal(state)
            return {
                BackendStatus.COMPLETED: "completed",
                BackendStatus.FAILED: "error",
                BackendStatus.CANCELED: "canceled",
            }[state.status]
        pod = await provider.get_pod(state.raw["binding"]["pod_id"])
        if pod.get("desiredStatus") != "RUNNING":
            return "unknown"
        value = await self._observe(state, "status")
        return {
            "complete": "completed",
            "failed": "error",
            "timed_out": "error",
            "canceled": "canceled",
            "running": "running",
        }.get(value["phase"], "unknown")

    async def logs(self, job_id: str, tail: int = 50) -> list[str]:
        _, _, owner = self._ports()
        if type(tail) is not int or not 1 <= tail <= 1000:
            raise ValueError("RunPod log tail must be between 1 and 1000")
        value = await self._observe(self._load(owner, job_id), "logs", tail=tail)
        return list(value.get("logs", []))

    async def stop(self, job_id: str) -> None:
        _, _, owner = self._ports()
        state = self._load(owner, job_id)
        if not BackendStatus.is_terminal(state.status):
            await self._observe(state, "stop")

    async def teardown(self) -> None:
        provider, _, _ = self._ports()
        if self._allocation is not None:
            await provider.terminate_pod(self._allocation.pod_id)
            self._allocation = None
