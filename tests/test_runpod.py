"""RunPod CPU process fixtures; provider allocation and GPU training are not executed."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import platform
import shlex
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from carl_studio.adapters._common import JobState
from carl_studio.compute.runpod import RunPodAllocation, RunPodBackend, _digest


class StateOwner:
    def __init__(self):
        self.rows = {}

    def claim(self, state: JobState):
        if state.run_id in self.rows:
            return self.load(state.run_id), False
        self.save(state)
        return self.load(state.run_id), True

    def load(self, job_id):
        return copy.deepcopy(self.rows[job_id])

    def save(self, state):
        self.rows[state.run_id] = copy.deepcopy(state)


class Provider:
    def __init__(self, allocation):
        self.allocation = allocation
        self.calls = 0
        self.status = "RUNNING"
        self.terminated = []

    async def provision(self, hardware, timeout):
        self.calls += 1
        return self.allocation

    async def get_pod(self, pod_id):
        assert pod_id == self.allocation.pod_id
        return {"desiredStatus": self.status}

    async def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)


class LocalSSH:
    def __init__(self):
        self.actions = []
        self.lose_response = False
        self.mutate_response = None

    async def __call__(self, pod_id, command, payload, timeout):
        assert pod_id == "fixture-pod"
        request = json.loads(payload)
        self.actions.append(request["action"])
        process = await asyncio.create_subprocess_exec(
            *shlex.split(command),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(payload.encode()), timeout)
        if process.returncode:
            raise RuntimeError(stderr.decode())
        if self.lose_response:
            self.lose_response = False
            raise TimeoutError("Fixture lost an executed SSH response")
        value = json.loads(stdout)
        if self.mutate_response:
            value = self.mutate_response(value)
        return json.dumps(value)


@pytest.fixture
def ports(tmp_path):
    lock = {"python": platform.python_version()}
    allocation = RunPodAllocation(
        "fixture-pod",
        "h200",
        2,
        "fixture@sha256:" + "a" * 64,
        sys.executable,
        lock,
        "sha256:" + _digest(lock),
        "sha256:" + "b" * 64,
        "sha256:" + "c" * 64,
        "sha256:" + "d" * 64,
        str(tmp_path / "jobs with spaces"),
        int(time.time()) + 20,
        1,
        "sha256:" + "e" * 64,
        "sha256:" + "f" * 64,
    )
    provider, ssh, state = Provider(allocation), LocalSSH(), StateOwner()
    return provider, ssh, state


async def ready(ports):
    provider, ssh, state = ports
    backend = RunPodBackend(provider=provider, ssh_runner=ssh, job_state=state)
    await backend.provision("h200", 30)
    return backend


async def terminal(backend, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = await backend.status(job_id)
        if status in {"completed", "error", "canceled"}:
            return status
        await asyncio.sleep(0.03)
    raise AssertionError("CPU fixture did not reach a worker terminal")


IDENTITY = {"request_id": "fixture-request", "plan_id": "Eprep_fixture", "run_id": "fixture-run"}


@pytest.mark.asyncio
async def test_unbound_backend_refuses_before_allocation(ports):
    provider, _, _ = ports
    backend = RunPodBackend(provider=provider)
    with pytest.raises(RuntimeError, match="admitted provider"):
        await backend.provision("h200", 30)
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_real_cpu_execution_binds_process_runtime_and_reuses_result(ports, tmp_path):
    backend = await ready(ports)
    marker = tmp_path / "executed.txt"
    script = f"from pathlib import Path\nprint('CPU fixture ran')\nPath({str(marker)!r}).write_text('once')\n"
    job = await backend.execute(script, **IDENTITY)
    assert job != "fixture-pod"
    assert await terminal(backend, job) == "completed"
    row = ports[2].load(job)
    observation = row.raw["observation"]
    binding = row.raw["binding"]
    assert marker.read_text() == "once"
    assert observation["pid"] > 0 and observation["start_ticks"] > 0
    assert binding["script_sha256"] == hashlib.sha256(script.encode()).hexdigest()
    assert binding["runtime_ref"] == ports[0].allocation.runtime_ref
    assert binding["gpu_count"] == 2
    assert binding["deadline_epoch_s"] == ports[0].allocation.deadline_epoch_s
    assert observation["exit_code"] == 0
    assert await backend.execute(script, **IDENTITY) == job
    assert ports[1].actions.count("launch") == 1
    assert "CPU fixture ran" in await backend.logs(job)
    ports[0].status = "EXITED"
    assert await backend.status(job) == "completed"


@pytest.mark.asyncio
async def test_lost_launch_response_reconciles_same_worker_without_launch(ports):
    backend = await ready(ports)
    ports[1].lose_response = True
    script = "import time\ntime.sleep(0.1)\n"
    with pytest.raises(TimeoutError):
        await backend.execute(script, **IDENTITY)
    job = await backend.execute(script, **IDENTITY)
    assert await terminal(backend, job) == "completed"
    assert ports[1].actions.count("launch") == 1
    assert ports[1].actions.count("status") >= 1
    assert ports[2].load(job).raw["reconcile_required"] is False


@pytest.mark.asyncio
async def test_exited_allocation_and_missing_worker_never_mean_completion(ports):
    backend = await ready(ports)
    job = await backend.execute("import time\ntime.sleep(0.3)\n", **IDENTITY)
    ports[0].status = "EXITED"
    assert await backend.status(job) == "unknown"
    ports[0].status = "RUNNING"
    assert await terminal(backend, job) == "completed"
    row = ports[2].load(job)
    root = Path(row.raw["binding"]["remote_root"]) / job
    (root / "terminal.json").unlink()
    row.status = "running"
    ports[2].save(row)
    assert await backend.status(job) == "unknown"


@pytest.mark.asyncio
async def test_changed_script_and_foreign_terminal_refuse(ports):
    backend = await ready(ports)
    job = await backend.execute("import time\ntime.sleep(0.2)\n", **IDENTITY)
    with pytest.raises(ValueError, match="conflicts"):
        await backend.execute("print('different')", **IDENTITY)
    ports[1].mutate_response = lambda value: {
        **value,
        "binding": {**value["binding"], "run_id": "foreign"},
    }
    with pytest.raises(ValueError, match="does not bind"):
        await backend.status(job)
    ports[1].mutate_response = None
    assert await terminal(backend, job) == "completed"


@pytest.mark.asyncio
async def test_reused_pid_start_identity_refuses(ports):
    backend = await ready(ports)
    job = await backend.execute("import time\ntime.sleep(0.2)\n", **IDENTITY)
    ports[1].mutate_response = lambda value: {**value, "start_ticks": value["start_ticks"] + 1}
    with pytest.raises(ValueError, match="process identity changed"):
        await backend.status(job)
    ports[1].mutate_response = None
    assert await terminal(backend, job) == "completed"


@pytest.mark.asyncio
async def test_worker_failure_and_acknowledged_cancellation(ports):
    backend = await ready(ports)
    failed = await backend.execute("raise RuntimeError('CPU fixture failure')\n", **IDENTITY)
    assert await terminal(backend, failed) == "error"
    identity = {**IDENTITY, "request_id": "cancel-fixture"}
    canceled = await backend.execute("import time\ntime.sleep(30)\n", **identity)
    await backend.stop(canceled)
    assert ports[2].load(canceled).raw["observation"]["phase"] == "cancel_requested"
    assert await terminal(backend, canceled) == "canceled"


@pytest.mark.asyncio
async def test_expired_work_and_invalid_runtime_refuse(ports):
    provider, ssh, state = ports
    provider.allocation = replace(provider.allocation, deadline_epoch_s=int(time.time()) - 1)
    with pytest.raises(ValueError, match="admitted"):
        await ready(ports)
    provider.allocation = replace(
        provider.allocation,
        deadline_epoch_s=int(time.time()) + 20,
        runtime_ref="sha256:" + "0" * 64,
    )
    with pytest.raises(ValueError, match="admitted"):
        await ready(ports)
    assert ssh.actions == [] and state.rows == {}


@pytest.mark.asyncio
async def test_actual_runtime_mismatch_is_not_success(ports):
    provider, _, state = ports
    lock = {"python": "0.0.0"}
    provider.allocation = replace(
        provider.allocation, runtime_lock=lock, runtime_ref="sha256:" + _digest(lock)
    )
    backend = await ready(ports)
    with pytest.raises(RuntimeError, match="runtime differs"):
        await backend.execute("print('must not execute')", **IDENTITY)
    assert next(iter(state.rows.values())).raw["reconcile_required"]


@pytest.mark.asyncio
async def test_unwitnessed_cached_success_and_unsafe_job_id_refuse(ports):
    backend = await ready(ports)
    job = await backend.execute("print('CPU fixture')\n", **IDENTITY)
    assert await terminal(backend, job) == "completed"
    state = ports[2].load(job)
    state.raw.pop("observation")
    ports[2].save(state)
    with pytest.raises(TypeError, match="does not bind"):
        await backend.status(job)
    with pytest.raises(ValueError, match="Invalid RunPod job"):
        await backend.stop("../../foreign")


@pytest.mark.asyncio
async def test_completed_replay_survives_original_deadline(ports, monkeypatch):
    backend = await ready(ports)
    script = "print('CPU fixture')\n"
    job = await backend.execute(script, **IDENTITY)
    assert await terminal(backend, job) == "completed"
    monkeypatch.setattr(time, "time", lambda: ports[0].allocation.deadline_epoch_s + 1)
    assert await backend.execute(script, **IDENTITY) == job
    assert ports[1].actions.count("launch") == 1


@pytest.mark.asyncio
async def test_worker_deadline_has_a_real_process_timeout(ports):
    ports[0].allocation = replace(ports[0].allocation, deadline_epoch_s=int(time.time()) + 4)
    backend = await ready(ports)
    job = await backend.execute("import time\ntime.sleep(30)\n", **IDENTITY)
    assert await terminal(backend, job) == "error"
    assert ports[2].load(job).raw["observation"]["phase"] == "timed_out"


@pytest.mark.asyncio
async def test_production_ports_reuse_admitted_pod_and_existing_job_state(
    ports, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from carl_studio.compute.runpod import AtomicRunPodJobState, RunPodSSHConnection

    provider, ssh, _ = ports
    monkeypatch.setenv("CARL_ADAPTER_STATE_DIR", str(tmp_path / "adapter-state"))

    class Connection:
        async def run(self, command, *, input, check, timeout):
            assert check is True
            return SimpleNamespace(stdout=await ssh("fixture-pod", command, input, timeout))

    runner = RunPodSSHConnection("fixture-pod", Connection())
    owner = AtomicRunPodJobState()
    backend = RunPodBackend(
        provider=provider,
        ssh_runner=runner,
        job_state=owner,
        allocation=provider.allocation,
    )
    script = "print('CPU fixture using native port')\n"
    job = await backend.execute(script, **IDENTITY)
    assert await terminal(backend, job) == "completed"
    assert provider.calls == 0
    reopened = RunPodBackend(
        provider=provider, ssh_runner=runner, job_state=owner, allocation=provider.allocation
    )
    assert await reopened.execute(script, **IDENTITY) == job
    assert ssh.actions.count("launch") == 1
    with pytest.raises(ValueError, match="another pod"):
        await runner("foreign-pod", "unused", "{}", 1)
    assert "logs" not in owner.load(job).raw["observation"]
    retained = owner.load(job)
    changed = copy.deepcopy(retained)
    changed.status = "pending"
    with pytest.raises(ValueError, match="terminal observation is immutable"):
        owner.save(changed)
    changed = copy.deepcopy(retained)
    changed.raw["observation"]["start_ticks"] += 1
    with pytest.raises(ValueError, match="terminal observation is immutable"):
        owner.save(changed)
    pending = JobState(
        "runpod-" + "a" * 24, "runpod", raw={"binding": {"job_id": "runpod-" + "a" * 24}}
    )
    stale, _ = owner.claim(pending)
    pending.raw.update(cancel_requested=True, reconcile_required=True)
    owner.save(pending)
    owner.save(stale)
    assert owner.load(pending.run_id).raw["cancel_requested"] is True
    assert owner.load(pending.run_id).raw["reconcile_required"] is True
    row_path = tmp_path / "adapter-state" / "runpod" / (job + ".json")
    corrupted = json.loads(row_path.read_text())
    corrupted["backend"] = "foreign"
    row_path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="recorded job identity differs"):
        owner.claim(retained)


@pytest.mark.asyncio
async def test_admitted_runtime_lock_is_snapshotted(ports):
    backend = await ready(ports)
    ports[0].allocation.runtime_lock["python"] = "0.0.0"
    job = await backend.execute("print('Original admitted runtime')\n", **IDENTITY)
    assert await terminal(backend, job) == "completed"
    assert ports[2].load(job).raw["binding"]["runtime_lock"]["python"] == platform.python_version()


@pytest.mark.asyncio
async def test_canceled_ssh_launch_keeps_reconciliation_custody(ports):
    backend = await ready(ports)
    original = ports[1].__call__

    class CancelAfterLaunch:
        lost = False

        async def __call__(self, pod_id, command, payload, timeout):
            result = await original(pod_id, command, payload, timeout)
            if not self.lost:
                self.lost = True
                raise asyncio.CancelledError()
            return result

    backend._ssh_runner = CancelAfterLaunch()
    script = "import time\ntime.sleep(0.2)\n"
    with pytest.raises(asyncio.CancelledError):
        await backend.execute(script, **IDENTITY)
    assert next(iter(ports[2].rows.values())).raw["reconcile_required"]
    job = await backend.execute(script, **IDENTITY)
    assert await terminal(backend, job) == "completed"
    assert ports[1].actions.count("launch") == 1


@pytest.mark.asyncio
async def test_cancel_waits_for_term_ignoring_descendant(ports, tmp_path):
    backend = await ready(ports)
    marker = tmp_path / "descendant.json"
    descendant = (
        "import json,os,signal,time,pathlib\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        "stat=pathlib.Path('/proc',str(os.getpid()),'stat').read_text().rsplit(')',1)[1].split()\n"
        f"pathlib.Path({str(marker)!r}).write_text(json.dumps({{'pid':os.getpid(),'start':int(stat[19])}}))\n"
        "time.sleep(30)\n"
    )
    script = (
        "import subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable,'-c',{descendant!r}])\n"
        "time.sleep(30)\n"
    )
    job = await backend.execute(script, **IDENTITY)
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    assert marker.exists()
    await backend.stop(job)
    assert await terminal(backend, job) == "canceled"
    witness = json.loads(marker.read_text())
    path = Path("/proc", str(witness["pid"]), "stat")
    if path.exists():
        observed = path.read_text().rsplit(")", 1)[1].split()
        assert observed[0] == "Z" or int(observed[19]) != witness["start"]


@pytest.mark.asyncio
async def test_cancel_during_worker_validation_never_spawns_script(ports, tmp_path, monkeypatch):
    from carl_studio.compute import runpod

    monkeypatch.setattr(
        runpod,
        "_WORKER",
        runpod._WORKER.replace(
            "    runtime(binding)\n", "    time.sleep(1)\n    runtime(binding)\n"
        ),
    )
    backend = await ready(ports)
    marker = tmp_path / "must-not-run"
    script = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
    job = await backend.execute(script, **IDENTITY)
    await backend.stop(job)
    assert await terminal(backend, job) == "canceled"
    assert not marker.exists()


@pytest.mark.asyncio
async def test_stop_transport_timeout_respects_remaining_allocation(ports, monkeypatch):
    backend = await ready(ports)
    job = await backend.execute("import time\ntime.sleep(30)\n", **IDENTITY)
    original = ports[1].__call__
    captured = []

    class CaptureTimeout:
        async def __call__(self, pod_id, command, payload, timeout):
            captured.append(timeout)
            return await original(pod_id, command, payload, timeout)

    backend._ssh_runner = CaptureTimeout()
    with monkeypatch.context() as scope:
        scope.setattr(time, "time", lambda: ports[0].allocation.deadline_epoch_s - 0.8)
        await backend.stop(job)
    assert 0 < captured[0] <= 0.800001
    assert await terminal(backend, job) == "canceled"
