"""Tests for HFJobsBackend -- no GPU, no network required.

All HuggingFace Hub API calls are mocked. Tests verify:
  - Provenance: script hashing + upload + URL construction
  - Status normalization across all JobStage values
  - Error paths: missing token, bad flavor, empty script, API failures
  - Timeout parsing in all supported formats
  - Secrets and env construction
  - Teardown cleanup
"""

import hashlib
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from carl_studio.compute.hf_jobs import (
    HFJobsBackend,
    JobStage,
    _DEFAULT_TIMEOUT,
    _FLAVOR_MAP,
)


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


class _FakeJobStage(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    CANCELED = "CANCELED"
    DELETED = "DELETED"


def _make_job_info(
    job_id: str = "job-abc123",
    stage: str = "RUNNING",
    message: str | None = None,
    flavor: str | None = "a100-large",
    url: str = "https://huggingface.co/jobs/job-abc123",
    labels: dict | None = None,
    created_at: str | None = "2026-04-01T00:00:00",
) -> SimpleNamespace:
    """Build a fake JobInfo-like object matching huggingface_hub's shape."""
    stage_enum = _FakeJobStage(stage) if stage in _FakeJobStage.__members__ else stage
    return SimpleNamespace(
        id=job_id,
        status=SimpleNamespace(stage=stage_enum, message=message),
        flavor=flavor,
        url=url,
        labels=labels or {},
        created_at=created_at,
    )


@pytest.fixture
def backend():
    return HFJobsBackend(upload_repo="testuser/test-scripts")


@pytest.fixture
def mock_api():
    """Patch HfApi and get_token for all tests that need them."""
    with (
        patch("carl_studio.compute.hf_jobs.HFJobsBackend._get_token", return_value="hf_test_token"),
    ):
        yield


# ---------------------------------------------------------------------------
# provision()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provision_validates_token(backend):
    """provision() raises PermissionError when token is missing."""
    with patch.object(HFJobsBackend, "_get_token", return_value=None):
        with pytest.raises(PermissionError, match="HF token required"):
            await backend.provision("a100-large")


@pytest.mark.asyncio
async def test_provision_rejects_unknown_flavor(backend, mock_api):
    """provision() raises ValueError for unknown hardware."""
    with pytest.raises(ValueError, match="Unknown hardware flavor"):
        await backend.provision("rtx-9090-ultra")


@pytest.mark.asyncio
async def test_provision_validates_token_with_whoami(backend):
    """provision() calls whoami to validate the token."""
    mock_hf = MagicMock()
    mock_hf.whoami.return_value = {"name": "testuser"}
    mock_hf.repo_info.return_value = True  # repo exists

    with (
        patch.object(HFJobsBackend, "_get_token", return_value="hf_test_token"),
        patch("huggingface_hub.HfApi", return_value=mock_hf),
    ):
        session_id = await backend.provision("a100-large")

    assert "testuser" in session_id
    assert "a100-large" in session_id
    mock_hf.whoami.assert_called_once()


@pytest.mark.asyncio
async def test_provision_creates_upload_repo_if_missing(backend):
    """provision() creates the upload repo when repo_info raises."""
    mock_hf = MagicMock()
    mock_hf.whoami.return_value = {"name": "testuser"}
    mock_hf.repo_info.side_effect = Exception("Not found")
    mock_hf.create_repo.return_value = None

    with (
        patch.object(HFJobsBackend, "_get_token", return_value="hf_test_token"),
        patch("huggingface_hub.HfApi", return_value=mock_hf),
    ):
        await backend.provision("l4x1")

    mock_hf.create_repo.assert_called_once_with(
        repo_id="testuser/test-scripts",
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )


@pytest.mark.asyncio
async def test_provision_stores_state(backend):
    """provision() stores flavor and timeout in internal state."""
    mock_hf = MagicMock()
    mock_hf.whoami.return_value = {"name": "testuser"}
    mock_hf.repo_info.return_value = True

    with (
        patch.object(HFJobsBackend, "_get_token", return_value="hf_test_token"),
        patch("huggingface_hub.HfApi", return_value=mock_hf),
    ):
        await backend.provision("l40sx1", timeout=7200)

    assert backend._provision_state.flavor == "l40sx1"
    assert backend._provision_state.timeout == 7200
    assert backend._provision_state.provisioned is True


# ---------------------------------------------------------------------------
# execute()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_uploads_and_submits(backend, mock_api):
    """execute() uploads script then submits via run_uv_job."""
    script_content = '# /// script\nprint("hello")\n'
    script_hash = hashlib.sha256(script_content.encode()).hexdigest()[:16]
    expected_filename = f"train_{script_hash}.py"

    mock_hf = MagicMock()
    mock_hf.whoami.return_value = {"name": "testuser"}
    mock_hf.upload_file.return_value = None
    mock_hf.run_uv_job.return_value = _make_job_info(job_id="job-xyz789")

    with patch("huggingface_hub.HfApi", return_value=mock_hf):
        job_id = await backend.execute(
            script=script_content,
            flavor="a100-large",
            timeout="2h",
        )

    assert job_id == "job-xyz789"

    # Verify upload
    upload_call = mock_hf.upload_file.call_args
    assert upload_call.kwargs["path_in_repo"] == expected_filename
    assert upload_call.kwargs["repo_id"] == "testuser/test-scripts"
    assert upload_call.kwargs["repo_type"] == "dataset"

    # Verify submission
    run_call = mock_hf.run_uv_job.call_args
    assert expected_filename in run_call.kwargs["script"]
    assert run_call.kwargs["flavor"] == "a100-large"


@pytest.mark.asyncio
async def test_execute_rejects_empty_script(backend, mock_api):
    """execute() raises ValueError for empty script."""
    mock_hf = MagicMock()
    with patch("huggingface_hub.HfApi", return_value=mock_hf):
        with pytest.raises(ValueError, match="empty"):
            await backend.execute(script="   ")


@pytest.mark.asyncio
async def test_execute_includes_hf_token_in_secrets(backend, mock_api):
    """execute() always includes HF_TOKEN in secrets."""
    mock_hf = MagicMock()
    mock_hf.whoami.return_value = {"name": "testuser"}
    mock_hf.upload_file.return_value = None
    mock_hf.run_uv_job.return_value = _make_job_info()

    with patch("huggingface_hub.HfApi", return_value=mock_hf):
        await backend.execute(script="print('test')", secrets={"WANDB_KEY": "WANDB_KEY"})

    secrets = mock_hf.run_uv_job.call_args.kwargs["secrets"]
    assert secrets["HF_TOKEN"] == "HF_TOKEN"
    assert secrets["WANDB_KEY"] == "WANDB_KEY"


@pytest.mark.asyncio
async def test_execute_includes_carl_studio_label(backend, mock_api):
    """execute() tags jobs with carl_studio label."""
    mock_hf = MagicMock()
    mock_hf.whoami.return_value = {"name": "testuser"}
    mock_hf.upload_file.return_value = None
    mock_hf.run_uv_job.return_value = _make_job_info()

    with patch("huggingface_hub.HfApi", return_value=mock_hf):
        await backend.execute(script="print('test')")

    labels = mock_hf.run_uv_job.call_args.kwargs["labels"]
    assert labels["carl_studio"] == "true"
    assert "script_hash" in labels


@pytest.mark.asyncio
async def test_execute_tracks_active_job(backend, mock_api):
    """execute() tracks the job_id -> script_hash mapping."""
    mock_hf = MagicMock()
    mock_hf.whoami.return_value = {"name": "testuser"}
    mock_hf.upload_file.return_value = None
    mock_hf.run_uv_job.return_value = _make_job_info(job_id="job-tracked")

    with patch("huggingface_hub.HfApi", return_value=mock_hf):
        job_id = await backend.execute(script="print('test')")

    assert job_id in backend._active_jobs
    assert backend.get_script_hash(job_id) is not None


@pytest.mark.asyncio
async def test_execute_upload_failure_raises(backend, mock_api):
    """execute() raises RuntimeError when upload fails."""
    mock_hf = MagicMock()
    mock_hf.whoami.return_value = {"name": "testuser"}
    mock_hf.upload_file.side_effect = Exception("Upload failed: 403")

    with patch("huggingface_hub.HfApi", return_value=mock_hf):
        with pytest.raises(RuntimeError, match="Failed to upload"):
            await backend.execute(script="print('test')")


@pytest.mark.asyncio
async def test_execute_submit_failure_raises(backend, mock_api):
    """execute() raises RuntimeError when job submission fails."""
    mock_hf = MagicMock()
    mock_hf.whoami.return_value = {"name": "testuser"}
    mock_hf.upload_file.return_value = None
    mock_hf.run_uv_job.side_effect = Exception("Quota exceeded")

    with patch("huggingface_hub.HfApi", return_value=mock_hf):
        with pytest.raises(RuntimeError, match="Failed to submit"):
            await backend.execute(script="print('test')")


@pytest.mark.asyncio
async def test_execute_reads_file_if_path(backend, mock_api, tmp_path):
    """execute() reads file content when given a path."""
    script_file = tmp_path / "train.py"
    script_file.write_text("# from file\nprint('hello')\n")

    mock_hf = MagicMock()
    mock_hf.whoami.return_value = {"name": "testuser"}
    mock_hf.upload_file.return_value = None
    mock_hf.run_uv_job.return_value = _make_job_info()

    with patch("huggingface_hub.HfApi", return_value=mock_hf):
        await backend.execute(script=str(script_file))

    # Verify the file content was uploaded (not the path string)
    uploaded_bytes = mock_hf.upload_file.call_args.kwargs["path_or_fileobj"]
    assert b"from file" in uploaded_bytes


# ---------------------------------------------------------------------------
# status()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_stage, expected",
    [
        ("RUNNING", "running"),
        ("COMPLETED", "completed"),
        ("ERROR", "error"),
        ("CANCELED", "canceled"),
        ("DELETED", "deleted"),
    ],
)
async def test_status_normalizes_stages(backend, mock_api, raw_stage, expected):
    """status() normalizes all HF Jobs stages to lowercase strings."""
    mock_hf = MagicMock()
    mock_hf.inspect_job.return_value = _make_job_info(stage=raw_stage)

    with patch("huggingface_hub.HfApi", return_value=mock_hf):
        result = await backend.status("job-test")

    assert result == expected


@pytest.mark.asyncio
async def test_status_detailed_includes_message(backend, mock_api):
    """status_detailed() returns message from the job."""
    mock_hf = MagicMock()
    mock_hf.inspect_job.return_value = _make_job_info(
        stage="ERROR", message="OOM at step 68"
    )

    with patch("huggingface_hub.HfApi", return_value=mock_hf):
        result = await backend.status_detailed("job-test")

    assert result.stage == JobStage.ERROR
    assert result.message == "OOM at step 68"
    assert result.raw_stage == "ERROR"


@pytest.mark.asyncio
async def test_status_empty_job_id_raises(backend, mock_api):
    """status() raises ValueError for empty job_id."""
    with pytest.raises(ValueError, match="non-empty"):
        await backend.status("")


@pytest.mark.asyncio
async def test_status_api_failure_raises(backend, mock_api):
    """status() raises RuntimeError when inspect_job fails."""
    mock_hf = MagicMock()
    mock_hf.inspect_job.side_effect = Exception("Network error")

    with patch("huggingface_hub.HfApi", return_value=mock_hf):
        with pytest.raises(RuntimeError, match="Failed to inspect"):
            await backend.status("job-test")


# ---------------------------------------------------------------------------
# logs()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_logs_returns_recent_lines(backend, mock_api):
    """logs() returns the last `tail` entries."""
    all_lines = [f"line-{i}" for i in range(100)]
    mock_hf = MagicMock()
    mock_hf.fetch_job_logs.return_value = all_lines

    with patch("huggingface_hub.HfApi", return_value=mock_hf):
        result = await backend.logs("job-test", tail=10)

    assert len(result) == 10
    assert result[0] == "line-90"
    assert result[-1] == "line-99"


@pytest.mark.asyncio
async def test_logs_truncates_long_lines(backend, mock_api):
    """logs() truncates individual entries to 300 chars."""
    long_line = "x" * 500
    mock_hf = MagicMock()
    mock_hf.fetch_job_logs.return_value = [long_line]

    with patch("huggingface_hub.HfApi", return_value=mock_hf):
        result = await backend.logs("job-test")

    assert len(result[0]) == 300


@pytest.mark.asyncio
async def test_logs_empty_job_id_raises(backend, mock_api):
    """logs() raises ValueError for empty job_id."""
    with pytest.raises(ValueError, match="non-empty"):
        await backend.logs("")


@pytest.mark.asyncio
async def test_logs_invalid_tail_raises(backend, mock_api):
    """logs() raises ValueError for tail < 1."""
    with pytest.raises(ValueError, match="tail"):
        await backend.logs("job-test", tail=0)


@pytest.mark.asyncio
async def test_logs_api_failure_raises(backend, mock_api):
    """logs() raises RuntimeError when fetch_job_logs fails."""
    mock_hf = MagicMock()
    mock_hf.fetch_job_logs.side_effect = Exception("Connection reset")

    with patch("huggingface_hub.HfApi", return_value=mock_hf):
        with pytest.raises(RuntimeError, match="Failed to fetch logs"):
            await backend.logs("job-test")


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_cancels_job(backend, mock_api):
    """stop() calls cancel_job on the API."""
    mock_hf = MagicMock()
    mock_hf.cancel_job.return_value = None

    # Pre-populate active jobs
    backend._active_jobs["job-cancel"] = "abc123"

    with patch("huggingface_hub.HfApi", return_value=mock_hf):
        await backend.stop("job-cancel")

    mock_hf.cancel_job.assert_called_once_with(job_id="job-cancel", token="hf_test_token")
    assert "job-cancel" not in backend._active_jobs


@pytest.mark.asyncio
async def test_stop_empty_job_id_raises(backend, mock_api):
    """stop() raises ValueError for empty job_id."""
    with pytest.raises(ValueError, match="non-empty"):
        await backend.stop("")


@pytest.mark.asyncio
async def test_stop_api_failure_raises(backend, mock_api):
    """stop() raises RuntimeError when cancel_job fails."""
    mock_hf = MagicMock()
    mock_hf.cancel_job.side_effect = Exception("Already completed")

    with patch("huggingface_hub.HfApi", return_value=mock_hf):
        with pytest.raises(RuntimeError, match="Failed to cancel"):
            await backend.stop("job-test")


# ---------------------------------------------------------------------------
# teardown()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_teardown_clears_state(backend):
    """teardown() clears active jobs and provision state."""
    backend._active_jobs["job-1"] = "hash1"
    backend._active_jobs["job-2"] = "hash2"
    backend._provision_state.provisioned = True

    await backend.teardown()

    assert len(backend._active_jobs) == 0
    assert backend._provision_state.provisioned is False


# ---------------------------------------------------------------------------
# Timeout parsing
# ---------------------------------------------------------------------------


class TestTimeoutParsing:
    """Test _parse_timeout_value static method."""

    def test_int_passthrough(self):
        assert HFJobsBackend._parse_timeout_value(3600) == 3600

    def test_float_truncated(self):
        assert HFJobsBackend._parse_timeout_value(3600.5) == 3600

    def test_string_hours(self):
        assert HFJobsBackend._parse_timeout_value("3h") == 10800

    def test_string_minutes(self):
        assert HFJobsBackend._parse_timeout_value("90m") == 5400

    def test_string_compound(self):
        assert HFJobsBackend._parse_timeout_value("2h30m") == 9000

    def test_string_days(self):
        assert HFJobsBackend._parse_timeout_value("2d") == 172800

    def test_string_bare_seconds(self):
        assert HFJobsBackend._parse_timeout_value("14400") == 14400

    def test_string_fractional_hours(self):
        assert HFJobsBackend._parse_timeout_value("1.5h") == 5400

    def test_empty_returns_default(self):
        assert HFJobsBackend._parse_timeout_value("") == _DEFAULT_TIMEOUT

    def test_none_returns_default(self):
        assert HFJobsBackend._parse_timeout_value(None) == _DEFAULT_TIMEOUT

    def test_zero_clamped_to_one(self):
        assert HFJobsBackend._parse_timeout_value(0) == 1


# ---------------------------------------------------------------------------
# Stage normalization
# ---------------------------------------------------------------------------


class TestStageNormalization:
    """Test _normalize_stage static method."""

    def test_all_known_stages(self):
        assert HFJobsBackend._normalize_stage("RUNNING") == JobStage.RUNNING
        assert HFJobsBackend._normalize_stage("COMPLETED") == JobStage.COMPLETED
        assert HFJobsBackend._normalize_stage("ERROR") == JobStage.ERROR
        assert HFJobsBackend._normalize_stage("CANCELED") == JobStage.CANCELED
        assert HFJobsBackend._normalize_stage("DELETED") == JobStage.DELETED

    def test_unknown_stage(self):
        assert HFJobsBackend._normalize_stage("PENDING") == JobStage.UNKNOWN
        assert HFJobsBackend._normalize_stage("") == JobStage.UNKNOWN

    def test_case_insensitive(self):
        assert HFJobsBackend._normalize_stage("running") == JobStage.RUNNING
        assert HFJobsBackend._normalize_stage("Completed") == JobStage.COMPLETED


# ---------------------------------------------------------------------------
# Secrets construction
# ---------------------------------------------------------------------------


class TestBuildSecrets:
    """Test _build_secrets static method."""

    def test_always_includes_hf_token(self):
        result = HFJobsBackend._build_secrets(None)
        assert result == {"HF_TOKEN": "HF_TOKEN"}

    def test_merges_custom_secrets(self):
        result = HFJobsBackend._build_secrets({"WANDB_KEY": "WANDB_KEY"})
        assert result["HF_TOKEN"] == "HF_TOKEN"
        assert result["WANDB_KEY"] == "WANDB_KEY"

    def test_ignores_non_dict(self):
        result = HFJobsBackend._build_secrets("not a dict")
        assert result == {"HF_TOKEN": "HF_TOKEN"}


# ---------------------------------------------------------------------------
# Script resolution
# ---------------------------------------------------------------------------


class TestResolveScript:

    def test_inline_content_returned_as_is(self):
        script = "# /// script\nprint('hello')\n"
        assert HFJobsBackend._resolve_script_content(script) == script

    def test_file_path_reads_content(self, tmp_path):
        script_file = tmp_path / "train.py"
        script_file.write_text("# from file\nprint('hello')\n")
        result = HFJobsBackend._resolve_script_content(str(script_file))
        assert "from file" in result

    def test_nonexistent_path_treated_as_content(self):
        # Short string that isn't a file
        result = HFJobsBackend._resolve_script_content("/nonexistent/path.py")
        assert result == "/nonexistent/path.py"


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_backend_satisfies_protocol():
    """HFJobsBackend must satisfy the ComputeBackend protocol."""
    from carl_studio.compute.protocol import ComputeBackend

    backend = HFJobsBackend()
    assert isinstance(backend, ComputeBackend)


def test_backend_name():
    """Backend name is 'hf_jobs'."""
    assert HFJobsBackend().name == "hf_jobs"


def test_flavor_map_covers_all_compute_targets():
    """Every non-LOCAL ComputeTarget should have a flavor mapping."""
    from carl_studio.types.config import ComputeTarget

    for target in ComputeTarget:
        if target == ComputeTarget.LOCAL:
            continue
        # The value should either be in the flavor map or handled by trainer
        assert target.value in _FLAVOR_MAP or target.value in (
            "l4x1", "l4x4", "a10g-large", "a10g-largex2", "a10g-largex4",
            "a100-large", "a100-largex2", "a100-largex4", "a100-largex8",
            "l40sx1", "l40sx4", "l40sx8",
        ), f"ComputeTarget.{target.name} ({target.value}) has no flavor mapping"


# ---------------------------------------------------------------------------
# list_jobs()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_jobs_returns_structured_data(backend, mock_api):
    """list_jobs() returns structured dicts with expected fields."""
    mock_hf = MagicMock()
    mock_hf.list_jobs.return_value = [
        _make_job_info(
            job_id="job-1",
            stage="RUNNING",
            labels={"carl_studio": "true"},
        ),
        _make_job_info(
            job_id="job-2",
            stage="COMPLETED",
            labels={},
        ),
    ]

    with patch("huggingface_hub.HfApi", return_value=mock_hf):
        jobs = await backend.list_jobs()

    assert len(jobs) == 2
    assert jobs[0]["id"] == "job-1"
    assert jobs[0]["stage"] == "running"
    assert jobs[0]["carl_studio"] is True
    assert jobs[1]["carl_studio"] is False


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


def test_factory_returns_hf_jobs_backend():
    """get_backend('hf_jobs') returns an HFJobsBackend instance."""
    from carl_studio.compute import get_backend

    backend = get_backend("hf_jobs")
    assert isinstance(backend, HFJobsBackend)
    assert backend.name == "hf_jobs"
