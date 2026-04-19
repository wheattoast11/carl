"""HuggingFace Jobs compute backend.

Implements the ComputeBackend protocol for submitting training jobs
to HuggingFace infrastructure via ``huggingface_hub.HfApi.run_uv_job()``.

Provenance workflow:
  1. Hash the local script (SHA-256)
  2. Upload to a dataset repo on the Hub
  3. Submit from the hosted URL (guarantees the remote runs *exactly*
     the code that was hashed)
  4. Return the job ID for monitoring

This prevents the stale-script gotcha documented in CLAUDE.md:
  "submit_training.py pulls from HF URL, not local. Upload first."
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware flavor mapping
# ---------------------------------------------------------------------------

# Canonical mapping from ComputeTarget enum values to HF Jobs flavor strings.
# Kept here (not imported from trainer.py) so the backend is self-contained.
_FLAVOR_MAP: dict[str, str] = {
    "l4x1": "l4x1",
    "l4x4": "l4x4",
    "a10g-large": "a10g-large",
    "a10g-largex2": "a10g-largex2",
    "a10g-largex4": "a10g-largex4",
    "a100-large": "a100-large",
    "a100-largex2": "a100-largex2",
    "a100-largex4": "a100-largex4",
    "a100-largex8": "a100-largex8",
    "l40sx1": "l40sx1",
    "l40sx4": "l40sx4",
    "l40sx8": "l40sx8",
    "h200": "h200",
}


class JobStage(str, Enum):
    """Normalized job stages returned by status()."""

    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELED = "canceled"
    DELETED = "deleted"
    UNKNOWN = "unknown"


@dataclass
class JobResult:
    """Structured result from status() with richer info than a bare string."""

    stage: JobStage
    message: str | None = None
    raw_stage: str | None = None


@dataclass
class ProvisionState:
    """Tracks provisioning config for the current session."""

    flavor: str = "l4x1"
    timeout: int = 10800  # 3h default
    provisioned: bool = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_UPLOAD_REPO = "carl-studio-scripts"
_DEFAULT_TIMEOUT = 10800  # 3h in seconds
_SCRIPT_BRANCH = "main"


class HFJobsBackend:
    """Submit training jobs via huggingface_hub.run_uv_job().

    Implements the ComputeBackend protocol. HF Jobs is serverless --
    no pre-provisioning is needed, but provision() validates the token
    and hardware config so failures surface early.
    """

    def __init__(
        self,
        upload_repo: str | None = None,
        namespace: str | None = None,
    ) -> None:
        """Initialize the HF Jobs backend.

        Args:
            upload_repo: Dataset repo for script upload provenance.
                         If None, auto-detected from HF user + default name.
            namespace: HF namespace for job submission (org or user).
                       If None, uses the authenticated user.
        """
        self._upload_repo = upload_repo
        self._namespace = namespace
        self._provision_state = ProvisionState()
        self._active_jobs: dict[str, str] = {}  # job_id -> script_hash

    @property
    def name(self) -> str:
        return "hf_jobs"

    # ------------------------------------------------------------------
    # Protocol: provision
    # ------------------------------------------------------------------

    async def provision(self, hardware: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
        """Validate token and hardware config. HF Jobs is serverless.

        Args:
            hardware: HF Jobs flavor string (e.g. "a100-large").
            timeout: Job timeout in seconds.

        Returns:
            Session identifier string.

        Raises:
            ValueError: If hardware flavor is unknown.
            PermissionError: If HF token is missing or invalid.
        """
        # Validate flavor
        if hardware not in _FLAVOR_MAP.values():
            available = sorted(_FLAVOR_MAP.values())
            raise ValueError(
                f"Unknown hardware flavor: {hardware!r}. "
                f"Available: {', '.join(available)}"
            )

        # Validate token
        token = self._get_token()
        if not token:
            raise PermissionError(
                "HF token required for HF Jobs backend. "
                "Run `huggingface-cli login` or set HF_TOKEN."
            )

        # Validate token is usable by calling whoami
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
            user_info = api.whoami()
            username = user_info.get("name", "unknown")
            logger.info("HF Jobs: authenticated as %s", username)
        except Exception as exc:
            raise PermissionError(
                f"HF token validation failed: {exc}. "
                "Check your token at https://huggingface.co/settings/tokens"
            ) from exc

        # Ensure upload repo exists
        self._ensure_upload_repo(api, username)

        self._provision_state = ProvisionState(
            flavor=hardware,
            timeout=timeout,
            provisioned=True,
        )
        return f"hf_jobs:{username}:{hardware}"

    # ------------------------------------------------------------------
    # Protocol: execute
    # ------------------------------------------------------------------

    async def execute(self, script: str, **kwargs: object) -> str:
        """Upload script and submit as HF Job.

        Args:
            script: Either the script content (string) or a local file path.
            **kwargs:
                flavor: Override hardware flavor from provision().
                timeout: Override timeout from provision().
                secrets: Dict of secret name -> env var name mappings.
                env: Dict of non-secret environment variables.
                labels: Dict of labels for the job.

        Returns:
            Job ID string.

        Raises:
            RuntimeError: If job submission fails.
            ValueError: If script is empty.
        """
        from huggingface_hub import HfApi

        token = self._get_token()
        api = HfApi(token=token)

        # Resolve flavor and timeout
        flavor = str(kwargs.get("flavor", self._provision_state.flavor))
        timeout_raw = kwargs.get("timeout", self._provision_state.timeout)
        timeout = self._parse_timeout_value(timeout_raw)

        # Resolve script content
        script_content = self._resolve_script_content(script)
        if not script_content.strip():
            raise ValueError("Script content is empty")

        # Compute provenance hash
        script_hash = hashlib.sha256(script_content.encode("utf-8")).hexdigest()[:16]
        script_filename = f"train_{script_hash}.py"

        logger.info(
            "HF Jobs: uploading script %s (hash=%s, %d bytes)",
            script_filename,
            script_hash,
            len(script_content),
        )

        # Upload script to dataset repo for provenance
        upload_repo = self._resolve_upload_repo(api)
        try:
            api.upload_file(
                path_or_fileobj=script_content.encode("utf-8"),
                path_in_repo=script_filename,
                repo_id=upload_repo,
                repo_type="dataset",
                commit_message=f"carl-studio: upload training script {script_hash}",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to upload script to {upload_repo}: {exc}"
            ) from exc

        # Build the hosted URL for the script
        script_url = (
            f"https://huggingface.co/datasets/{upload_repo}/"
            f"resolve/{_SCRIPT_BRANCH}/{script_filename}"
        )
        logger.info("HF Jobs: script URL = %s", script_url)

        # Build secrets dict
        secrets = self._build_secrets(kwargs.get("secrets"))

        # Build env dict
        env: dict[str, str] = {"PYTHONUNBUFFERED": "1"}
        extra_env = kwargs.get("env")
        if isinstance(extra_env, dict):
            env.update({str(k): str(v) for k, v in extra_env.items()})

        # Build labels
        labels: dict[str, str] = {
            "carl_studio": "true",
            "script_hash": script_hash,
        }
        extra_labels = kwargs.get("labels")
        if isinstance(extra_labels, dict):
            labels.update({str(k): str(v) for k, v in extra_labels.items()})

        # Submit the job
        try:
            job = api.run_uv_job(
                script=script_url,
                flavor=flavor,
                timeout=timeout,
                env=env,
                secrets=secrets,
                labels=labels,
                namespace=self._namespace,
                token=token,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to submit HF Job (flavor={flavor}): {exc}"
            ) from exc

        job_id = job.id
        self._active_jobs[job_id] = script_hash
        logger.info(
            "HF Jobs: submitted job %s (flavor=%s, timeout=%ds, hash=%s)",
            job_id,
            flavor,
            timeout,
            script_hash,
        )
        return job_id

    # ------------------------------------------------------------------
    # Protocol: status
    # ------------------------------------------------------------------

    async def status(self, job_id: str) -> str:
        """Get job status as a normalized stage string.

        Args:
            job_id: HF Job ID.

        Returns:
            Normalized status string: "running", "completed", "error",
            "canceled", "deleted", or "unknown".

        Raises:
            RuntimeError: If the status check fails.
        """
        result = await self.status_detailed(job_id)
        return result.stage.value

    async def status_detailed(self, job_id: str) -> JobResult:
        """Get detailed job status with message.

        Args:
            job_id: HF Job ID.

        Returns:
            JobResult with stage, message, and raw_stage.
        """
        if not job_id or not job_id.strip():
            raise ValueError("job_id must be a non-empty string")

        from huggingface_hub import HfApi

        token = self._get_token()
        api = HfApi(token=token)

        try:
            job_info = api.inspect_job(job_id=job_id, token=token)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to inspect job {job_id}: {exc}"
            ) from exc

        raw_stage = str(job_info.status.stage.value) if job_info.status else "UNKNOWN"
        message = job_info.status.message if job_info.status else None

        stage = self._normalize_stage(raw_stage)

        return JobResult(
            stage=stage,
            message=message,
            raw_stage=raw_stage,
        )

    # ------------------------------------------------------------------
    # Protocol: logs
    # ------------------------------------------------------------------

    async def logs(self, job_id: str, tail: int = 50) -> list[str]:
        """Get recent log lines from a job.

        Args:
            job_id: HF Job ID.
            tail: Number of most recent lines to return.

        Returns:
            List of log line strings (max 300 chars each to prevent
            memory issues with binary output).

        Raises:
            RuntimeError: If log retrieval fails.
        """
        if not job_id or not job_id.strip():
            raise ValueError("job_id must be a non-empty string")
        if tail < 1:
            raise ValueError("tail must be >= 1")

        from huggingface_hub import HfApi

        token = self._get_token()
        api = HfApi(token=token)

        try:
            raw_logs = list(api.fetch_job_logs(job_id=job_id, token=token))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch logs for job {job_id}: {exc}"
            ) from exc

        # Take the last `tail` entries, truncate each to 300 chars
        recent = raw_logs[-tail:] if len(raw_logs) > tail else raw_logs
        return [str(entry)[:300] for entry in recent]

    # ------------------------------------------------------------------
    # Protocol: stop
    # ------------------------------------------------------------------

    async def stop(self, job_id: str) -> None:
        """Cancel a running job.

        Args:
            job_id: HF Job ID.

        Raises:
            RuntimeError: If cancellation fails.
        """
        if not job_id or not job_id.strip():
            raise ValueError("job_id must be a non-empty string")

        from huggingface_hub import HfApi

        token = self._get_token()
        api = HfApi(token=token)

        try:
            api.cancel_job(job_id=job_id, token=token)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to cancel job {job_id}: {exc}"
            ) from exc

        logger.info("HF Jobs: canceled job %s", job_id)
        self._active_jobs.pop(job_id, None)

    # ------------------------------------------------------------------
    # Protocol: teardown
    # ------------------------------------------------------------------

    async def teardown(self) -> None:
        """Release resources. HF Jobs is serverless -- no persistent resources.

        Clears the active jobs tracking dict.
        """
        active_count = len(self._active_jobs)
        self._active_jobs.clear()
        self._provision_state = ProvisionState()
        if active_count > 0:
            logger.info(
                "HF Jobs: teardown cleared %d tracked job(s)", active_count
            )

    # ------------------------------------------------------------------
    # Extended API (beyond protocol)
    # ------------------------------------------------------------------

    async def list_jobs(self) -> list[dict[str, Any]]:
        """List all jobs for the current user/namespace.

        Returns:
            List of dicts with id, stage, flavor, created_at fields.
        """
        from huggingface_hub import HfApi

        token = self._get_token()
        api = HfApi(token=token)

        try:
            jobs = api.list_jobs(namespace=self._namespace, token=token)
        except Exception as exc:
            raise RuntimeError(f"Failed to list jobs: {exc}") from exc

        results: list[dict[str, Any]] = []
        for job in jobs:
            stage = self._normalize_stage(
                str(job.status.stage.value) if job.status else "UNKNOWN"
            )
            results.append(
                {
                    "id": job.id,
                    "stage": stage.value,
                    "flavor": str(job.flavor) if job.flavor else None,
                    "created_at": str(job.created_at) if job.created_at else None,
                    "url": getattr(job, "url", None),
                    "labels": getattr(job, "labels", None),
                    "carl_studio": bool(
                        getattr(job, "labels", None)
                        and job.labels.get("carl_studio") == "true"
                    ),
                }
            )
        return results

    def get_script_hash(self, job_id: str) -> str | None:
        """Get the script hash for a tracked job (provenance)."""
        return self._active_jobs.get(job_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_token() -> str | None:
        """Get HF token. Prefers huggingface_hub credentials over env var.

        Per CLAUDE.md: "Token is in HF credentials, not exported.
        Use `from huggingface_hub import get_token; token = get_token()`
        instead of `os.environ.get('HF_TOKEN')`"
        """
        try:
            from huggingface_hub import get_token

            token = get_token()
            if token:
                return token
        except Exception:
            pass

        # Fallback to env var
        return os.environ.get("HF_TOKEN")

    def _resolve_upload_repo(self, api: Any) -> str:
        """Resolve the dataset repo for script uploads.

        If not explicitly set, uses {username}/{_DEFAULT_UPLOAD_REPO}.
        """
        if self._upload_repo:
            return self._upload_repo

        try:
            user_info = api.whoami()
            username = user_info.get("name", "unknown")
        except Exception:
            username = "unknown"

        return f"{username}/{_DEFAULT_UPLOAD_REPO}"

    def _ensure_upload_repo(self, api: Any, username: str) -> None:
        """Ensure the upload dataset repo exists, create if not."""
        repo_id = self._upload_repo or f"{username}/{_DEFAULT_UPLOAD_REPO}"

        try:
            api.repo_info(repo_id=repo_id, repo_type="dataset")
            logger.debug("Upload repo exists: %s", repo_id)
        except Exception:
            try:
                api.create_repo(
                    repo_id=repo_id,
                    repo_type="dataset",
                    private=True,
                    exist_ok=True,
                )
                logger.info("Created upload repo: %s", repo_id)
            except Exception as exc:
                logger.warning(
                    "Could not create upload repo %s: %s. "
                    "Script upload may fail.",
                    repo_id,
                    exc,
                )

        # Cache the resolved repo
        if not self._upload_repo:
            self._upload_repo = repo_id

    @staticmethod
    def _resolve_script_content(script: str) -> str:
        """Resolve script to content string.

        If `script` looks like a file path and the file exists, read it.
        Otherwise treat it as inline script content.
        """
        # Check if it's a file path
        if len(script) < 500 and not script.startswith("#"):
            expanded = os.path.expanduser(script)
            if os.path.isfile(expanded):
                with open(expanded, "r", encoding="utf-8") as f:
                    return f.read()

        return script

    @staticmethod
    def _build_secrets(
        secrets_input: object,
    ) -> dict[str, str]:
        """Build the secrets dict for run_uv_job.

        HF Jobs secrets are injected as env vars. The value in the dict
        is the name of the HF secret (not the secret value itself).

        Always includes HF_TOKEN.
        """
        result: dict[str, str] = {"HF_TOKEN": "HF_TOKEN"}

        if isinstance(secrets_input, dict):
            for key, value in secrets_input.items():
                result[str(key)] = str(value)

        return result

    @staticmethod
    def _normalize_stage(raw_stage: str) -> JobStage:
        """Normalize a raw HF Jobs stage string to our enum."""
        stage_upper = raw_stage.strip().upper()
        mapping: dict[str, JobStage] = {
            "RUNNING": JobStage.RUNNING,
            "COMPLETED": JobStage.COMPLETED,
            "ERROR": JobStage.ERROR,
            "CANCELED": JobStage.CANCELED,
            "DELETED": JobStage.DELETED,
        }
        return mapping.get(stage_upper, JobStage.UNKNOWN)

    @staticmethod
    def _parse_timeout_value(raw: object) -> int:
        """Parse a timeout value to seconds.

        Accepts:
          - int/float: seconds directly
          - str: "3h", "90m", "2h30m", "14400", "2d"
        """
        if isinstance(raw, (int, float)):
            return max(1, int(raw))

        if not isinstance(raw, str):
            return _DEFAULT_TIMEOUT

        text = raw.strip()
        if not text:
            return _DEFAULT_TIMEOUT

        total = 0

        # Days
        import re

        d_match = re.search(r"(\d+(?:\.\d+)?)\s*d", text, re.IGNORECASE)
        if d_match:
            total += int(float(d_match.group(1)) * 86400)
            text = text[: d_match.start()] + text[d_match.end() :]

        # Hours
        h_match = re.search(r"(\d+(?:\.\d+)?)\s*h", text, re.IGNORECASE)
        if h_match:
            total += int(float(h_match.group(1)) * 3600)
            text = text[: h_match.start()] + text[h_match.end() :]

        # Minutes
        m_match = re.search(r"(\d+(?:\.\d+)?)\s*m(?!s)", text, re.IGNORECASE)
        if m_match:
            total += int(float(m_match.group(1)) * 60)
            text = text[: m_match.start()] + text[m_match.end() :]

        # Seconds (bare number or with 's')
        s_match = re.search(r"(\d+)\s*s?", text.strip())
        if s_match and s_match.group(1):
            val = int(s_match.group(1))
            if total == 0:
                total = val
            else:
                total += val

        return max(1, total) if total > 0 else _DEFAULT_TIMEOUT
