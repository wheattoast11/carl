"""SSH compute backend — remote execution via SSH."""
from __future__ import annotations

import os


class SSHBackend:
    """Execute training scripts on remote machines via SSH."""

    def __init__(self, host: str = "", user: str = "root", key_path: str | None = None) -> None:
        self._host = host or os.getenv("CARL_SSH_HOST", "")
        self._user = user
        self._key_path = key_path or os.getenv("CARL_SSH_KEY", None)
        self._conn = None

    @property
    def name(self) -> str:
        return "ssh"

    async def provision(self, hardware: str, timeout: int) -> str:
        if not self._host:
            raise ValueError("SSH host required. Set CARL_SSH_HOST or pass host= to SSHBackend()")
        # Test connection
        try:
            import asyncssh
        except ImportError:
            raise ImportError("asyncssh required: pip install asyncssh")

        self._conn = await asyncssh.connect(
            self._host,
            username=self._user,
            client_keys=[self._key_path] if self._key_path else None,
            known_hosts=None,
        )
        return f"ssh-{self._host}"

    async def execute(self, script: str, **kwargs: object) -> str:
        if not self._conn:
            raise RuntimeError("Must provision() before execute()")

        # Upload script
        remote_path = "/tmp/carl_train.py"
        async with self._conn.start_sftp_client() as sftp:
            async with sftp.open(remote_path, "w") as f:
                await f.write(script)

        # Execute in background
        result = await self._conn.run(
            f"nohup python {remote_path} > /tmp/carl_train.log 2>&1 & echo $!",
            check=True,
        )
        pid = result.stdout.strip()
        return f"ssh-pid-{pid}"

    async def status(self, job_id: str) -> str:
        if not self._conn:
            return "unknown"
        pid = job_id.replace("ssh-pid-", "")
        result = await self._conn.run(f"kill -0 {pid} 2>/dev/null && echo running || echo done")
        return "running" if "running" in result.stdout else "completed"

    async def logs(self, job_id: str, tail: int = 50) -> list[str]:
        if not self._conn:
            return []
        result = await self._conn.run(f"tail -n {tail} /tmp/carl_train.log")
        return result.stdout.strip().split("\n")

    async def stop(self, job_id: str) -> None:
        if self._conn:
            pid = job_id.replace("ssh-pid-", "")
            await self._conn.run(f"kill {pid} 2>/dev/null || true")

    async def teardown(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
