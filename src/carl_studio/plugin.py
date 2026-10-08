"""Source-bound native plugin projections and owned local installation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from carl_core.errors import CARLError

HOSTS = ("codex", "claude", "opencode")


def source_root() -> Path:
    """Find the editable checkout's canonical portable plugin."""
    explicit = os.environ.get("CARL_PLUGIN_SOURCE")
    candidates = [Path(explicit)] if explicit else [Path.cwd(), *Path(__file__).parents]
    for candidate in candidates:
        if (candidate / "plugin.json").is_file() and (candidate / "skills/carl/SKILL.md").is_file():
            return candidate.resolve()
    raise CARLError("Run from a CARL checkout or set CARL_PLUGIN_SOURCE", code="carl.plugin.source")


def source_digest(root: Path) -> str:
    """Bind source and runtime bytes while excluding credentials and artifacts."""
    paths = [root / name for name in ("pyproject.toml", "uv.lock", "plugin.json", "mcp.json")]
    for name in ("skills", "src/carl_studio", "packages/carl-core/src"):
        paths.extend(
            path
            for path in (root / name).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise CARLError("Plugin source escapes checkout", code="carl.plugin.path")
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, (json.dumps(value, indent=2) + "\n").encode())


def _atomic_write(path: Path, content: bytes) -> None:
    fd, name = tempfile.mkstemp(prefix=".carl-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def project(root: Path, output: Path, *, python: str = sys.executable) -> dict[str, Any]:
    """Generate a small native marketplace from the portable source."""
    for path in [output, *output.rglob("*")]:
        if path.is_symlink():
            raise CARLError("Projection target is a symlink", code="carl.plugin.ownership")
    digest = source_digest(root)
    manifest = json.loads((root / "plugin.json").read_text())
    snapshot = output / "plugins/carl"
    snapshot.mkdir(parents=True, exist_ok=True)
    portable = json.loads((root / "mcp.json").read_text())
    native_mcp = {
        "mcpServers": {
            "carl": {
                "command": python,
                "args": ["-m", "carl_studio.mcp"],
                "env": {"CARL_PLUGIN_SOURCE": str(root)},
            }
        }
    }
    _write_json(snapshot / "plugin.json", manifest)
    _write_json(snapshot / "mcp.json", portable)
    _write_json(snapshot / ".mcp.json", native_mcp)
    for path in (root / "skills/carl").rglob("*"):
        if path.is_file():
            destination = snapshot / "skills/carl" / path.relative_to(root / "skills/carl")
            destination.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(destination, path.read_bytes())
    native = {key: manifest[key] for key in ("name", "description", "author")}
    native["version"] = manifest["version"] + "+source." + digest[:12]
    _write_json(snapshot / ".claude-plugin/plugin.json", native)
    codex = {
        **native,
        "skills": "./skills/",
        "mcpServers": "./.mcp.json",
        "interface": manifest["extensions"]["com.openai"]["interface"],
    }
    _write_json(snapshot / ".codex-plugin/plugin.json", codex)
    marketplace = {
        "name": "carl-local",
        "owner": manifest["author"],
        "plugins": [{"name": "carl", "source": "./plugins/carl"}],
    }
    _write_json(output / ".claude-plugin/marketplace.json", marketplace)
    _write_json(output / ".agents/plugins/marketplace.json", marketplace)
    record = {
        "schema_version": 1,
        "source": str(root),
        "source_digest": digest,
        "python": python,
        "version": native["version"],
        "snapshot": str(snapshot),
    }
    _write_json(output / "projection.json", record)
    return record


def _identity(path: Path) -> str | None:
    if path.is_symlink():
        return "symlink:" + str(path.readlink())
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PluginInstaller:
    """Modify only entries created by this installer and preserve foreign state."""

    def __init__(
        self, root: Path, *, home: Path | None = None, python: str = sys.executable
    ) -> None:
        self.root = root.resolve()
        self.home = home or Path.home()
        self.python = python
        self.state = self.home / ".carl/plugins/carl"
        self.marker = self.state / "install.json"

    def _load(self) -> dict[str, Any]:
        if self.marker.is_symlink():
            raise CARLError("Installer marker is a symlink", code="carl.plugin.ownership")
        return (
            json.loads(self.marker.read_text())
            if self.marker.exists()
            else {"files": {}, "hosts": []}
        )

    def installed_hosts(self) -> list[str]:
        """Return this installer's registered native hosts."""
        return self._load().get("hosts", [])

    def _owned_target(self, path: Path) -> None:
        current = path
        while current != self.home:
            if current.is_symlink():
                raise CARLError("Installer target is a symlink", code="carl.plugin.ownership")
            if current == current.parent:
                raise CARLError("Installer target escapes home", code="carl.plugin.path")
            current = current.parent

    def _put(self, path: Path, content: bytes, record: dict[str, Any]) -> None:
        self._owned_target(path)
        before = _identity(path)
        expected = record["files"].get(str(path))
        wanted = hashlib.sha256(content).hexdigest()
        if before is not None and before not in {expected, wanted}:
            raise CARLError("Foreign or edited carl entry", code="carl.plugin.ownership")
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, content)
        record["files"][str(path)] = wanted

    def install(self, hosts: list[str] | None = None, *, native: bool = True) -> dict[str, Any]:
        """Install discovered hosts and standalone aliases without replacing foreign skills."""
        self._owned_target(self.state)
        selected = hosts if hosts is not None else [host for host in HOSTS if shutil.which(host)]
        if any(host not in HOSTS for host in selected):
            raise CARLError("Unknown plugin host", code="carl.plugin.host")
        record = self._load()
        for name, expected in record.get("projection_files", {}).items():
            target = Path(name)
            self._owned_target(target)
            if _identity(target) != expected:
                raise CARLError("Edited projection entry", code="carl.plugin.ownership")
        skill = self.root / "skills/carl"
        destinations: list[Path] = []
        if "codex" in selected or "opencode" in selected:
            destinations.append(self.home / ".agents/skills/carl")
        if "claude" in selected:
            destinations.append(self.home / ".claude/skills/carl")
        for destination in destinations:
            for path in skill.rglob("*"):
                if path.is_file():
                    target = destination / path.relative_to(skill)
                    self._owned_target(target)
                    current = _identity(target)
                    wanted = hashlib.sha256(path.read_bytes()).hexdigest()
                    if current is not None and current not in {
                        record["files"].get(str(target)),
                        wanted,
                    }:
                        raise CARLError(
                            "Foreign or edited carl entry", code="carl.plugin.ownership"
                        )
        projection = project(self.root, self.state / "marketplace", python=self.python)
        for destination in destinations:
            for path in skill.rglob("*"):
                if path.is_file():
                    self._put(destination / path.relative_to(skill), path.read_bytes(), record)
        record.update(projection)
        record["projection_files"] = {
            str(path): _identity(path)
            for path in (self.state / "marketplace").rglob("*")
            if path.is_file()
        }
        record["hosts"] = selected
        _write_json(self.marker, record)
        if native:
            for host in selected:
                if host == "codex":
                    self._run(
                        [host, "plugin", "marketplace", "add", str(self.state / "marketplace")]
                    )
                    self._run([host, "plugin", "add", "carl@carl-local"])
                elif host == "claude":
                    self._run(
                        [host, "plugin", "marketplace", "add", str(self.state / "marketplace")]
                    )
                    self._run([host, "plugin", "install", "carl@carl-local", "--scope", "user"])
                    self._run([host, "plugin", "update", "carl@carl-local", "--scope", "user"])
                else:
                    self._opencode(record)
        _write_json(self.marker, record)
        return self.doctor()

    def _run(self, argv: list[str]) -> None:
        result = subprocess.run(argv, capture_output=True, timeout=60, check=False)
        if result.returncode:
            raise CARLError(f"{argv[0]} plugin command failed", code="carl.plugin.native")

    def _opencode(self, record: dict[str, Any]) -> None:
        path = self.home / ".config/opencode/opencode.json"
        self._owned_target(path)
        data: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
        servers: dict[str, Any] = data.setdefault("mcp", {})
        wanted = {
            "type": "local",
            "command": [self.python, "-m", "carl_studio.mcp"],
            "enabled": True,
        }
        existing = servers.get("carl")
        previous = record.get("opencode_entry")
        if existing is not None and existing not in (previous, wanted):
            raise CARLError("Foreign OpenCode carl server", code="carl.plugin.ownership")
        servers["carl"] = wanted
        _write_json(path, data)
        record["opencode_entry"] = wanted

    def doctor(self) -> dict[str, Any]:
        """Compare canonical, projected, and installer-owned bytes."""
        record = self._load()
        drift = [
            path for path, expected in record["files"].items() if _identity(Path(path)) != expected
        ]
        digest = source_digest(self.root)
        if record.get("source_digest") != digest:
            drift.append("source_digest")
        for path, expected in record.get("projection_files", {}).items():
            if _identity(Path(path)) != expected:
                drift.append(path)
        if record.get("opencode_entry"):
            path = self.home / ".config/opencode/opencode.json"
            try:
                self._owned_target(path)
                data = json.loads(path.read_text())
                if data.get("mcp", {}).get("carl") != record["opencode_entry"]:
                    drift.append(str(path))
            except (CARLError, OSError, ValueError):
                drift.append(str(path))
        return {
            "installed": bool(record.get("hosts")),
            "hosts": record.get("hosts", []),
            "source_digest": digest,
            "drift": drift,
            "healthy": not drift,
            "runtime": self.python,
        }

    def uninstall(self, *, native: bool = True) -> dict[str, Any]:
        """Remove only matching owned entries; retain edited content."""
        record = self._load()
        retained: list[str] = []
        for name, expected in record["files"].items():
            path = Path(name)
            try:
                self._owned_target(path)
            except CARLError:
                retained.append(name)
                continue
            if _identity(path) == expected:
                path.unlink()
            elif path.exists() or path.is_symlink():
                retained.append(name)
        if native:
            for host in record.get("hosts", []):
                if host in {"codex", "claude"}:
                    command = "remove" if host == "codex" else "uninstall"
                    self._run([host, "plugin", command, "carl@carl-local"])
        path = self.home / ".config/opencode/opencode.json"
        opencode_owned = True
        try:
            self._owned_target(path)
        except CARLError:
            opencode_owned = False
            retained.append(str(path))
        if record.get("opencode_entry") and path.exists() and opencode_owned:
            data = json.loads(path.read_text())
            if data.get("mcp", {}).get("carl") == record["opencode_entry"]:
                del data["mcp"]["carl"]
                _write_json(path, data)
            else:
                retained.append(str(path))
        record["hosts"] = []
        _write_json(self.marker, record)
        return {"removed": True, "retained": retained}
