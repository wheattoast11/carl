"""Audit exact locked releases against public advisories and release metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SNAPSHOT = "2026-10-08T00:35:32Z"


def _read(url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def audit(lock: Path) -> dict[str, Any]:
    """Bind an advisory lookup to exact registry versions and lock bytes."""
    payload = lock.read_bytes()
    document = tomllib.loads(payload.decode())
    packages = [
        package for package in document["package"] if "registry" in package.get("source", {})
    ]
    queries = [
        {"package": {"name": package["name"], "ecosystem": "PyPI"}, "version": package["version"]}
        for package in packages
    ]
    matches: list[dict[str, Any]] = []
    for offset in range(0, len(queries), 128):
        response = _read(
            "https://api.osv.dev/v1/querybatch", {"queries": queries[offset : offset + 128]}
        )
        for package, result in zip(
            packages[offset : offset + 128], response["results"], strict=True
        ):
            if result.get("vulns"):
                matches.append(
                    {
                        "name": package["name"],
                        "version": package["version"],
                        "advisories": [item["id"] for item in result["vulns"]],
                    }
                )

    def release(package: dict[str, Any]) -> dict[str, Any]:
        name, version = package["name"], package["version"]
        url = f"https://pypi.org/pypi/{name}/{version}/json"
        metadata = _read(url)
        files = metadata.get("urls", [])
        return {
            "name": name,
            "version": version,
            "source": url,
            "first_upload": min((item["upload_time_iso_8601"] for item in files), default=None),
            "all_yanked": bool(files) and all(item["yanked"] for item in files),
            "requires_python": metadata["info"].get("requires_python"),
        }

    with ThreadPoolExecutor(max_workers=4) as executor:
        releases = list(executor.map(release, packages))
    return {
        "schema_version": 1,
        "queried_at": datetime.now(UTC).isoformat(),
        "release_snapshot": SNAPSHOT,
        "lock_sha256": hashlib.sha256(payload).hexdigest(),
        "registry_entries_checked": len(packages),
        "advisory_matches": matches,
        "release_metadata": releases,
        "evidence_class": "registry metadata and advisory lookup",
        "limits": "Dependency matches require operation-level applicability review.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=Path("uv.lock"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.lock)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
