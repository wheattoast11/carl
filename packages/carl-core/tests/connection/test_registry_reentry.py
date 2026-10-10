"""Connection finalizers may reenter registration during garbage collection."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_collection_during_registration_can_unregister() -> None:
    script = textwrap.dedent("""
        import gc
        import weakref
        from carl_core.connection.registry import ConnectionRegistry

        gc.disable()
        registry = ConnectionRegistry()

        class Connection:
            def __init__(self, name):
                self.connection_id = name
                self.cycle = self

            def __del__(self):
                registry.unregister(self.connection_id)

        old = Connection("old")
        registry.register(old)
        del old
        original = weakref.WeakValueDictionary.__setitem__

        def collect_then_set(self, key, value):
            gc.collect()
            original(self, key, value)

        weakref.WeakValueDictionary.__setitem__ = collect_then_set
        new = Connection("new")
        registry.register(new)
        assert registry.get("old") is None
        assert registry.get("new") is new
        weakref.WeakValueDictionary.__setitem__ = original
        del new
        gc.collect()
        assert registry.count() == 0
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
        capture_output=True,
        check=False,
        text=True,
        timeout=3,
    )
    assert result.returncode == 0, result.stderr
