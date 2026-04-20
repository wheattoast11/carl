"""Tests for the ``carl metrics`` Typer sub-app.

Covers: missing-extra exit code, help rendering, and end-to-end bind +
scrape + graceful shutdown on SIGINT.
"""

from __future__ import annotations

import http.client
import threading
import time

import pytest
from typer.testing import CliRunner


@pytest.fixture(autouse=True)
def _reset_metrics_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the metrics singleton per test so scrape output is deterministic."""
    from carl_studio import metrics as metrics_mod

    monkeypatch.setattr(metrics_mod, "_registry_instance", None, raising=False)


def test_metrics_serve_without_extra_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing extra must surface a clear install hint and exit code 2."""
    from carl_studio.cli import metrics as cli_metrics

    monkeypatch.setattr(cli_metrics, "is_available", lambda: False)
    runner = CliRunner()
    result = runner.invoke(cli_metrics.metrics_app, ["serve", "--port", "0"])
    assert result.exit_code == 2
    assert "prometheus-client not installed" in result.output
    assert "pip install 'carl-studio[metrics]'" in result.output


def test_metrics_serve_help_lists_flags() -> None:
    """``--help`` must list the ``--port`` and ``--host`` flags."""
    from carl_studio.cli.metrics import metrics_app

    runner = CliRunner()
    result = runner.invoke(metrics_app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.output
    assert "--host" in result.output


def test_metrics_serve_binds_and_shuts_down() -> None:
    """End-to-end: bind on OS-assigned port, scrape ``/metrics``, shut down.

    We run ``serve`` on a worker thread with port=0 so the OS picks a free
    port. The actual bound port is discovered by monkeypatching
    ``prometheus_client.start_http_server`` to capture the server handle,
    then we issue a real HTTP GET against ``/metrics`` and verify the
    Prometheus text content-type and a known metric line. Finally we
    raise SIGINT on the main process (the CLI installs a handler) and
    ``join`` the worker to verify clean shutdown.
    """
    pytest.importorskip("prometheus_client")
    from prometheus_client import start_http_server as real_start

    from carl_studio import metrics as metrics_mod
    from carl_studio.cli import metrics as cli_metrics

    captured: dict[str, object] = {}

    def _wrap(port: int, **kwargs: object) -> object:
        server, thread = real_start(port, **kwargs)  # type: ignore[arg-type]
        captured["server"] = server
        captured["thread"] = thread
        return server, thread

    # Seed a metric so the scrape output has something to assert on.
    metrics_mod.get_registry().record_training_step()

    # Patch at the cli module's reference — the function imports
    # ``start_http_server`` lazily inside ``serve``.
    import prometheus_client

    original = prometheus_client.start_http_server
    prometheus_client.start_http_server = _wrap  # type: ignore[assignment]

    runner = CliRunner()
    # Run the CLI on a thread so the blocking loop doesn't deadlock the test.
    result_holder: dict[str, object] = {}

    def _run() -> None:
        try:
            result_holder["result"] = runner.invoke(
                cli_metrics.metrics_app,
                ["serve", "--port", "0", "--host", "127.0.0.1"],
                catch_exceptions=False,
            )
        except BaseException as exc:  # pragma: no cover - defensive
            result_holder["error"] = exc

    worker = threading.Thread(target=_run, name="cli-metrics-test", daemon=True)
    worker.start()

    try:
        # Wait for the server to come up (poll the captured handle).
        deadline = time.monotonic() + 5.0
        while "server" not in captured and time.monotonic() < deadline:
            time.sleep(0.05)
        assert "server" in captured, "metrics server failed to start within deadline"

        server = captured["server"]
        bound_port = server.server_address[1]  # type: ignore[attr-defined]
        assert bound_port > 0

        # Scrape /metrics via a real HTTP GET.
        conn = http.client.HTTPConnection("127.0.0.1", bound_port, timeout=2.0)
        try:
            conn.request("GET", "/metrics")
            response = conn.getresponse()
            body = response.read().decode("utf-8")
        finally:
            conn.close()

        assert response.status == 200
        content_type = response.getheader("Content-Type") or ""
        # Prometheus exposition is "text/plain; version=0.0.4; charset=utf-8"
        assert "text/plain" in content_type
        assert "carl_training_steps_total" in body

        # Graceful shutdown: we're on a pytest worker thread, so the CLI's
        # ``signal.signal`` install will have raised ``ValueError`` and
        # fallen back to polling ``server._BaseServer__is_shut_down``.
        # Triggering ``server.shutdown()`` here flips that event and the
        # CLI's polling loop breaks out of its ``while not stopping`` block.
        server.shutdown()  # type: ignore[attr-defined]
        worker.join(timeout=5.0)
    finally:
        prometheus_client.start_http_server = original  # type: ignore[assignment]
        # Make sure the worker never outlives the test.
        if worker.is_alive():
            server_handle = captured.get("server")
            if server_handle is not None:
                try:
                    server_handle.shutdown()  # type: ignore[attr-defined]
                except BaseException:
                    pass
            worker.join(timeout=2.0)

    assert not worker.is_alive(), "CLI worker did not shut down cleanly"
    if "result" in result_holder:
        cli_result = result_holder["result"]
        # Exit code 0 (stopped normally) or None (runner swallowed None).
        assert cli_result.exit_code == 0  # type: ignore[attr-defined]
        assert "carl metrics stopped." in cli_result.output  # type: ignore[attr-defined]
