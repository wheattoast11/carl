"""``carl metrics`` CLI — Prometheus scrape endpoint.

Thin Typer sub-app wrapping :func:`prometheus_client.start_http_server` so
operators can expose CARL metrics to their Prometheus scraper without
depending on the heartbeat daemon being up. When the ``[metrics]`` extra
is absent the commands exit with a clear install hint.
"""

from __future__ import annotations

import signal
import time
from types import FrameType

import typer

from carl_studio.metrics import get_registry, is_available

metrics_app = typer.Typer(help="Prometheus metrics commands.", no_args_is_help=True)


@metrics_app.callback()
def _metrics_root() -> None:  # pyright: ignore[reportUnusedFunction]
    """Force Typer to treat this as a sub-command group.

    Without an explicit callback, Typer collapses a single-command app
    into its sole command — which breaks ``carl metrics serve`` routing
    when the sub-app is tested standalone. The callback is a no-op.
    """


@metrics_app.command("serve")
def serve(
    port: int = typer.Option(9464, "--port", "-p", help="HTTP port to bind."),
    host: str = typer.Option("127.0.0.1", "--host", help="Host/IP to bind."),
) -> None:
    """Serve Prometheus scrape endpoint on ``http://<host>:<port>/metrics``.

    Blocks until SIGINT / SIGTERM. The underlying ``start_http_server``
    spawns a daemon thread; we install signal handlers so Ctrl-C shuts
    down cleanly instead of leaving the WSGI server and its thread alive
    on process exit paths that rely on Typer's runtime.
    """
    if not is_available():
        typer.echo(
            "ERROR: prometheus-client not installed. Install with:\n"
            "  pip install 'carl-studio[metrics]'",
            err=True,
        )
        raise typer.Exit(code=2)

    from prometheus_client import start_http_server

    registry = get_registry().registry
    server, thread = start_http_server(port, addr=host, registry=registry)
    # ``port=0`` yields an OS-assigned port; surface the actual bound port
    # so tests (and humans) can discover where the endpoint landed.
    bound_port = server.server_address[1]
    typer.echo(f"carl metrics serving on http://{host}:{bound_port}/metrics")

    stopping = {"val": False}

    def _stop(_signum: int, _frame: FrameType | None) -> None:
        stopping["val"] = True

    # ``signal.signal`` raises ``ValueError`` when called off the main
    # thread (e.g. under pytest worker threads). That's a legitimate
    # production constraint — users running ``carl metrics serve`` from a
    # terminal always hit the main thread — but we must not crash the
    # command just because a test driver invoked us from elsewhere.
    previous_sigint: object = None
    previous_sigterm: object = None
    try:
        previous_sigint = signal.signal(signal.SIGINT, _stop)
        previous_sigterm = signal.signal(signal.SIGTERM, _stop)
        signals_installed = True
    except ValueError:
        signals_installed = False

    try:
        while not stopping["val"]:
            # When running on a non-main thread, we rely on the server
            # being shut down externally (the test driver closes it and
            # we observe ``server.socket._closed``). The polite stop-flag
            # check above is still the primary termination signal.
            if not signals_installed:
                try:
                    # Probe the server socket: once closed, break out.
                    shutdown_event = getattr(
                        server, "_BaseServer__is_shut_down", None,
                    )
                    if shutdown_event is not None and bool(shutdown_event.is_set()):
                        break
                except Exception:  # pragma: no cover - defensive
                    pass
            time.sleep(0.25)
    finally:
        try:
            server.shutdown()
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            server.server_close()
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            thread.join(timeout=2.0)
        except Exception:  # pragma: no cover - defensive
            pass
        # Restore prior handlers so nested invocations (tests) don't leak
        # state across runs.
        if signals_installed:
            try:
                signal.signal(signal.SIGINT, previous_sigint)  # type: ignore[arg-type]
                signal.signal(signal.SIGTERM, previous_sigterm)  # type: ignore[arg-type]
            except (ValueError, TypeError):  # pragma: no cover - defensive
                pass
        typer.echo("carl metrics stopped.")


__all__ = ["metrics_app"]
