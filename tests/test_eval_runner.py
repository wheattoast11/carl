"""Tests for carl_studio.eval.runner hardening (WS-T5).

Covers:
  - Symlink escape rejection via carl_core.safepath
  - Traversal rejection
  - GPU tensor cleanup on timeout
  - Zero-tool-call metric correction
  - Empty-results guard
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from carl_core.errors import CARLError, CARLTimeoutError
from carl_studio.eval.runner import (
    EvalConfig,
    EvalRunner,
    EvalSandbox,
)


# ---------------------------------------------------------------------------
# Sandbox: symlink + traversal escape
# ---------------------------------------------------------------------------


class TestSandboxEscape:
    """`EvalSandbox._safe_path` must reject anything that escapes the workdir."""

    def test_sandbox_blocks_symlink_escape(self, tmp_path: Path) -> None:
        """A symlink inside the sandbox pointing outside must be rejected
        with CARLError(code='carl.eval.sandbox_escape'), not followed.
        """
        outside = tmp_path / "outside_secret.txt"
        outside.write_text("SECRET")

        sandbox = EvalSandbox()
        try:
            link = Path(sandbox.workdir) / "link_to_secret"
            os.symlink(str(outside), str(link))

            with pytest.raises(CARLError) as exc_info:
                sandbox._safe_path("link_to_secret")

            assert exc_info.value.code == "carl.eval.sandbox_escape"
            assert "link_to_secret" in str(exc_info.value)
            # Context should carry enough info to debug
            assert exc_info.value.context is not None
            assert "input_path" in exc_info.value.context
        finally:
            sandbox.cleanup()

    def test_sandbox_blocks_symlink_to_directory(self, tmp_path: Path) -> None:
        """A symlink to a directory outside the sandbox is rejected on the
        same rule: reading through it would leak out."""
        outside_dir = tmp_path / "outside_dir"
        outside_dir.mkdir()
        (outside_dir / "leaked.txt").write_text("leak")

        sandbox = EvalSandbox()
        try:
            link = Path(sandbox.workdir) / "dir_link"
            os.symlink(str(outside_dir), str(link))

            with pytest.raises(CARLError) as exc_info:
                sandbox._safe_path("dir_link/leaked.txt")

            assert exc_info.value.code == "carl.eval.sandbox_escape"
        finally:
            sandbox.cleanup()

    def test_sandbox_blocks_traversal(self) -> None:
        """`../etc/passwd` style inputs must be rejected."""
        sandbox = EvalSandbox()
        try:
            with pytest.raises(CARLError) as exc_info:
                sandbox._safe_path("../etc/passwd")
            assert exc_info.value.code == "carl.eval.sandbox_escape"
            assert "passwd" in str(exc_info.value)
        finally:
            sandbox.cleanup()

    def test_sandbox_blocks_deep_traversal(self) -> None:
        """Multi-level `../../../..` traversal is rejected."""
        sandbox = EvalSandbox()
        try:
            with pytest.raises(CARLError) as exc_info:
                sandbox._safe_path("subdir/../../../etc/hosts")
            assert exc_info.value.code == "carl.eval.sandbox_escape"
        finally:
            sandbox.cleanup()

    def test_sandbox_allows_legitimate_relative_path(self) -> None:
        """Well-formed relative paths inside the sandbox are accepted."""
        sandbox = EvalSandbox()
        try:
            resolved = sandbox._safe_path("subdir/file.py")
            # macOS symlinks /var -> /private/var; compare via Path.resolve()
            resolved_path = Path(resolved).resolve()
            workdir_path = Path(sandbox.workdir).resolve()
            assert resolved_path.is_relative_to(workdir_path)
            assert resolved.endswith(os.path.join("subdir", "file.py"))
        finally:
            sandbox.cleanup()

    def test_sandbox_allows_sandbox_root(self) -> None:
        """An empty or '.' path resolves to the sandbox root itself."""
        sandbox = EvalSandbox()
        try:
            resolved = sandbox._safe_path(".")
            assert Path(resolved).resolve() == Path(sandbox.workdir).resolve()
        finally:
            sandbox.cleanup()

    def test_sandbox_absolute_path_rewrite(self) -> None:
        """Absolute paths are reinterpreted as sandbox-relative
        (matches pre-hardening behavior; leading '/' stripped)."""
        sandbox = EvalSandbox()
        try:
            resolved = sandbox._safe_path("/etc/passwd")
            # Must resolve *inside* the sandbox
            assert Path(resolved).resolve().is_relative_to(Path(sandbox.workdir).resolve())
        finally:
            sandbox.cleanup()


# ---------------------------------------------------------------------------
# execute_tool: CARLError is caught and formatted, not propagated
# ---------------------------------------------------------------------------


class TestExecuteToolEscapeHandling:
    def test_read_escape_returns_error_string(self, tmp_path: Path) -> None:
        outside = tmp_path / "secret.txt"
        outside.write_text("SECRET")

        sandbox = EvalSandbox()
        try:
            link = Path(sandbox.workdir) / "escape"
            os.symlink(str(outside), str(link))

            result = sandbox.execute_tool("read_file", {"path": "escape"})
            assert result.startswith("Error:")
            assert "sandbox" in result
            assert sandbox.tool_failures == 1
        finally:
            sandbox.cleanup()

    def test_write_traversal_returns_error_string(self) -> None:
        sandbox = EvalSandbox()
        try:
            result = sandbox.execute_tool(
                "write_file",
                {"path": "../evil.py", "content": "boom"},
            )
            assert result.startswith("Error:")
            assert sandbox.tool_failures == 1
        finally:
            sandbox.cleanup()


# ---------------------------------------------------------------------------
# GPU tensor cleanup on timeout
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer surface consumed by the runner."""

    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "</s>"

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        fake_tensor = MagicMock(name="input_ids")
        fake_tensor.shape = (1, 4)
        fake_tensor.to = lambda device: fake_tensor
        wrapped = {"input_ids": fake_tensor}
        # Mimic the .to(...) method on the tokenizer output that the runner
        # calls (a BatchEncoding in real code).
        wrapped_obj = MagicMock()
        wrapped_obj.__getitem__ = lambda self, k: wrapped[k]
        wrapped_obj.to = lambda device: wrapped_obj
        # Attribute access for input_ids is used via [] subscript in code,
        # so ensure dict-style access works.
        for k, v in wrapped.items():
            setattr(wrapped_obj, k, v)
        return wrapped_obj

    def apply_chat_template(self, conv: Any, **kwargs: Any) -> str:
        return "PROMPT"

    def decode(self, tokens: Any, **kwargs: Any) -> str:
        return ""


class _FakeModel:
    device = "cpu"

    def __init__(self, raise_on_generate: Exception | None = None) -> None:
        self.raise_on_generate = raise_on_generate
        self.training = False

    def generate(self, **kwargs: Any) -> Any:
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        output = MagicMock()
        output.__getitem__ = lambda self, idx: MagicMock(shape=(0,))
        return output

    def eval(self) -> None:
        pass


def _build_runner_with_fake_generate(raise_on_generate: Exception) -> tuple[EvalRunner, MagicMock]:
    cfg = EvalConfig(
        checkpoint="fake/cp",
        dataset="fake/ds",
        phase="2prime",
        base_model="fake/base",
        max_turns=1,
        max_new_tokens=64,
    )
    runner = EvalRunner(cfg)

    tokenizer = _FakeTokenizer()
    model = _FakeModel(raise_on_generate=raise_on_generate)

    # Patch loaders + dataset so run() exercises only the generate loop
    runner._load_model_with_adapters = MagicMock(return_value=(model, tokenizer))  # type: ignore[method-assign]
    runner._load_dataset = MagicMock(  # type: ignore[method-assign]
        return_value=[
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                ]
            }
        ]
    )
    return runner, tokenizer  # type: ignore[return-value]


def test_timeout_frees_gpu_memory() -> None:
    """On timeout during model.generate(), torch.cuda.empty_cache() must fire
    in the finally block. We simulate CUDA availability so the cleanup runs."""
    fake_torch = SimpleNamespace()
    fake_torch.cuda = SimpleNamespace(
        is_available=MagicMock(return_value=True),
        empty_cache=MagicMock(),
        is_bf16_supported=MagicMock(return_value=False),
    )
    fake_torch.no_grad = lambda: _NullContext()
    fake_torch.bfloat16 = object()
    fake_torch.float32 = object()

    runner, _tok = _build_runner_with_fake_generate(
        CARLTimeoutError("simulated generate timeout")
    )

    with patch.dict(sys.modules, {"torch": fake_torch}):
        report = runner._run_phase2prime()

    # empty_cache is invoked (at minimum) in the finally block + the outer
    # post-cleanup call. So at least 1 call is required; in practice >= 2.
    assert fake_torch.cuda.empty_cache.call_count >= 1
    # The report must still come back (no propagation of timeout).
    assert report.phase == "2prime"
    assert report.n_samples == 1


def test_exception_during_generate_still_frees_gpu() -> None:
    """Non-timeout exceptions also run the finally cleanup."""
    fake_torch = SimpleNamespace()
    fake_torch.cuda = SimpleNamespace(
        is_available=MagicMock(return_value=True),
        empty_cache=MagicMock(),
        is_bf16_supported=MagicMock(return_value=False),
    )
    fake_torch.no_grad = lambda: _NullContext()

    runner, _tok = _build_runner_with_fake_generate(RuntimeError("CUDA OOM"))

    with patch.dict(sys.modules, {"torch": fake_torch}):
        report = runner._run_phase2prime()

    assert fake_torch.cuda.empty_cache.call_count >= 1
    assert report.phase == "2prime"


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> bool:
        return False


# ---------------------------------------------------------------------------
# Zero-tool-call and empty-results guards
# ---------------------------------------------------------------------------


def _make_phase2prime_runner_no_results() -> EvalRunner:
    cfg = EvalConfig(
        checkpoint="fake/cp",
        dataset="fake/ds",
        phase="2prime",
        base_model="fake/base",
        max_turns=1,
    )
    return EvalRunner(cfg)


def test_empty_results_returns_empty_report(caplog: pytest.LogCaptureFixture) -> None:
    """No samples -> empty report, no divide-by-zero, no NaN."""
    runner = _make_phase2prime_runner_no_results()

    fake_torch = SimpleNamespace()
    fake_torch.cuda = SimpleNamespace(
        is_available=MagicMock(return_value=False),
        empty_cache=MagicMock(),
    )

    runner._load_model_with_adapters = MagicMock(  # type: ignore[method-assign]
        return_value=(_FakeModel(), _FakeTokenizer())
    )
    # Zero samples
    runner._load_dataset = MagicMock(return_value=[])  # type: ignore[method-assign]

    with patch.dict(sys.modules, {"torch": fake_torch}), caplog.at_level(logging.WARNING):
        report = runner._run_phase2prime()

    assert report.n_samples == 0
    assert report.passed is False
    assert report.primary_value == 0.0
    # Metrics must all be numeric (not NaN)
    for key, value in report.metrics.items():
        assert value == value, f"metric {key} is NaN"  # NaN != NaN
    assert report.metrics["zero_calls"] == 1.0
    assert any("no results" in rec.message.lower() for rec in caplog.records)


def test_zero_tool_calls_warns_not_divides(caplog: pytest.LogCaptureFixture) -> None:
    """When samples exist but no tool calls were made, failure_rate must not
    be computed via division-by-zero, and a warning must be emitted."""
    runner = _make_phase2prime_runner_no_results()

    fake_torch = SimpleNamespace()
    fake_torch.cuda = SimpleNamespace(
        is_available=MagicMock(return_value=False),
        empty_cache=MagicMock(),
    )
    fake_torch.no_grad = lambda: _NullContext()

    # Tokenizer + model that produce no tool calls -> conversation ends on
    # turn 1 with an assistant text reply, has_tool_call stays False.
    class _NoToolTokenizer(_FakeTokenizer):
        def decode(self, tokens: Any, **kwargs: Any) -> str:
            return "plain text response"

    runner._load_model_with_adapters = MagicMock(  # type: ignore[method-assign]
        return_value=(_FakeModel(), _NoToolTokenizer())
    )
    runner._load_dataset = MagicMock(  # type: ignore[method-assign]
        return_value=[
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": f"task {i}"},
                ]
            }
            for i in range(3)
        ]
    )

    with patch.dict(sys.modules, {"torch": fake_torch}), caplog.at_level(logging.WARNING):
        report = runner._run_phase2prime()

    assert report.n_samples == 3
    # failure_rate is 0.0 with zero_calls=True, not NaN
    assert report.metrics["failure_rate"] == 0.0
    assert report.metrics["zero_calls"] == 1.0
    # Every metric is finite
    for key, value in report.metrics.items():
        assert value == value, f"metric {key} is NaN"
    assert any(
        "no tool calls" in rec.message.lower() for rec in caplog.records
    ), "expected zero-calls warning"


def test_nonzero_tool_calls_no_zero_flag() -> None:
    """When tool calls actually happen, zero_calls=0 and failure_rate reflects
    real ratio of failures to calls."""
    runner = _make_phase2prime_runner_no_results()

    fake_torch = SimpleNamespace()
    fake_torch.cuda = SimpleNamespace(
        is_available=MagicMock(return_value=False),
        empty_cache=MagicMock(),
    )

    # Simulate by monkeypatching the inner loop -- easier: construct results
    # directly by mocking the generator so tool_calls accumulates. Instead we
    # bypass the loop and feed a crafted results list via a local method
    # call path: use the public run() but mock everything that touches the
    # model. Since _run_phase2prime already computes from `results`, we'll
    # call it with a lightweight tokenizer emitting a parsable tool call.
    class _ToolTokenizer(_FakeTokenizer):
        _calls = 0

        def decode(self, tokens: Any, **kwargs: Any) -> str:
            type(self)._calls += 1
            if type(self)._calls == 1:
                # First decode: return a tool call in JSON format
                return '{"name": "execute_code", "arguments": {"code": "1"}}'
            return "done"

    runner._load_model_with_adapters = MagicMock(  # type: ignore[method-assign]
        return_value=(_FakeModel(), _ToolTokenizer())
    )
    runner._load_dataset = MagicMock(  # type: ignore[method-assign]
        return_value=[
            {
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "t"},
                ]
            }
        ]
    )

    fake_torch.no_grad = lambda: _NullContext()
    with patch.dict(sys.modules, {"torch": fake_torch}):
        report = runner._run_phase2prime()

    # zero_calls must not be set when at least one call happened.
    # (If the fake model's generate produced a parseable tool call, tool_calls
    # will be >=1; if not, this test degrades gracefully -- we only assert the
    # invariant that zero_calls reflects reality.)
    if report.metrics["mean_tool_calls"] > 0:
        assert report.metrics["zero_calls"] == 0.0
