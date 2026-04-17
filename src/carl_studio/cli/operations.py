"""Operation registry for ``carl flow`` slash-prefixed chains.

Each operation is ``(chain, args) -> chain``. Operations look up their
dependencies lazily — they import the real command functions inside the op
body to avoid circular imports and to keep this module fast to load.

Every op emits a single ``ActionType.CLI_CMD`` step on the chain:

- Success  → ``success=True``, ``output={"ok": True, ...}``
- SystemExit(code) → ``success=(code == 0)``, ``output={"exit_code": code}``
- Other exception → ``success=False``, ``output={"error": str(exc), ...}``

Adding an operation:

    def simplify_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
        return _run(chain, "simplify", args, lambda: _invoke_something(args))

    OPERATIONS["simplify"] = simplify_op
"""
from __future__ import annotations

from time import monotonic
from typing import Any, Callable, cast

import typer

from carl_core.interaction import ActionType, InteractionChain

Operation = Callable[[InteractionChain, list[str]], InteractionChain]

__all__ = ["OPERATIONS", "Operation", "get_operation", "list_operations"]


# ---------------------------------------------------------------------------
# Core execution helper — records one CLI_CMD step with unified outcome shape
# ---------------------------------------------------------------------------


def _run(
    chain: InteractionChain,
    name: str,
    args: list[str],
    thunk: Callable[[], Any],
    *,
    extra_input: dict[str, Any] | None = None,
) -> InteractionChain:
    """Invoke ``thunk`` and record a single step on ``chain``.

    ``thunk`` is the zero-arg callable that executes the underlying command.
    Classification rules:

    - Returns cleanly → ``success=True``.
    - Raises ``SystemExit(code)`` → ``success=(code == 0)``,
      ``output={"exit_code": code}``. Non-zero codes are failures;
      zero is success.
    - Raises any other exception → ``success=False``,
      ``output={"error": str(exc), "type": exc.__class__.__name__}``.
    """
    started = monotonic()
    step_input: dict[str, Any] = {"args": list(args)}
    if extra_input:
        for _k, _v in extra_input.items():
            step_input[str(_k)] = _v

    try:
        result = thunk()
    except SystemExit as exc:
        code_raw = exc.code
        code = int(code_raw) if isinstance(code_raw, int) else (0 if code_raw is None else 1)
        chain.record(
            ActionType.CLI_CMD,
            name,
            input=step_input,
            output={"exit_code": code},
            success=(code == 0),
            duration_ms=(monotonic() - started) * 1000,
        )
        return chain
    except BaseException as exc:  # noqa: BLE001 — we want to record and return
        output: dict[str, Any] = {
            "error": str(exc),
            "type": exc.__class__.__name__,
        }
        explicit_code = getattr(exc, "code", None)
        if isinstance(explicit_code, int):
            output["code"] = explicit_code
        chain.record(
            ActionType.CLI_CMD,
            name,
            input=step_input,
            output=output,
            success=False,
            duration_ms=(monotonic() - started) * 1000,
        )
        return chain

    step_output: dict[str, Any] = {"ok": True}
    if isinstance(result, dict):
        _result_dict = cast(dict[Any, Any], result)
        for _k in list(_result_dict.keys()):
            step_output[str(_k)] = _result_dict[_k]
    elif result is not None:
        step_output["result"] = str(result)[:500]
    chain.record(
        ActionType.CLI_CMD,
        name,
        input=step_input,
        output=step_output,
        success=True,
        duration_ms=(monotonic() - started) * 1000,
    )
    return chain


# ---------------------------------------------------------------------------
# Ops: lifecycle
# ---------------------------------------------------------------------------


def _doctor_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    def _do() -> None:
        from carl_studio.cli.startup import doctor as doctor_cmd

        doctor_cmd(check_freshness=False, json_output=False)

    return _run(chain, "doctor", args, _do)


def _start_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    def _do() -> None:
        from carl_studio.cli.startup import start as start_cmd

        start_cmd(inventory=False, json_output=False)

    return _run(chain, "start", args, _do)


def _init_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    def _do() -> None:
        from carl_studio.cli.init import init_cmd

        init_cmd(
            skip_extras=False,
            skip_project=False,
            force=False,
            json_output=False,
        )

    return _run(chain, "init", args, _do)


def _freshness_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    def _do() -> dict[str, Any]:
        from carl_studio.freshness import run_freshness_check

        report = run_freshness_check(force=True)
        # Side-effect: stash the summary on the chain for downstream ops.
        chain.context["freshness"] = report.summary
        return {"summary": report.summary, "has_issues": bool(report.has_issues)}

    return _run(chain, "freshness", args, _do)


def _doctor_freshness_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Run ``carl doctor --check-freshness`` (doctor + freshness audit)."""

    def _do() -> None:
        from carl_studio.cli.startup import doctor as doctor_cmd

        doctor_cmd(check_freshness=True, json_output=False)

    return _run(chain, "doctor_freshness", args, _do)


def _project_status_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Show the current project configuration via ``carl project show``.

    Uses the caller-supplied ``--config`` arg if present; otherwise carl.yaml.
    """

    def _do() -> None:
        from carl_studio.cli.project_data import project_show

        config_path = "carl.yaml"
        if args:
            for i, tok in enumerate(args):
                if tok in ("--config", "-c") and i + 1 < len(args):
                    config_path = args[i + 1]
                    break
                if tok.startswith("--config="):
                    config_path = tok.split("=", 1)[1]
                    break
        project_show(config=config_path)

    return _run(chain, "project_status", args, _do)


# ---------------------------------------------------------------------------
# Ops: agentic assistance
# ---------------------------------------------------------------------------


def _ask_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    prompt = " ".join(args).strip() or "hello"

    def _do() -> None:
        from carl_studio.cli.chat import run_one_shot_agent

        run_one_shot_agent(prompt)

    return _run(chain, "ask", args, _do, extra_input={"prompt": prompt[:200]})


def _echo_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Trivial pass-through — useful for tests and debugging chains.

    Records the args as a successful CLI_CMD step without emitting to
    stdout, so callers that parse ``carl flow --json`` can rely on the
    JSON payload being the only line on stdout.
    """
    msg = " ".join(args)
    chain.record(
        ActionType.CLI_CMD,
        "echo",
        input={"args": args},
        output={"message": msg},
    )
    return chain


def _chat_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Open the interactive chat loop. Args become the first user turn if provided."""

    def _do() -> None:
        if args:
            from carl_studio.cli.chat import run_one_shot_agent

            run_one_shot_agent(" ".join(args))
        else:
            from carl_studio.cli.chat import chat_cmd

            chat_cmd()

    return _run(
        chain,
        "chat",
        args,
        _do,
        extra_input={"first_turn": " ".join(args)[:200]} if args else None,
    )


def _train_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Hand the chain off to the agent as a training instruction."""
    prompt = "train " + " ".join(args) if args else "Help me set up a training run for this project."

    def _do() -> None:
        from carl_studio.cli.chat import run_one_shot_agent

        run_one_shot_agent(prompt)

    return _run(chain, "train", args, _do, extra_input={"prompt": prompt[:200]})


def _review_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Ask Carl for a code/config review on the given target."""
    target = " ".join(args) or "the most recent change"
    prompt = (
        f"Review {target}. Look for bugs, bad patterns, and things that would confuse "
        f"a new reader. Keep it under 200 words."
    )

    def _do() -> None:
        from carl_studio.cli.chat import run_one_shot_agent

        run_one_shot_agent(prompt)

    return _run(chain, "review", args, _do, extra_input={"target": target})


def _simplify_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Ask Carl to simplify the target file or module."""
    target = " ".join(args) or "the current file"
    prompt = (
        f"Simplify {target}. Remove dead code, redundant abstractions, and anything that "
        f"doesn't pull weight. Preserve behavior. Report each simplification before applying."
    )

    def _do() -> None:
        from carl_studio.cli.chat import run_one_shot_agent

        run_one_shot_agent(prompt)

    return _run(chain, "simplify", args, _do, extra_input={"target": target})


def _ship_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Run the full quality pipeline via the agent: tests, lint, type, bundle, commit."""
    extra = " ".join(args)
    prompt = (
        "Ship this: run tests, run lint, run type checks, fix any failures, then stage "
        "and commit the fix. Report what was green, what was red, and what you changed."
    )
    if extra:
        prompt += f"\n\nExtra instructions: {extra}"

    def _do() -> None:
        from carl_studio.cli.chat import run_one_shot_agent

        run_one_shot_agent(prompt)

    return _run(chain, "ship", args, _do, extra_input={"extra": extra[:200]})


# ---------------------------------------------------------------------------
# Ops: workbench commands (bench / align / learn / eval / infer / publish / push)
# ---------------------------------------------------------------------------


def _bench_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Wrap ``carl lab bench <model_id>``. First positional arg is model ID."""
    if not args:
        def _missing() -> None:
            raise SystemExit(2)

        return _run(
            chain,
            "bench",
            args,
            _missing,
            extra_input={"error": "bench requires a model_id argument"},
        )

    model_id = args[0]

    def _do() -> None:
        from carl_studio.cli import lab as _lab

        _lab.bench_cmd(
            ctx=cast(typer.Context, None),
            model_id=model_id,
            suite="all",
            compare="",
        )

    return _run(chain, "bench", args, _do, extra_input={"model_id": model_id})


def _align_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Wrap ``carl lab align --mode <mode>``. Requires --mode in args."""
    mode = ""
    source = ""
    model_id = ""
    quick = False
    config = "carl.yaml"
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in ("--mode", "-m") and i + 1 < len(args):
            mode = args[i + 1]
            i += 2
            continue
        if tok in ("--source", "-s") and i + 1 < len(args):
            source = args[i + 1]
            i += 2
            continue
        if tok == "--model" and i + 1 < len(args):
            model_id = args[i + 1]
            i += 2
            continue
        if tok in ("--config", "-c") and i + 1 < len(args):
            config = args[i + 1]
            i += 2
            continue
        if tok == "--quick":
            quick = True
            i += 1
            continue
        i += 1

    if not mode:
        def _missing() -> None:
            raise SystemExit(2)

        return _run(
            chain,
            "align",
            args,
            _missing,
            extra_input={"error": "align requires --mode"},
        )

    def _do() -> None:
        from carl_studio.cli import lab as _lab

        _lab.align_cmd(
            ctx=cast(typer.Context, None),
            mode=mode,
            source=source,
            model_id=model_id,
            quick=quick,
            config=config,
        )

    return _run(chain, "align", args, _do, extra_input={"mode": mode})


def _learn_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Wrap ``carl lab learn <source>``. First positional arg is source."""
    if not args:
        def _missing() -> None:
            raise SystemExit(2)

        return _run(
            chain,
            "learn",
            args,
            _missing,
            extra_input={"error": "learn requires a source argument"},
        )

    source = args[0]

    def _do() -> None:
        from carl_studio.cli import lab as _lab

        _lab.learn_cmd(
            ctx=cast(typer.Context, None),
            source=source,
            model_id="",
            depth="shallow",
            quality_threshold=0.9,
            output="",
            config="carl.yaml",
            synthesize=False,
            count=10,
            kit="",
            recipe="",
            frame="",
        )

    return _run(chain, "learn", args, _do, extra_input={"source": source[:200]})


def _eval_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Wrap ``carl eval --adapter <id>``. Parses --adapter/--dataset from args."""
    adapter = ""
    dataset = ""
    base_model = ""
    phase = "auto"
    threshold = 0.30
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in ("--adapter", "-a") and i + 1 < len(args):
            adapter = args[i + 1]
            i += 2
            continue
        if tok in ("--dataset", "-d") and i + 1 < len(args):
            dataset = args[i + 1]
            i += 2
            continue
        if tok in ("--base-model", "-b") and i + 1 < len(args):
            base_model = args[i + 1]
            i += 2
            continue
        if tok in ("--phase", "-p") and i + 1 < len(args):
            phase = args[i + 1]
            i += 2
            continue
        if tok in ("--threshold", "-t") and i + 1 < len(args):
            try:
                threshold = float(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        i += 1

    if not adapter:
        def _missing() -> None:
            raise SystemExit(2)

        return _run(
            chain,
            "eval",
            args,
            _missing,
            extra_input={"error": "eval requires --adapter"},
        )

    def _do() -> None:
        from carl_studio.cli import training as _training

        _training.eval_cmd(
            adapter=adapter,
            base_model=base_model,
            sft_adapter=None,
            dataset=dataset,
            data_files=None,
            phase=phase,
            threshold=threshold,
            max_samples=None,
            max_turns=10,
            remote=False,
            hardware="l40sx1",
            job_id=None,
            json_output=False,
        )

    return _run(chain, "eval", args, _do, extra_input={"adapter": adapter})


def _infer_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Wrap ``carl infer``. --model/--adapter parsed from args; remainder becomes prompt."""
    model = ""
    adapter = ""
    ttt = "none"
    live = False
    system_prompt = ""
    max_tokens = 2048
    prompt_tokens: list[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in ("--model", "-m") and i + 1 < len(args):
            model = args[i + 1]
            i += 2
            continue
        if tok in ("--adapter", "-a") and i + 1 < len(args):
            adapter = args[i + 1]
            i += 2
            continue
        if tok == "--ttt" and i + 1 < len(args):
            ttt = args[i + 1]
            i += 2
            continue
        if tok in ("--live", "-l"):
            live = True
            i += 1
            continue
        if tok == "--system" and i + 1 < len(args):
            system_prompt = args[i + 1]
            i += 2
            continue
        if tok == "--max-tokens" and i + 1 < len(args):
            try:
                max_tokens = int(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        prompt_tokens.append(tok)
        i += 1
    prompt = " ".join(prompt_tokens)

    def _do() -> None:
        from carl_studio.cli import infer as _infer

        _infer.infer_cmd(
            model=model,
            adapter=adapter,
            ttt=ttt,
            live=live,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            prompt=prompt,
        )

    return _run(
        chain,
        "infer",
        args,
        _do,
        extra_input={"model": model, "adapter": adapter, "prompt": prompt[:200]},
    )


def _publish_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Wrap ``carl publish <hub_id>``. First positional arg is hub_id."""
    if not args:
        def _missing() -> None:
            raise SystemExit(2)

        return _run(
            chain,
            "publish",
            args,
            _missing,
            extra_input={"error": "publish requires a hub_id argument"},
        )

    hub_id = args[0]
    item_type = "model"
    name = ""
    description = ""
    public = True
    base_model = ""
    rank = 64
    i = 1
    while i < len(args):
        tok = args[i]
        if tok in ("--type", "-t") and i + 1 < len(args):
            item_type = args[i + 1]
            i += 2
            continue
        if tok in ("--name", "-n") and i + 1 < len(args):
            name = args[i + 1]
            i += 2
            continue
        if tok in ("--description", "-d") and i + 1 < len(args):
            description = args[i + 1]
            i += 2
            continue
        if tok == "--private":
            public = False
            i += 1
            continue
        if tok == "--public":
            public = True
            i += 1
            continue
        if tok in ("--base", "-b") and i + 1 < len(args):
            base_model = args[i + 1]
            i += 2
            continue
        if tok in ("--rank", "-r") and i + 1 < len(args):
            try:
                rank = int(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        i += 1

    def _do() -> None:
        from carl_studio.cli import marketplace as _marketplace

        _marketplace.publish_cmd(
            ctx=cast(typer.Context, None),
            hub_id=hub_id,
            item_type=item_type,
            name=name,
            description=description,
            public=public,
            base_model=base_model,
            rank=rank,
        )

    return _run(chain, "publish", args, _do, extra_input={"hub_id": hub_id})


def _push_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Wrap ``carl camp sync push`` to push local runs to carl.camp."""
    types_arg = "runs"
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in ("--types", "-t") and i + 1 < len(args):
            types_arg = args[i + 1]
            i += 2
            continue
        i += 1

    def _do() -> None:
        from carl_studio.cli import platform as _platform

        _platform.sync_push_cmd(types=types_arg)

    return _run(chain, "push", args, _do, extra_input={"types": types_arg})


def _diagnose_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Wrap ``carl observe --diagnose ...``. Parses --file/--url from args."""
    url: str | None = None
    file: str | None = None
    run_name = ""
    project = ""
    space = ""
    api_key: str | None = None
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in ("--url", "-u") and i + 1 < len(args):
            url = args[i + 1]
            i += 2
            continue
        if tok in ("--file", "-f") and i + 1 < len(args):
            file = args[i + 1]
            i += 2
            continue
        if tok == "--run" and i + 1 < len(args):
            run_name = args[i + 1]
            i += 2
            continue
        if tok == "--project" and i + 1 < len(args):
            project = args[i + 1]
            i += 2
            continue
        if tok == "--space" and i + 1 < len(args):
            space = args[i + 1]
            i += 2
            continue
        if tok == "--api-key" and i + 1 < len(args):
            api_key = args[i + 1]
            i += 2
            continue
        i += 1

    def _do() -> None:
        from carl_studio.cli import observe as _observe

        _observe.observe(
            url=url,
            file=file,
            live=False,
            source="auto",
            diagnose=True,
            api_key=api_key,
            poll=None,
            project=project,
            run_name=run_name,
            space=space,
        )

    return _run(
        chain,
        "diagnose",
        args,
        _do,
        extra_input={"url": url, "file": file, "run": run_name},
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


OPERATIONS: dict[str, Operation] = {
    # lifecycle
    "doctor": _doctor_op,
    "doctor_freshness": _doctor_freshness_op,
    "start": _start_op,
    "init": _init_op,
    "freshness": _freshness_op,
    "project_status": _project_status_op,
    # agentic
    "ask": _ask_op,
    "echo": _echo_op,
    "chat": _chat_op,
    "train": _train_op,
    "review": _review_op,
    "simplify": _simplify_op,
    "ship": _ship_op,
    # workbench
    "bench": _bench_op,
    "align": _align_op,
    "learn": _learn_op,
    "eval": _eval_op,
    "infer": _infer_op,
    "publish": _publish_op,
    "push": _push_op,
    "diagnose": _diagnose_op,
}


def get_operation(name: str) -> Operation | None:
    """Return the op registered under ``name`` or None if unknown."""
    return OPERATIONS.get(name)


def list_operations() -> list[str]:
    """Return all registered op names, sorted."""
    return sorted(OPERATIONS.keys())
