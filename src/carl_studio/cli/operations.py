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
    OPERATION_DESCRIPTIONS["simplify"] = "one-line help text"
"""
from __future__ import annotations

from time import monotonic
from typing import Any, Callable, cast

import typer

from carl_core.interaction import ActionType, InteractionChain

Operation = Callable[[InteractionChain, list[str]], InteractionChain]

__all__ = [
    "OPERATIONS",
    "OPERATION_DESCRIPTIONS",
    "Operation",
    "get_operation",
    "get_description",
    "list_operations",
    "parse_flags",
]


# ---------------------------------------------------------------------------
# Shared flag parser — single source of truth for ``--flag value`` /
# ``--flag=value`` / ``-x`` short-flag parsing across ops.
# ---------------------------------------------------------------------------


def parse_flags(
    args: list[str],
    spec: dict[str, tuple[type, Any]],
    *,
    aliases: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Parse ``--flag value`` / ``--flag=value`` / ``-x`` style args.

    ``spec`` maps canonical flag names to ``(type, default)``. If ``type`` is
    ``bool`` the flag is treated as a boolean switch — presence sets it to
    ``True``, absence leaves the default. Any other type is called on the
    value string (e.g. ``int("5")``, ``float("0.3")``).

    ``aliases`` maps short or alternate names to canonical names (e.g.
    ``{"m": "mode", "cfg": "config"}``). Both ``-m value`` and ``--m value``
    resolve to ``mode``.

    Tokens not recognised as flags (or their values) are returned verbatim
    in ``remaining``, preserving their order. An unknown ``--foo`` token is
    passed through as positional so callers stay lenient with extras.

    Raises ``ValueError`` if a non-bool flag is provided without a value.
    """
    aliases = aliases or {}
    parsed: dict[str, Any] = {name: default for name, (_, default) in spec.items()}
    remaining: list[str] = []

    def _coerce(typ: type, raw: str) -> Any:
        if typ is str:
            return raw
        if typ is bool:
            # Accept common truthy spellings for explicit ``--flag=value`` form.
            return raw.lower() not in ("0", "false", "no", "off", "")
        try:
            return typ(raw)
        except (TypeError, ValueError):
            # Preserve the user's raw value rather than silently coercing to
            # the default — callers can decide whether to validate further.
            return raw

    i = 0
    while i < len(args):
        token = args[i]
        if not token.startswith("-") or token == "-" or token == "--":
            remaining.append(token)
            i += 1
            continue

        # Split ``--flag=value`` into key + inline value.
        key, sep, inline_val = token.partition("=")
        name = key.lstrip("-")
        canonical = aliases.get(name, name)

        if canonical not in spec:
            # Unknown flag — pass through as a positional token. If the next
            # token is clearly its value, pass that through too so we don't
            # mis-attribute it to a later canonical flag.
            remaining.append(token)
            i += 1
            continue

        typ, _ = spec[canonical]
        if typ is bool:
            if sep == "=":
                parsed[canonical] = _coerce(bool, inline_val)
            else:
                parsed[canonical] = True
            i += 1
            continue

        if sep == "=":
            parsed[canonical] = _coerce(typ, inline_val)
            i += 1
            continue

        if i + 1 >= len(args):
            raise ValueError(f"flag --{name} requires a value")
        parsed[canonical] = _coerce(typ, args[i + 1])
        i += 2

    return parsed, remaining


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


def _missing_arg(chain: InteractionChain, op_name: str, args: list[str], error: str) -> InteractionChain:
    """Record a ``SystemExit(2)`` failure for an op that's missing a required arg."""

    def _missing() -> None:
        raise SystemExit(2)

    return _run(chain, op_name, args, _missing, extra_input={"error": error})


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
    flags, _ = parse_flags(
        args,
        {"config": (str, "carl.yaml")},
        aliases={"c": "config"},
    )
    config_path = cast(str, flags["config"])

    def _do() -> None:
        from carl_studio.cli.project_data import project_show

        project_show(config=config_path)

    return _run(chain, "project_status", args, _do)


# ---------------------------------------------------------------------------
# Ops: agentic assistance (prompt-template macros live in _PROMPT_OPS below)
# ---------------------------------------------------------------------------


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


# Prompt-template macros. Each op takes the joined positional args and
# formats them into the template as ``{args}``. The registered callable
# passes the final prompt string to ``run_one_shot_agent``.
#
# Templates use a simple convention: ``{args}`` is substituted with the
# trimmed positional tokens joined by spaces. If the user invokes an op
# with no args, the template is still formatted with an empty string and
# a default-prompt fallback applies inside each wrapper (see ``_prompt_op``).
_PROMPT_OPS: dict[str, tuple[str, str, str]] = {
    # name: (template, description, empty_args_fallback)
    "ask": (
        "{args}",
        "Ask Carl a single-shot question with tool access.",
        "hello",
    ),
    "chat": (
        "Open a working session about: {args}",
        "Start an interactive chat session (args become the first turn).",
        "",
    ),
    "train": (
        "Plan and execute a training run for: {args}",
        "Hand the chain off to the agent as a training instruction.",
        "Help me set up a training run for this project.",
    ),
    "review": (
        "Review {args}. Look for bugs, bad patterns, and things that would confuse "
        "a new reader. Keep it under 200 words.",
        "Ask Carl for a code/config review on the given target.",
        "the most recent change",
    ),
    "simplify": (
        "Simplify {args}. Remove dead code, redundant abstractions, and anything that "
        "doesn't pull weight. Preserve behavior. Report each simplification before applying.",
        "Ask Carl to simplify the target file or module.",
        "the current file",
    ),
    "ship": (
        "Ship this: run tests, run lint, run type checks, fix any failures, then stage "
        "and commit the fix. Report what was green, what was red, and what you changed."
        "{args}",
        "Run the full quality pipeline: tests, lint, type, bundle, commit.",
        "",
    ),
}


def _prompt_op(name: str, template: str, fallback: str) -> Operation:
    """Build an op that formats positional args into ``template`` and hands it
    to ``run_one_shot_agent``.

    ``fallback`` is used when no positional args are given. For ``chat`` this
    is the empty string, which routes the op into the interactive chat REPL
    instead of a one-shot agent call — preserving the original behavior.
    """

    def _inner(chain: InteractionChain, args: list[str]) -> InteractionChain:
        joined = " ".join(args).strip()
        # Special-case chat: no args → open the interactive REPL; args → one-shot.
        if name == "chat" and not joined:
            def _do_chat() -> None:
                from carl_studio.cli.chat import chat_cmd

                chat_cmd()

            return _run(chain, "chat", args, _do_chat)

        effective = joined if joined else fallback
        # ``ship`` appends extra instructions only when the user supplied them.
        if name == "ship":
            prompt = template.format(args=f"\n\nExtra instructions: {effective}" if joined else "")
        else:
            prompt = template.format(args=effective)

        def _do() -> None:
            from carl_studio.cli.chat import run_one_shot_agent

            run_one_shot_agent(prompt)

        # Preserve the original per-op ``extra_input`` shape so any trace
        # consumers keep working.
        extra: dict[str, Any] = {"prompt": prompt[:200]}
        if name == "chat" and joined:
            extra = {"first_turn": joined[:200]}
        elif name == "review" or name == "simplify":
            extra = {"target": effective}
        elif name == "ship":
            extra = {"extra": joined[:200]}
        return _run(chain, name, args, _do, extra_input=extra)

    _inner.__name__ = f"_{name}_op"
    _inner.__doc__ = f"{name} — templated prompt ({template[:40]}...)"
    return _inner


# ---------------------------------------------------------------------------
# Ops: workbench commands (bench / align / learn / eval / infer / publish / push)
# ---------------------------------------------------------------------------


def _bench_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Wrap ``carl lab bench <model_id>``. First positional arg is model ID."""
    if not args:
        return _missing_arg(chain, "bench", args, "bench requires a model_id argument")

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
    flags, _ = parse_flags(
        args,
        {
            "mode": (str, ""),
            "source": (str, ""),
            "model": (str, ""),
            "config": (str, "carl.yaml"),
            "quick": (bool, False),
        },
        aliases={"m": "mode", "s": "source", "c": "config"},
    )
    mode = cast(str, flags["mode"])
    if not mode:
        return _missing_arg(chain, "align", args, "align requires --mode")

    def _do() -> None:
        from carl_studio.cli import lab as _lab

        _lab.align_cmd(
            ctx=cast(typer.Context, None),
            mode=mode,
            source=cast(str, flags["source"]),
            model_id=cast(str, flags["model"]),
            quick=cast(bool, flags["quick"]),
            config=cast(str, flags["config"]),
        )

    return _run(chain, "align", args, _do, extra_input={"mode": mode})


def _learn_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Wrap ``carl lab learn <source>``. First positional arg is source."""
    if not args:
        return _missing_arg(chain, "learn", args, "learn requires a source argument")

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
    flags, _ = parse_flags(
        args,
        {
            "adapter": (str, ""),
            "dataset": (str, ""),
            "base-model": (str, ""),
            "phase": (str, "auto"),
            "threshold": (float, 0.30),
        },
        aliases={"a": "adapter", "d": "dataset", "b": "base-model", "p": "phase", "t": "threshold"},
    )
    adapter = cast(str, flags["adapter"])
    if not adapter:
        return _missing_arg(chain, "eval", args, "eval requires --adapter")

    def _do() -> None:
        from carl_studio.cli import training as _training

        _training.eval_cmd(
            adapter=adapter,
            base_model=cast(str, flags["base-model"]),
            sft_adapter=None,
            dataset=cast(str, flags["dataset"]),
            data_files=None,
            phase=cast(str, flags["phase"]),
            threshold=cast(float, flags["threshold"]),
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
    flags, remaining = parse_flags(
        args,
        {
            "model": (str, ""),
            "adapter": (str, ""),
            "ttt": (str, "none"),
            "live": (bool, False),
            "system": (str, ""),
            "max-tokens": (int, 2048),
        },
        aliases={"m": "model", "a": "adapter", "l": "live"},
    )
    prompt = " ".join(remaining)
    model = cast(str, flags["model"])
    adapter = cast(str, flags["adapter"])

    def _do() -> None:
        from carl_studio.cli import infer as _infer

        _infer.infer_cmd(
            model=model,
            adapter=adapter,
            ttt=cast(str, flags["ttt"]),
            live=cast(bool, flags["live"]),
            system_prompt=cast(str, flags["system"]),
            max_tokens=cast(int, flags["max-tokens"]),
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
        return _missing_arg(chain, "publish", args, "publish requires a hub_id argument")

    hub_id = args[0]
    flags, _ = parse_flags(
        args[1:],
        {
            "type": (str, "model"),
            "name": (str, ""),
            "description": (str, ""),
            "private": (bool, False),
            "public": (bool, False),
            "base": (str, ""),
            "rank": (int, 64),
        },
        aliases={"t": "type", "n": "name", "d": "description", "b": "base", "r": "rank"},
    )
    # Public is the default; --private wins if set; --public explicitly sets True.
    public = True
    if cast(bool, flags["private"]):
        public = False
    elif cast(bool, flags["public"]):
        public = True

    def _do() -> None:
        from carl_studio.cli import marketplace as _marketplace

        _marketplace.publish_cmd(
            ctx=cast(typer.Context, None),
            hub_id=hub_id,
            item_type=cast(str, flags["type"]),
            name=cast(str, flags["name"]),
            description=cast(str, flags["description"]),
            public=public,
            base_model=cast(str, flags["base"]),
            rank=cast(int, flags["rank"]),
        )

    return _run(chain, "publish", args, _do, extra_input={"hub_id": hub_id})


def _push_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Wrap ``carl camp sync push`` to push local runs to carl.camp."""
    flags, _ = parse_flags(
        args,
        {"types": (str, "runs")},
        aliases={"t": "types"},
    )
    types_arg = cast(str, flags["types"])

    def _do() -> None:
        from carl_studio.cli import platform as _platform

        _platform.sync_push_cmd(types=types_arg)

    return _run(chain, "push", args, _do, extra_input={"types": types_arg})


def _hypothesize_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Wrap ``carl hypothesize <statement>``. Parses -o/--output and flags from args."""
    flags, remaining = parse_flags(
        args,
        {
            "output": (str, "carl.yaml"),
            "model": (str, ""),
            "dry-run": (bool, False),
            "force": (bool, False),
        },
        aliases={"o": "output", "m": "model"},
    )
    statement = " ".join(remaining).strip()
    if not statement:
        return _missing_arg(chain, "hypothesize", args, "hypothesize requires a statement")

    output = cast(str, flags["output"])

    def _do() -> None:
        from carl_studio.cli.hypothesize import hypothesize_cmd

        hypothesize_cmd(
            statement=statement,
            output=output,
            model=cast(str, flags["model"]),
            dry_run=cast(bool, flags["dry-run"]),
            force=cast(bool, flags["force"]),
        )

    return _run(
        chain,
        "hypothesize",
        args,
        _do,
        extra_input={"statement": statement[:200], "output": output},
    )


def _commit_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Wrap ``carl commit <learning>`` with --id/--priority/--tags/--from-session/--dry-run."""
    flags, remaining = parse_flags(
        args,
        {
            "id": (str, ""),
            "priority": (int, 60),
            "tags": (str, ""),
            "from-session": (str, ""),
            "dry-run": (bool, False),
        },
        aliases={"p": "priority"},
    )
    learning = " ".join(remaining).strip()
    from_session = cast(str, flags["from-session"])

    def _do() -> None:
        from carl_studio.cli.commit import commit_cmd

        # ``commit_cmd`` declares ``learning: str`` and normalises via
        # ``(learning or "").strip()``. Passing an empty string is
        # equivalent to the legacy ``None`` and keeps pyright happy.
        commit_cmd(
            learning=learning,
            rule_id=cast(str, flags["id"]),
            priority=cast(int, flags["priority"]),
            tags=cast(str, flags["tags"]),
            from_session=from_session,
            dry_run=cast(bool, flags["dry-run"]),
        )

    return _run(
        chain,
        "commit",
        args,
        _do,
        extra_input={
            "learning": learning[:200],
            "from_session": from_session,
        },
    )


# ---------------------------------------------------------------------------
# Ops: environments hub (pull-env / publish-env)
# ---------------------------------------------------------------------------


def _pull_env_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Wrap ``carl_studio.environments.registry.from_hub``.

    Usage: ``/pull-env <hub-name> [--revision <rev>]``. Pulls the remote
    env snapshot, dynamically imports it, and registers it in the local
    registry so ``get_environment`` can see it.
    """
    if not args:
        return _missing_arg(chain, "pull-env", args, "pull-env requires a hub-name argument")

    hub_name = args[0]
    flags, _ = parse_flags(
        args[1:],
        {"revision": (str, "")},
        aliases={"r": "revision"},
    )
    revision_raw = cast(str, flags["revision"])
    revision: str | None = revision_raw or None

    def _do() -> dict[str, Any]:
        from carl_studio.environments.registry import from_hub

        cls = from_hub(hub_name, revision=revision)
        spec = cls.spec
        return {
            "name": spec.name,
            "lane": spec.lane.value,
            "tools": list(spec.tools),
            "class": cls.__name__,
        }

    return _run(
        chain,
        "pull-env",
        args,
        _do,
        extra_input={"hub_name": hub_name, "revision": revision},
    )


def _publish_env_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Wrap ``carl_studio.environments.registry.publish_to_hub``.

    Usage: ``/publish-env <local-name> <repo-id> [--private]``. The local
    env must already be registered (via ``@register_environment`` or
    ``/pull-env``).
    """
    if len(args) < 2:
        return _missing_arg(
            chain,
            "publish-env",
            args,
            "publish-env requires <local-name> <repo-id>",
        )

    local_name = args[0]
    repo_id = args[1]
    flags, _ = parse_flags(
        args[2:],
        {
            "private": (bool, False),
            "public": (bool, False),
        },
    )
    private = cast(bool, flags["private"])
    if cast(bool, flags["public"]):
        private = False

    def _do() -> dict[str, Any]:
        from carl_studio.environments.registry import get_environment, publish_to_hub

        env_cls = get_environment(local_name)
        commit_url = publish_to_hub(env_cls, repo_id, private=private)
        return {
            "name": local_name,
            "repo_id": repo_id,
            "private": private,
            "commit_url": commit_url,
        }

    return _run(
        chain,
        "publish-env",
        args,
        _do,
        extra_input={
            "local_name": local_name,
            "repo_id": repo_id,
            "private": private,
        },
    )


def _diagnose_op(chain: InteractionChain, args: list[str]) -> InteractionChain:
    """Wrap ``carl observe --diagnose ...``. Parses --file/--url from args."""
    flags, _ = parse_flags(
        args,
        {
            "url": (str, ""),
            "file": (str, ""),
            "run": (str, ""),
            "project": (str, ""),
            "space": (str, ""),
            "api-key": (str, ""),
        },
        aliases={"u": "url", "f": "file"},
    )
    url = cast(str, flags["url"]) or None
    file = cast(str, flags["file"]) or None
    api_key_raw = cast(str, flags["api-key"])
    api_key: str | None = api_key_raw or None
    run_name = cast(str, flags["run"])
    project = cast(str, flags["project"])
    space = cast(str, flags["space"])

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
# Registry — callable-only mapping, with descriptions in a parallel dict so
# the existing ``OPERATIONS[name](chain, args)`` contract is preserved for
# every downstream caller (tests, chat_agent, etc.).
# ---------------------------------------------------------------------------


OPERATIONS: dict[str, Operation] = {
    # lifecycle
    "doctor": _doctor_op,
    "doctor_freshness": _doctor_freshness_op,
    "start": _start_op,
    "init": _init_op,
    "freshness": _freshness_op,
    "project_status": _project_status_op,
    # agentic — core non-template ops
    "echo": _echo_op,
    # workbench
    "bench": _bench_op,
    "align": _align_op,
    "learn": _learn_op,
    "eval": _eval_op,
    "infer": _infer_op,
    "publish": _publish_op,
    "push": _push_op,
    "diagnose": _diagnose_op,
    # environments hub
    "pull-env": _pull_env_op,
    "publish-env": _publish_env_op,
    # research cycle
    "hypothesize": _hypothesize_op,
    "commit": _commit_op,
}


OPERATION_DESCRIPTIONS: dict[str, str] = {
    # lifecycle
    "doctor": "Run readiness checks (binary + config + toolchain).",
    "doctor_freshness": "Run doctor plus a full freshness audit of packages and tokens.",
    "start": "Print a quick start summary (profile, entrypoints, next steps).",
    "init": "One-shot wizard: configure ~/.carl, install extras, write carl.yaml.",
    "freshness": "Audit package/token freshness and stash the summary on the chain.",
    "project_status": "Show the active project config via carl project show.",
    # agentic
    "echo": "Record a pass-through step (useful for debugging chains).",
    # workbench
    "bench": "Benchmark a model by id (carl lab bench <model_id>).",
    "align": "Run alignment with --mode (carl lab align).",
    "learn": "Ingest a learning source (carl lab learn <source>).",
    "eval": "Evaluate an adapter (carl eval --adapter <id>).",
    "infer": "Run inference on a model/adapter with prompt tail.",
    "publish": "Publish a model or adapter to the hub (carl publish).",
    "push": "Push local runs/metrics to carl.camp (carl camp sync push).",
    "diagnose": "Diagnose a training run from file/url (carl observe --diagnose).",
    # environments hub
    "pull-env": "Pull a remote environment snapshot from the hub and register it.",
    "publish-env": "Publish a locally-registered environment to the hub.",
    # research cycle
    "hypothesize": "Propose a carl.yaml edit from a one-line hypothesis statement.",
    "commit": "Commit a learning as a durable rule (carl commit <learning>).",
}


# Register prompt-template ops. Each entry produces both a callable and a
# matching description — keeps the two in lockstep and the registry DRY.
for _name, (_tmpl, _desc, _fallback) in _PROMPT_OPS.items():
    OPERATIONS[_name] = _prompt_op(_name, _tmpl, _fallback)
    OPERATION_DESCRIPTIONS[_name] = _desc


def get_operation(name: str) -> Operation | None:
    """Return the op registered under ``name`` or None if unknown."""
    return OPERATIONS.get(name)


def get_description(name: str) -> str:
    """Return the one-line description for ``name``.

    Falls back to an empty string if no description is registered — callers
    should render that as "(no description)" or similar.
    """
    return OPERATION_DESCRIPTIONS.get(name, "")


def list_operations() -> list[str]:
    """Return all registered op names, sorted."""
    return sorted(OPERATIONS.keys())
