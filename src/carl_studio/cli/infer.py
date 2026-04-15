"""Inference CLI -- ``carl infer``.

Run inference with a CARL-trained model, optionally with TTT (test-time training).
Register in cli.py via: ``app.command('infer')(infer_cmd)``
"""
from __future__ import annotations

from typing import Any

import typer

from carl_studio.console import get_console


def _resolve_model_adapter(
    model_arg: str, adapter_arg: str
) -> tuple[str, str]:
    """Resolve model + adapter from CLI args or carl.yaml project config.

    Returns (model_id, adapter_id). Either may be empty if unresolvable.
    """
    model = model_arg
    adapter = adapter_arg

    if not model or not adapter:
        try:
            from carl_studio.settings import CARLSettings

            settings = CARLSettings.load()
            if not model:
                model = settings.default_model
            if not adapter and settings.hub_namespace:
                # Convention: latest adapter = namespace/il-terminals-carl-*
                # But we don't guess -- user must specify or set in carl.yaml
                pass
        except Exception:
            pass

    return model, adapter


def _check_ttt_availability(ttt_mode: str) -> tuple[bool, str]:
    """Check if TTT mode is available.

    Returns (available, message).
    """
    if ttt_mode == "none":
        return True, ""

    # TTT requires PAID tier
    try:
        from carl_studio.tier import check_tier

        allowed, effective, required = check_tier("observe.live")
        if not allowed:
            return False, (
                f"TTT requires CARL Paid tier. Current: {effective.value}. "
                "Upgrade at https://carl.camp/pricing"
            )
    except Exception:
        pass

    # SLOT specifically needs admin or terminals-runtime
    if ttt_mode in ("slot", "all"):
        try:
            from carl_studio.admin import is_admin

            if not is_admin():
                # Check if terminals-runtime is available
                try:
                    import terminals_runtime  # noqa: F401
                except ImportError:
                    return False, (
                        "SLOT requires terminals-runtime or admin unlock. "
                        "Install via: pip install terminals-runtime "
                        "or run: carl admin unlock"
                    )
        except ImportError:
            return False, (
                "SLOT requires terminals-runtime or admin unlock. "
                "Install via: pip install terminals-runtime "
                "or run: carl admin unlock"
            )

    return True, ""


def _load_model_for_inference(
    model_id: str, adapter_id: str
) -> tuple[Any, Any]:
    """Load model + tokenizer with optional LoRA adapter merge.

    Returns (model, tokenizer). Raises ImportError if torch/transformers unavailable.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "Inference requires torch + transformers. "
            "Install via: pip install carl-studio[training]"
        )

    c = get_console()
    c.info(f"Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
    )

    if adapter_id:
        c.info(f"Merging adapter: {adapter_id}")
        try:
            from peft import PeftModel  # type: ignore[import-untyped]

            model = PeftModel.from_pretrained(model, adapter_id)
            model = model.merge_and_unload()
            c.ok("Adapter merged")
        except ImportError:
            c.warn("peft not installed -- skipping adapter merge")
        except Exception as exc:
            c.error(f"Adapter merge failed: {exc}")

    return model, tokenizer


def _generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
) -> str:
    """Run a single generation pass."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback for models without chat template
        text = prompt

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    try:
        import torch  # type: ignore[import-untyped]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
    except ImportError:
        raise ImportError("torch is required for inference")

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _run_repl(
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    max_tokens: int,
) -> None:
    """Interactive REPL loop."""
    c = get_console()
    c.rule("CARL Inference REPL")
    c.info("Type 'exit' or 'quit' to leave. Ctrl+C also works.")
    c.blank()

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            c.blank()
            c.info("Exiting REPL")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            c.info("Exiting REPL")
            break

        try:
            response = _generate(model, tokenizer, user_input, system_prompt, max_tokens)
            c.blank()
            c.print(response)
            c.blank()
        except Exception as exc:
            c.error(f"Generation failed: {exc}")


def infer_cmd(
    model: str = typer.Option("", "--model", "-m", help="Base model ID"),
    adapter: str = typer.Option("", "--adapter", "-a", help="LoRA adapter repo ID"),
    ttt: str = typer.Option(
        "none", "--ttt", help="TTT mode: none, slot, lora, all"
    ),
    live: bool = typer.Option(False, "--live", "-l", help="Interactive mode (REPL)"),
    system_prompt: str = typer.Option("", "--system", help="System prompt override"),
    max_tokens: int = typer.Option(2048, "--max-tokens"),
    prompt: str = typer.Argument("", help="Input prompt (or empty for interactive)"),
) -> None:
    """Run inference with a CARL-trained model, optionally with TTT.

    Examples:
      carl infer --model Tesslate/OmniCoder-9B --adapter wheattoast11/il-terminals-carl-omni9b-v12
      carl infer --adapter wheattoast11/il-terminals-carl-omni9b-v12 --ttt slot --live
    """
    c = get_console()
    c.header("CARL Infer")

    # Validate TTT mode
    valid_ttt_modes = ("none", "slot", "lora", "all")
    if ttt not in valid_ttt_modes:
        c.error(f"Invalid --ttt mode: {ttt!r}. Must be one of: {', '.join(valid_ttt_modes)}")
        raise typer.Exit(code=1)

    # Resolve model + adapter
    resolved_model, resolved_adapter = _resolve_model_adapter(model, adapter)

    if not resolved_model:
        c.error(
            "No model specified. Provide --model or set default_model in carl.yaml"
        )
        raise typer.Exit(code=1)

    c.kv("Model", resolved_model)
    if resolved_adapter:
        c.kv("Adapter", resolved_adapter)
    c.kv("TTT", ttt)
    c.kv("Mode", "interactive" if live else "single-shot")
    c.blank()

    # Check TTT availability
    if ttt != "none":
        available, msg = _check_ttt_availability(ttt)
        if not available:
            c.warn(msg)
            c.info("Falling back to standard inference (no TTT)")
            c.info(
                "TTT requires GPU + terminals-runtime or admin unlock. "
                "See: https://carl.camp/docs/ttt"
            )
            ttt = "none"
        else:
            c.info(
                "TTT active -- per-sample optimization will run during inference. "
                "This requires GPU."
            )

    # Load model
    try:
        loaded_model, tokenizer = _load_model_for_inference(
            resolved_model, resolved_adapter
        )
    except ImportError as exc:
        c.error(str(exc))
        c.info("Install inference deps: pip install carl-studio[training]")
        raise typer.Exit(code=1)
    except Exception as exc:
        c.error(f"Failed to load model: {exc}")
        raise typer.Exit(code=1)

    # Determine mode
    if live or not prompt:
        _run_repl(loaded_model, tokenizer, system_prompt, max_tokens)
    else:
        try:
            response = _generate(
                loaded_model, tokenizer, prompt, system_prompt, max_tokens
            )
            c.print(response)
        except Exception as exc:
            c.error(f"Generation failed: {exc}")
            raise typer.Exit(code=1)
