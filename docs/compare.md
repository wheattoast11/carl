# carl-studio vs. the field

Honest feature matrix as of v0.4.0. carl-studio is the **agentic workbench
hub**; the other systems are **spokes** (and we adapt to them via
`carl_studio.adapters`). The axes below are what matters for day-to-day
research — not raw training speed, which is TRL's job and we happily pass
through to it.

## Matrix

| Feature                                  | carl-studio | TRL | Unsloth | Axolotl | Tinker | Atropos |
|------------------------------------------|:-----------:|:---:|:-------:|:-------:|:------:|:-------:|
| Agentic chat with tool use + sessions    | ✓           | —   | —       | —       | —      | —       |
| Typed error taxonomy (stable codes)      | ✓           | —   | —       | —       | —      | —       |
| InteractionChain trace (typed, JSONL)    | ✓           | —   | —       | —       | —      | —       |
| Coherence-aware rewards (φ, κ, σ)        | ✓           | —   | —       | —       | —      | —       |
| Hypothesize → Eval → Infer → Commit loop | ✓           | —   | —       | —       | —      | —       |
| Constitutional / layered memory          | ✓           | —   | —       | —       | —      | —       |
| Wallet-encrypted secrets at rest         | ✓           | —   | —       | —       | —      | —       |
| x402 payment rail client                 | ✓           | —   | —       | —       | —      | —       |
| Credit + consent primitives              | ✓           | —   | —       | —       | —      | —       |
| Marketplace idempotent publish           | ✓           | —   | —       | —       | —      | —       |
| Circuit-breaker retries                  | ✓           | —   | —       | —       | —      | —       |
| Path-sandboxed eval runner               | ✓           | —   | —       | —       | —      | —       |
| GRPO / PPO / DPO trainer                 | via TRL     | ✓   | ✓       | ✓       | ✓      | ✓       |
| 4/8-bit QLoRA                            | via spokes  | via peft | ✓  | ✓       | ✓      | —       |
| 2× faster LoRA fine-tune                 | via spoke   | —   | ✓       | —       | —      | —       |
| Multi-agent RL environment               | via spoke   | —   | —       | —       | —      | ✓       |
| Managed training service                 | via spoke   | —   | —       | —       | ✓      | —       |
| Config-driven YAML training              | ✓           | —   | —       | ✓       | —      | —       |

## The hub model

```
             ┌──────────────────────────┐
             │     carl chat/agent      │
             │   (hypothesize/commit)   │
             └────────────┬─────────────┘
                          │
                  carl_core primitives
                          │
                ┌─────────┴─────────┐
                │                   │
        UnifiedBackend      InteractionChain
                │                   │
     ┌────┬─────┼─────┬─────┐       │
     │    │     │     │     │       │
    TRL Unsloth Axol Tinker Atropos │
     │    │     │     │     │       │
     └────┴─────┼─────┴─────┘       │
                │                   │
           [Coherence gate + rewards applied
            on outputs regardless of backend]
                │
            Eval / Push / Publish
```

Carl picks the right backend for the experiment. Unsloth when you need fast
LoRA on a single GPU. Axolotl when the config surface suits you. TRL when
you need the library primitives. Tinker when you want a managed service.
Atropos when the experiment is multi-agent RL. carl-studio is the layer that
applies **coherence gating**, **typed metrics**, and **constitutional memory**
on top of whichever backend ran the actual training.

## Why the spoke model is the right abstraction

1. Training libraries churn fast. Locking into one is a tax.
2. The interesting stuff is agentic orchestration + memory, which none of
   the spokes provide. That's our MOAT.
3. Users can incrementally adopt carl without giving up the trainer they
   already know.
4. Each spoke gets the same experimental frame: `carl hypothesize` → run on
   spoke of choice → `carl infer` → `carl commit`.

## When to use what

- **Just want faster LoRA on a laptop?** `carl train --backend unsloth`.
- **Want config-file-driven experimentation?** `carl train --backend axolotl`.
- **Want managed post-training?** `carl train --backend tinker`.
- **Multi-agent RL?** `carl train --backend atropos`.
- **Coherence-aware GRPO with our reward stack?** `carl train --backend trl` (default).

In all cases the **outer loop** — hypothesize, gate on eval, infer next step,
commit durable learnings — is the same. That uniformity is what makes
carl-studio the hub.
