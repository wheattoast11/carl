# CARL workflows

Use `carl --help` and the selected subcommand's help for the installed release.

```bash
carl init
carl observe --help
carl train --help
carl eval --help
carl lab skill list
carl lab agent harnesses
```

The lightweight base package can measure coherence without training dependencies.
Training and optional backends require their documented extras and supported
hardware. Check availability before launching a job.

Use the actual objective and held-out eval to assess training. Coherence alone
does not establish task success. Preserve inputs, configuration, result artifacts,
and the named evaluation commands requested by the user.

CARL's delegation task IDs remain stable across its MCP and A2A observation
surfaces. Resume input only through the original live execution owner.

Prepare local training without starting a job:

```bash
carl train --prepare --config path/to/training.yaml --json
```

Relative datasets resolve beside the configuration first, then at the discovered
project root. Inspect `ready` and the reported issues before executing the plan.
The first checkpoint binding reads every shard. Later preparation reuses cached
byte digests while file identity, size, mtime, and ctime match; execution still
rechecks source bytes. A checkpoint index alone does not bind shard contents.

TCG reward datasets can supply `private_keys` as a list of private draft identifiers
per sample and `trajectory` as a list of events with `phase: private` or
`phase: public`. CARL checks both the recorded public events and the completion.
A leak zeroes task and coherence contributions and adds an unweighted terminal
reward of `-3.0`, outside cascade masks. Evaluation through `EvalConfig.private_keys`
checks public events in `EvalReport.detail` and refuses a leaking report.
Keep private draft identifiers and contents out of public logs and result artifacts.
