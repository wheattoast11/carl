# CLI

Modular Typer-based CLI package for the `carl` command. Domain logic lives in `carl_studio.*`; this package handles command registration, argument parsing, and output formatting.

## Key Abstractions

- `app` -- Root Typer application (`carl`). Defined in `apps.py`.
- `camp_app` -- Sub-app for platform commands (`carl camp`): account, sync, credits, marketplace.
- `lab_app` -- Sub-app for experimental commands (`carl lab`): admin, align, bench, chat, mcp, golf, paper.
- `wiring.py` -- Alias registration and optional sub-app mounting. When an optional module cannot be imported, a stub command is registered with an install hint instead of silently hiding the feature.

## Module Map

| Module | Commands |
|--------|----------|
| `core.py` | start, doctor, version |
| `training.py` | train, run |
| `observe.py` | observe, eval |
| `project_data.py` | project, data |
| `chat.py` | chat REPL |
| `billing.py` | billing surfaces |
| `marketplace.py` | marketplace browse/install |
| `consent.py` | consent management |
| `frame.py` | WorkFrame commands |
| `contract.py` | service contract witnessing |
| `carlito.py` | small agent management |
| `curriculum.py` | curriculum commands |

## Architecture

The `__init__.py` eagerly imports core modules to register commands, then `wiring.py` mounts optional sub-apps with fallback stubs. All domain imports are lazy (inside function bodies) to keep `carl --help` fast.

## Usage

```
carl start                  # interactive project setup
carl train config.yaml      # submit training job
carl observe <run-id>       # live coherence monitoring
carl chat                   # agentic conversation
```
