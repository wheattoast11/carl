"""Isolated offline encoder worker. Heavy dependencies stay in this interpreter."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any


def metadata() -> dict[str, Any]:
    dependencies = {}
    for name in ("torch", "transformers", "sentence-transformers", "peft"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "unavailable"
    dependencies["carl.encoder.environment"] = str(Path(sys.prefix).resolve())
    dependencies["carl.encoder.worker_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    dependencies["carl.encoder.processor_code_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    dependencies["carl.encoder.artifacts_sha256"] = hashlib.sha256(
        Path(__file__).with_name("artifacts.py").read_bytes()
    ).hexdigest()
    interpreter = Path(sys.executable).resolve()
    return {
        "interpreter": str(interpreter),
        "interpreter_sha256": hashlib.sha256(interpreter.read_bytes()).hexdigest(),
        "dependencies": dependencies,
    }


def load_model(path: str, device: str = "cpu", modalities: set[str] | None = None) -> Any:
    torch: Any = importlib.import_module("torch")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    SentenceTransformer: Any = importlib.import_module("sentence_transformers").SentenceTransformer

    selected = modalities or {"text"}
    config: dict[str, Any] = {}
    if not selected & {"image", "video"}:
        config["vision_config"] = None
    if "audio" not in selected:
        config["audio_config"] = None
    return SentenceTransformer(
        path,
        config_kwargs=config,
        local_files_only=True,
        device=device,
        model_kwargs={"torch_dtype": torch.float32 if device == "cpu" else torch.bfloat16},
    )


def features_for(model: Any, request: dict[str, Any]) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for part in request["parts"]:
        kind = part["modality"]
        if kind in {"text", "structured"}:
            content.append({"type": "text", "text": part["text"]})
        else:
            if part.get("start_s") is not None and not part.get("segment_applied"):
                raise ValueError("Provide an explicitly segmented source artifact")
            content.append({"type": kind, kind: part["path"]})
    recipe = request["recipe"]
    if recipe not in model.prompts:
        raise ValueError("Unknown model recipe")
    features = model.preprocess(
        [[{"role": "user", "content": content}]],
        prompt=model.prompts[recipe]
        if any(part["modality"] in {"text", "structured"} for part in request["parts"])
        else None,
        processing_kwargs={
            "text": {"truncation": False},
            "audio": {"sampling_rate": 16000},
            "video": {"fps": 1},
        },
    )
    if features["input_ids"].shape[-1] > request["max_tokens"]:
        raise ValueError("Combined processor token budget exceeded")
    return {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in features.items()
    }


def raw_forward(model: Any, features: dict[str, Any]) -> Any:
    return model[1](model[0](features))["sentence_embedding"]


def sibling(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def batch_features(model: Any, requests: list[dict[str, Any]]) -> dict[str, Any]:
    """Process one text recipe together without truncation or padding miscounts."""
    recipes = {r["recipe"] for r in requests}
    if len(recipes) != 1 or next(iter(recipes)) not in model.prompts:
        raise ValueError("A text batch requires one known recipe")
    if any(p["modality"] not in {"text", "structured"} for r in requests for p in r["parts"]):
        raise ValueError("Only text and structured inputs support batching")
    features = model.preprocess(
        [
            [{"role": "user", "content": [{"type": "text", "text": p["text"]} for p in r["parts"]]}]
            for r in requests
        ],
        prompt=model.prompts[next(iter(recipes))],
        processing_kwargs={"text": {"truncation": False, "padding": True}},
    )
    mask = features.get("attention_mask")
    if mask is None or len(mask.shape) != 2 or mask.shape[0] != len(requests):
        raise ValueError("Unsupported processor batch shape")
    for index, request in enumerate(requests):
        if int(mask[index].sum()) > request["max_tokens"]:
            raise ValueError("Combined processor token budget exceeded")
    return {k: v.to(model.device) if hasattr(v, "to") else v for k, v in features.items()}


def cache_carriers(request: dict[str, Any]) -> dict[str, Any]:
    """Materialize raw carriers once into the existing semantic artifact directory."""
    import resource

    artifacts = sibling("artifacts")
    wanted = artifacts.inputs(request)
    binding = request["cache_binding"]
    manifest = Path(request["manifest"])
    document: dict[str, Any] = {
        "schema": artifacts.SCHEMA,
        "binding": binding,
        "population": sorted(wanted),
        "entries": {},
    }
    if manifest.exists():
        previous = json.loads(manifest.read_bytes())
        if previous.get("population") != sorted(wanted):
            raise ValueError("A new cache population requires a successor manifest")
        _, document = artifacts.load_carriers(manifest, binding, set(previous["entries"]))
    pending = [(k, v) for k, v in wanted.items() if k not in document["entries"]]
    hits = len(wanted) - len(pending)
    forwards = 0
    tokens = 0
    started = request.get("_started", time.monotonic())
    settings = request["settings"]

    def limit() -> None:
        if time.monotonic() - started >= settings["runtime_s"]:
            raise TimeoutError("Encoder cache runtime exceeded")
        if (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            > settings["memory_gib"] * 1024**3
        ):
            raise TimeoutError("Encoder cache memory exceeded")

    limit()
    if pending:
        limit()
        torch: Any = importlib.import_module("torch")
        modalities = {p["modality"] for _, v in pending for p in v["parts"]}
        model = load_model(request["model"], binding["execution"]["device"], modalities)
        model.eval()
        buckets: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for key, value in pending:
            buckets.setdefault(value["recipe"], []).append((key, value))
        with torch.inference_mode():
            for values in buckets.values():
                size = (
                    settings.get("encode_batch_size", 4)
                    if modalities <= {"text", "structured"}
                    else 1
                )
                for offset in range(0, len(values), size):
                    limit()
                    chunk = values[offset : offset + size]
                    samples = [
                        {**v, "max_tokens": min(v["max_tokens"], settings["processed_tokens"])}
                        for _, v in chunk
                    ]
                    features = (
                        batch_features(model, samples)
                        if size > 1
                        else features_for(model, samples[0])
                    )
                    vectors = raw_forward(model, features).float().cpu().tolist()
                    if len(vectors) != len(chunk):
                        raise ValueError("Encoder output cardinality changed")
                    forwards += 1
                    mask = features.get("attention_mask")
                    tokens += (
                        int(mask.sum()) if mask is not None else int(features["input_ids"].numel())
                    )
                    for (key, _), vector in zip(chunk, vectors, strict=True):
                        document["entries"][key] = artifacts.write_carrier(
                            Path(request["artifact_dir"]), key, vector, binding
                        )
                    artifacts.atomic_json(manifest, document)
                    limit()
    limit()
    return {
        "manifest": str(manifest.resolve()),
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "carriers": len(wanted),
        "cache_hits": hits,
        "encoder_forwards": forwards,
        "model_loads": int(bool(pending)),
        "processed_tokens": tokens,
        "elapsed_seconds": time.monotonic() - started,
    }


def restore_candidate(
    model: Any, checkpoint: str
) -> tuple[Any, list[float] | dict[str, list[float]]]:
    torch: Any = importlib.import_module("torch")
    root = Path(checkpoint)
    state = json.loads((root / "trainer_state.json").read_text())
    if state["status"] != "complete":
        raise ValueError("Serving requires a completed checkpoint")
    saved = torch.load(root / "encoder_state.pt", map_location=model.device, weights_only=True)
    parameters = saved.get("selected_parameters") or saved["parameters"]
    if state.get("settings", {}).get("mode") == "adapter":
        peft: Any = importlib.import_module("peft")
        settings = state["settings"]
        model[0].auto_model = peft.get_peft_model(
            model[0].auto_model,
            peft.LoraConfig(
                r=settings["rank"],
                lora_alpha=settings["alpha"],
                lora_dropout=settings["dropout"],
                target_modules=state["targets"],
                bias="none",
            ),
        )
        for name, parameter in model.named_parameters():
            if "encoder." + name in parameters:
                parameter.data.copy_(parameters["encoder." + name])
    if state.get("settings", {}).get("head_layout") == "per_rung":
        return model, {
            str(d): parameters[f"ranker.{d}.weight"][0].float().cpu().tolist()
            for d in (128, 256, 512, 768)
        }
    return model, parameters["ranker.weight"][0].float().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        choices=("metadata", "encode", "qualify", "cache", "fit", "checkpoint_metadata"),
    )
    parser.add_argument("--request")
    args = parser.parse_args()
    if args.operation == "metadata":
        print(json.dumps(metadata()))
        return
    request = json.loads(Path(args.request).read_text())
    request["_started"] = request.get("started", time.monotonic())
    if args.operation == "cache":
        print(json.dumps(cache_carriers(request)))
        return
    if args.operation == "fit" and request["settings"].get("embedding_cache"):
        if request["settings"]["mode"] != "frozen_heads":
            raise ValueError("Adapter training cannot use frozen carriers")
        artifacts = sibling("artifacts")
        vectors, _ = artifacts.load_carriers(
            Path(request["settings"]["embedding_cache"]),
            request["cache_binding"],
            set(artifacts.inputs(request)),
        )
        print(
            json.dumps(sibling("fit_worker").fit(request, None, None, None, cached_vectors=vectors))
        )
        return
    torch: Any = importlib.import_module("torch")
    if args.operation == "checkpoint_metadata":
        root = Path(request["checkpoint"])
        state = json.loads((root / "trainer_state.json").read_text())
        saved = torch.load(root / "encoder_state.pt", map_location="cpu", weights_only=True)
        encoder: list[list[Any]] = []
        heads: list[list[Any]] = []
        for name, value in sorted(
            (saved.get("selected_parameters") or saved["parameters"]).items()
        ):
            digest = hashlib.sha256(
                value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            entry = [name, str(value.dtype), list(value.shape), digest]
            (encoder if name.startswith("encoder.") else heads).append(entry)
        print(
            json.dumps(
                {
                    "encoder_id": hashlib.sha256(
                        json.dumps(
                            {
                                "parameters": encoder,
                                "targets": state["targets"],
                                "rank": state["settings"]["rank"],
                                "alpha": state["settings"]["alpha"],
                            },
                            sort_keys=True,
                        ).encode()
                    ).hexdigest(),
                    "head_id": hashlib.sha256(
                        json.dumps(heads, sort_keys=True).encode()
                    ).hexdigest(),
                }
            )
        )
        return
    request["_started"] = request.get("started", time.monotonic())
    if args.operation == "fit":
        inputs = [
            value
            for rows in request["groups"].values()
            for row in rows
            for value in (row["query"], row["positive"], *row["negatives"])
        ]
        modalities = {part["modality"] for value in inputs for part in value["parts"]}
    elif args.operation == "qualify":
        modalities = set(request.get("modalities", ["text"]))
    else:
        modalities = {part["modality"] for part in request["input"]["parts"]}
    model = load_model(request["model"], request.get("device", "cpu"), modalities)
    if args.operation == "qualify":
        targets = [
            name
            for name, _ in model[0].auto_model.named_modules()
            if ".language_model." in "." + name and name.endswith((".q_proj", ".v_proj"))
        ]
        print(json.dumps({**metadata(), "trainable_modules": targets}))
        return
    if args.operation == "fit":
        print(json.dumps(sibling("fit_worker").fit(request, model, features_for, raw_forward)))
        return
    head_weights = None
    if request.get("checkpoint"):
        model, head_weights = restore_candidate(model, request["checkpoint"])
    model.eval()
    with torch.inference_mode():
        vector = raw_forward(model, features_for(model, request["input"]))
    print(json.dumps({"values": vector[0].float().cpu().tolist(), "head_weights": head_weights}))


if __name__ == "__main__":
    main()
