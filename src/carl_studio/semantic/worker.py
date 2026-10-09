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
        Path(__file__).with_name("local.py").read_bytes()
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


def restore_candidate(model: Any, checkpoint: str) -> tuple[Any, list[float]]:
    torch: Any = importlib.import_module("torch")
    root = Path(checkpoint)
    state = json.loads((root / "trainer_state.json").read_text())
    if state["status"] != "complete":
        raise ValueError("Serving requires a completed checkpoint")
    saved = torch.load(root / "encoder_state.pt", map_location=model.device, weights_only=True)
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
            if "encoder." + name in saved["parameters"]:
                parameter.data.copy_(saved["parameters"]["encoder." + name])
    return model, saved["parameters"]["ranker.weight"][0].float().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation", choices=("metadata", "encode", "qualify", "fit", "checkpoint_metadata")
    )
    parser.add_argument("--request")
    args = parser.parse_args()
    if args.operation == "metadata":
        print(json.dumps(metadata()))
        return
    torch: Any = importlib.import_module("torch")

    request = json.loads(Path(args.request).read_text())
    if args.operation == "checkpoint_metadata":
        root = Path(request["checkpoint"])
        state = json.loads((root / "trainer_state.json").read_text())
        saved = torch.load(root / "encoder_state.pt", map_location="cpu", weights_only=True)
        encoder: list[list[Any]] = []
        heads: list[list[Any]] = []
        for name, value in sorted(saved["parameters"].items()):
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
        spec = importlib.util.spec_from_file_location(
            "fit_worker", Path(__file__).with_name("fit_worker.py")
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(json.dumps(module.fit(request, model, features_for, raw_forward)))
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
