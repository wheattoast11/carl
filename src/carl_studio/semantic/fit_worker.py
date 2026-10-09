"""Bounded differentiable encoder learning in the isolated worker environment."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
import resource
import time
from pathlib import Path
from typing import Any


def fit(request: dict[str, Any], model: Any, features_for: Any, raw_forward: Any) -> dict[str, Any]:
    torch: Any = importlib.import_module("torch")
    functional: Any = importlib.import_module("torch.nn.functional")

    settings = request["settings"]
    random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    started = request.get("_started", time.monotonic())
    deadline = started + settings["runtime_s"]
    ceiling = settings["memory_gib"] * 1024**3
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=True)
    device = model.device
    mode = settings["mode"]
    model.requires_grad_(False)
    targets = []
    if mode == "adapter":
        from peft import LoraConfig, get_peft_model

        targets = [
            name
            for name, _ in model[0].auto_model.named_modules()
            if ".language_model." in "." + name and name.endswith((".q_proj", ".v_proj"))
        ]
        if not targets or tuple(targets) != tuple(settings["execution"]["trainable_modules"]):
            raise ValueError("Exact PEFT target modules are not bound")
        model[0].auto_model = get_peft_model(
            model[0].auto_model,
            LoraConfig(
                r=settings["rank"],
                lora_alpha=settings["alpha"],
                lora_dropout=settings["dropout"],
                target_modules=targets,
                bias="none",
            ),
        )
    ranker = torch.nn.Linear(768, 1, bias=False).to(device)
    relation = torch.nn.Linear(768, 4).to(device)
    with torch.no_grad():
        ranker.weight.fill_(1)
    trainable = [("ranker." + n, p) for n, p in ranker.named_parameters()]
    trainable += [("relation." + n, p) for n, p in relation.named_parameters()]
    trainable += [("encoder." + n, p) for n, p in model.named_parameters() if p.requires_grad]
    identities = tuple(name for name, _ in trainable)
    initial = {name: p.detach().clone() for name, p in trainable}
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=request["learning_rate"])

    def schedule(step: int) -> float:
        return max(0.0, 1 - step / settings["optimizer_steps"])

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    global_step = 0
    data_position = 0
    cache: dict[str, Any] = {}

    def limit() -> None:
        allocated = (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        if (output / "cancel").exists():
            raise InterruptedError("Encoder execution was cancelled")
        if time.monotonic() > deadline or allocated > ceiling:
            raise TimeoutError("Encoder pilot resource ceiling exceeded")

    def vector(sample: dict[str, Any], *, gradients: bool) -> Any:
        sample = {**sample, "max_tokens": min(sample["max_tokens"], settings["processed_tokens"])}
        key = hashlib.sha256(json.dumps(sample, sort_keys=True).encode()).hexdigest()
        if mode == "frozen_heads" and key in cache:
            return cache[key]
        limit()
        features = features_for(model, sample)
        with torch.set_grad_enabled(gradients and mode == "adapter"):
            value = raw_forward(model, features)[0]
        limit()
        if value.shape != (768,) or not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite or invalid encoder carrier")
        for d in (128, 256, 512, 768):
            if float(value[:d].norm()) <= 1e-12:
                raise ValueError("Collapsed encoder prefix")
        if mode == "frozen_heads":
            cache[key] = value.detach()
        return value

    labels = {
        name: index
        for index, name in enumerate(("supports", "contradicts", "corresponds", "corrects"))
    }
    model.eval()

    def measurements(rows: list[dict[str, Any]]) -> dict[str, Any]:
        correct = {str(d): 0 for d in (128, 256, 512, 768)}
        margins: list[float] = []
        vectors: list[Any] = []
        predictions: list[dict[str, str]] = []
        modalities: dict[str, list[bool]] = {}
        with torch.no_grad():
            for row in rows:
                query = vector(row["query"], gradients=False)
                candidates = torch.stack(
                    [vector(row["positive"], gradients=False)]
                    + [vector(n, gradients=False) for n in row["negatives"]]
                )
                vectors.append(query.cpu())
                has_media = any(
                    p["modality"] not in {"text", "structured"}
                    for item in (row["query"], row["positive"], *row["negatives"])
                    for p in item["parts"]
                )
                for d in (128, 256, 512, 768):
                    if has_media and d == 128:
                        continue
                    products = functional.normalize(query[:d], dim=-1) * functional.normalize(
                        candidates[:, :d], dim=-1
                    )
                    scores = (products * ranker.weight[0, :d]).sum(dim=-1)
                    win = bool(scores[0] > scores[1:].max())
                    correct[str(d)] += win
                    selected_modalities = {
                        part["modality"]
                        for item in (row["query"], row["positive"], *row["negatives"])
                        for part in item["parts"]
                    } & {"image", "audio", "video"}
                    slices = selected_modalities | {"multimodal"} if has_media else {"text"}
                    for modality in slices:
                        modalities.setdefault(modality + ":" + str(d), []).append(win)
                    if d == 768:
                        selected = int(scores.argmax())
                        candidate_inputs = [row["positive"], *row["negatives"]]
                        predictions.append(
                            {
                                "id": row["id"],
                                "selected_event_id": candidate_inputs[selected]["event_id"]
                                if float(scores.max()) > float(scores.sort().values[-2])
                                else "abstain",
                            }
                        )
                        margins.append(float(scores[0] - scores[1:].max()))
        return {
            "predictions": predictions,
            "ranking_accuracy": correct["768"] / len(rows),
            "positive_negative_margin": sum(margins) / len(margins),
            "variance": float(torch.stack(vectors).var(dim=0, unbiased=False).mean()),
            "finite": bool(torch.isfinite(torch.stack(vectors)).all()),
            "dimensions": 768,
            "slices": {name: sum(values) / len(values) for name, values in modalities.items()},
        }

    def save(status: str) -> None:
        state: dict[str, Any] = {
            "parameters": {name: p.detach().cpu() for name, p in trainable},
            "parameter_identities": identities,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "step": global_step,
            "data_position": data_position,
            "processor": settings["execution"]["processor_sha256"],
        }
        torch.save(state, output / "encoder_state.tmp")
        os.replace(output / "encoder_state.tmp", output / "encoder_state.pt")
        temporary = output / "trainer_state.tmp"
        temporary.write_text(
            json.dumps(
                {
                    "global_step": global_step,
                    "data_position": data_position,
                    "binding": request["binding"],
                    "status": status,
                    "targets": targets,
                    "settings": settings,
                    "parameter_identities": identities,
                    "run_id": request.get("run_id"),
                }
            )
        )
        os.replace(temporary, output / "trainer_state.json")

    try:
        baseline = measurements(request["groups"]["test"])
    except InterruptedError:
        save("stopped")
        return {"status": "stopped", "checkpoint": str(output), "steps": global_step}
    except BaseException:
        save("stopped")
        raise
    if request.get("resume"):
        resume = Path(request["resume"])
        state = json.loads((resume / "trainer_state.json").read_text())
        if state["status"] != "stopped" or state["binding"] != request["binding"]:
            raise ValueError("Resume requires exact stopped checkpoint binding")
        saved = torch.load(resume / "encoder_state.pt", map_location=device, weights_only=True)
        if saved["parameter_identities"] != identities:
            raise ValueError("Checkpoint trainable identities changed")
        for name, p in trainable:
            p.data.copy_(saved["parameters"][name])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        global_step, data_position = saved["step"], saved["data_position"]

    try:
        rows = request["groups"]["train"]
        while global_step < settings["optimizer_steps"]:
            if (output / "cancel").exists():
                save("stopped")
                return {"status": "stopped", "checkpoint": str(output), "steps": global_step}
            optimizer.zero_grad()
            model.train(mode == "adapter")
            for _ in range(settings["accumulation"]):
                losses: list[Any] = []
                for _ in range(settings["microbatch"]):
                    row = rows[data_position % len(rows)]
                    data_position += 1
                    q = vector(row["query"], gradients=True)
                    candidates = torch.stack(
                        [vector(row["positive"], gradients=True)]
                        + [vector(n, gradients=True) for n in row["negatives"]]
                    )
                    has_media = any(
                        p["modality"] not in {"text", "structured"}
                        for item in (row["query"], row["positive"], *row["negatives"])
                        for p in item["parts"]
                    )
                    rungs = (256, 512, 768) if has_media else (128, 256, 512, 768)
                    contrastive: list[Any] = []
                    for d in rungs:
                        products = functional.normalize(q[:d], dim=-1) * functional.normalize(
                            candidates[:, :d], dim=-1
                        )
                        logits = (products * ranker.weight[0, :d]).sum(dim=-1) / settings[
                            "temperature"
                        ]
                        contrastive.append(
                            functional.cross_entropy(
                                logits.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=device)
                            )
                        )
                    loss = torch.stack(contrastive).mean()
                    loss += functional.cross_entropy(
                        relation(
                            (
                                functional.normalize(q, dim=-1)
                                * functional.normalize(candidates[0], dim=-1)
                            ).unsqueeze(0)
                        ),
                        torch.tensor([labels[row["relation"]]], device=device),
                    )
                    losses.append(loss)
                objective = torch.stack(losses).mean() / settings["accumulation"]
                if not bool(torch.isfinite(objective)):
                    raise ValueError("Nonfinite encoder objective")
                objective.backward()
                limit()
            torch.nn.utils.clip_grad_norm_(
                [p for _, p in trainable], max_norm=1.0, error_if_nonfinite=True
            )
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for _, p in trainable):
                raise ValueError("Nonfinite trainable parameters")
            scheduler.step()
            global_step += 1
            save("running")
        model.eval()
        candidate = measurements(request["groups"]["test"])
        validation = measurements(request["groups"]["validation"])
        updated = [name for name, p in trainable if not torch.equal(initial[name], p)]
        save("complete")
        return {
            "status": "complete",
            "checkpoint": str(output),
            "steps": global_step,
            "baseline": baseline,
            "candidate": candidate,
            "validation": validation,
            "updated_parameters": updated,
            "targets": targets,
            "elapsed_seconds": time.monotonic() - started,
            "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }
    except InterruptedError:
        save("stopped")
        return {"status": "stopped", "checkpoint": str(output), "steps": global_step}
    except BaseException:
        save("stopped")
        raise
