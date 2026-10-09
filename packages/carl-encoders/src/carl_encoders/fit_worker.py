"""Bounded differentiable encoder learning in the isolated worker environment."""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import random
import resource
import time
from pathlib import Path
from typing import Any


def fit(
    request: dict[str, Any],
    model: Any,
    features_for: Any,
    raw_forward: Any,
    *,
    cached_vectors: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    torch: Any = importlib.import_module("torch")
    functional: Any = importlib.import_module("torch.nn.functional")
    spec = importlib.util.spec_from_file_location(
        "artifacts", Path(__file__).with_name("artifacts.py")
    )
    assert spec is not None and spec.loader is not None
    artifacts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(artifacts)

    settings = request["settings"]
    random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    started = request.get("_started", time.monotonic())
    deadline = started + settings["runtime_s"]
    ceiling = settings["memory_gib"] * 1024**3
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=True)
    device = model.device if model is not None else torch.device("cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    mode = settings["mode"]
    if model is not None:
        model.requires_grad_(False)
    elif cached_vectors is None or mode != "frozen_heads":
        raise ValueError("Frozen-head fitting requires bound carriers or an encoder")
    baseline_vectors = None
    if settings.get("baseline_cache"):
        if mode != "adapter" or model is None:
            raise ValueError("Baseline cache requires an adapter encoder")
        baseline_vectors, _ = artifacts.load_carriers(
            Path(settings["baseline_cache"]),
            request["cache_binding"],
            set(artifacts.inputs(request)),
        )
        sample = next(iter(artifacts.inputs(request).values()))
        sample = {**sample, "max_tokens": min(sample["max_tokens"], settings["processed_tokens"])}
        model.eval()
        with torch.no_grad():
            fresh = raw_forward(model, features_for(model, sample))[0].float().cpu()
        expected = torch.tensor(
            baseline_vectors[artifacts.sample_key(sample, settings["processed_tokens"])]
        )
        if not torch.allclose(fresh, expected, rtol=1e-5, atol=1e-6):
            raise ValueError("Baseline cache numerical correspondence failed")
    targets: list[str] = []
    if mode == "adapter":
        if model is None:
            raise ValueError("Adapter training requires an encoder")
        peft: Any = importlib.import_module("peft")
        LoraConfig = peft.LoraConfig
        get_peft_model = peft.get_peft_model

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
    per_rung = settings.get("head_layout", "shared") == "per_rung"
    ranker = (
        torch.nn.ModuleDict(
            {str(d): torch.nn.Linear(d, 1, bias=False) for d in (128, 256, 512, 768)}
        )
        if per_rung
        else torch.nn.Linear(768, 1, bias=False)
    ).to(device)
    relation_weight = settings.get("relation_weight", 1.0)
    relation = torch.nn.Linear(768, 4).to(device) if relation_weight else None
    with torch.no_grad():
        for parameter in ranker.parameters():
            parameter.fill_(1)

    def weights(dimension: int) -> Any:
        return ranker[str(dimension)].weight[0] if per_rung else ranker.weight[0, :dimension]

    trainable = [("ranker." + n, p) for n, p in ranker.named_parameters()]
    if relation is not None:
        trainable += [("relation." + n, p) for n, p in relation.named_parameters()]
    if model is not None:
        trainable += [("encoder." + n, p) for n, p in model.named_parameters() if p.requires_grad]
    identities = tuple(name for name, _ in trainable)
    initial = {name: p.detach().clone() for name, p in trainable}
    policy = request.get(
        "optimizer",
        {
            "weight_decay": 0.01,
            "warmup_ratio": 0,
            "lr_scheduler_type": "linear",
            "max_grad_norm": 1,
        },
    )
    if policy["lr_scheduler_type"] not in {"linear", "cosine", "constant"}:
        raise ValueError("Unsupported encoder learning-rate schedule")
    optimizer = torch.optim.AdamW(
        [p for _, p in trainable], lr=request["learning_rate"], weight_decay=policy["weight_decay"]
    )

    def schedule(step: int) -> float:
        warmup = int(settings["optimizer_steps"] * policy["warmup_ratio"])
        if step < warmup:
            return max(1, step) / warmup
        progress = min(1.0, (step - warmup) / max(1, settings["optimizer_steps"] - warmup))
        if policy["lr_scheduler_type"] == "constant":
            return 1.0
        return (
            (1 + math.cos(math.pi * progress)) / 2
            if policy["lr_scheduler_type"] == "cosine"
            else 1 - progress
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    global_step = 0
    data_position = 0
    cache: dict[str, Any] = {}
    counters = {
        "encoder_forwards": int(baseline_vectors is not None),
        "gradient_forwards": 0,
        "cache_hits": 0,
        "processed_tokens": 0,
    }
    timings = {
        "encoding_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "optimizer_seconds": 0.0,
        "checkpoint_seconds": 0.0,
    }
    history: list[dict[str, Any]] = []
    best_parameters: dict[str, Any] | None = None
    best_validation: dict[str, Any] | None = None
    baseline_phase = False
    evaluation_cache: dict[str, Any] = {}
    best_step = 0
    best_score: tuple[bool, float, float] | None = None
    stale_checks = 0
    order: list[int] = []
    order_position = 0
    current_parameters: dict[str, Any] | None = None
    boundary: tuple[Any, ...] | None = None

    def restore_boundary() -> None:
        nonlocal data_position, order, order_position
        if boundary is not None:
            data_position, order, order_position, python_rng, torch_rng, cuda_rng = boundary
            random.setstate(python_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng:
                torch.cuda.set_rng_state_all(cuda_rng)

    def next_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal order, order_position
        if order_position >= len(order):
            if settings.get("balanced_sampling", True):
                families: dict[str, list[int]] = {}
                for index, row in enumerate(rows):
                    families.setdefault(row.get("task_family", "general"), []).append(index)
                for members in families.values():
                    random.shuffle(members)
                names = sorted(families)
                order = []
                for offset in range(max(map(len, families.values()))):
                    random.shuffle(names)
                    order.extend(families[name][offset % len(families[name])] for name in names)
            else:
                order = list(range(len(rows)))
            order_position = 0
        row = rows[order[order_position]]
        order_position += 1
        return row

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
        limit()
        sample = {**sample, "max_tokens": min(sample["max_tokens"], settings["processed_tokens"])}
        key = artifacts.sample_key(sample, settings["processed_tokens"])
        if baseline_phase and baseline_vectors is not None:
            counters["cache_hits"] += 1
            return torch.tensor(baseline_vectors[key], device=device)
        if not gradients and key in evaluation_cache:
            counters["cache_hits"] += 1
            return evaluation_cache[key]
        if mode == "frozen_heads" and key in cache:
            counters["cache_hits"] += 1
            return cache[key]
        if model is None:
            raise ValueError("Embedding cache population incomplete")
        limit()
        began = time.monotonic()
        features = features_for(model, sample)
        with torch.set_grad_enabled(gradients and mode == "adapter"):
            value = raw_forward(model, features)[0]
        counters["encoder_forwards"] += 1
        counters["gradient_forwards"] += int(gradients and mode == "adapter")
        mask = features.get("attention_mask")
        if mask is not None:
            counters["processed_tokens"] += int(mask.sum())
        timings["encoding_seconds"] += time.monotonic() - began
        limit()
        if value.shape != (768,) or not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite or invalid encoder carrier")
        for d in (128, 256, 512, 768):
            if float(value[:d].detach().norm()) <= 1e-12:
                raise ValueError("Collapsed encoder prefix")
        if mode == "frozen_heads":
            cache[key] = value.detach()
        return value

    labels = {
        name: index
        for index, name in enumerate(("supports", "contradicts", "corresponds", "corrects"))
    }
    if model is not None:
        model.eval()

    def measurements(rows: list[dict[str, Any]]) -> dict[str, Any]:
        began = time.monotonic()
        evaluation_cache.clear()
        if mode == "adapter" and not baseline_phase:
            spec = importlib.util.spec_from_file_location(
                "encoder_batch_worker", Path(__file__).with_name("worker.py")
            )
            assert spec is not None and spec.loader is not None
            worker = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(worker)
            samples = artifacts.inputs({"groups": {"evaluation": rows}, "settings": settings})
            buckets: dict[str, list[tuple[str, Any]]] = {}
            for key, sample in samples.items():
                if all(p["modality"] in {"text", "structured"} for p in sample["parts"]):
                    buckets.setdefault(sample["recipe"], []).append((key, sample))
            with torch.no_grad():
                for bucket in buckets.values():
                    for offset in range(0, len(bucket), settings.get("encode_batch_size", 4)):
                        limit()
                        chunk = bucket[offset : offset + settings.get("encode_batch_size", 4)]
                        batch = [
                            {
                                **sample,
                                "max_tokens": min(
                                    sample["max_tokens"], settings["processed_tokens"]
                                ),
                            }
                            for _, sample in chunk
                        ]
                        features = worker.batch_features(model, batch)
                        values = raw_forward(model, features)
                        if values.shape != (len(chunk), 768):
                            raise ValueError("Adapter evaluation batch shape changed")
                        counters["encoder_forwards"] += 1
                        counters["processed_tokens"] += int(features["attention_mask"].sum())
                        for (key, _), value in zip(chunk, values, strict=True):
                            artifacts.admit(value.float().cpu().tolist())
                            evaluation_cache[key] = value.detach()
                        limit()
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
                    scores = (products * weights(d)).sum(dim=-1)
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
        timings["evaluation_seconds"] += time.monotonic() - began
        return {
            "predictions": predictions,
            "ranking_accuracy": correct["768"] / len(rows),
            "positive_negative_margin": sum(margins) / len(margins),
            "variance": float(torch.stack(vectors).var(dim=0, unbiased=False).mean()),
            "finite": bool(torch.isfinite(torch.stack(vectors)).all()),
            "dimensions": 768,
            "slices": {name: sum(values) / len(values) for name, values in modalities.items()},
        }

    def save(status: str, parameters: dict[str, Any] | None = None) -> None:
        began = time.monotonic()
        state: dict[str, Any] = {
            "parameters": parameters
            if parameters is not None
            else {name: p.detach().cpu() for name, p in trainable},
            "selected_parameters": best_parameters,
            "selected_step": best_step,
            "selected_validation": best_validation,
            "counters": counters,
            "best_score": best_score,
            "stale_checks": stale_checks,
            "sampler_order": order,
            "sampler_position": order_position,
            "history": history,
            "parameter_identities": identities,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "step": global_step,
            "data_position": data_position,
            "processor": settings["execution"]["processor_sha256"],
            "optimizer_policy": policy,
        }
        with (output / "encoder_state.tmp").open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
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
                    "selected_step": best_step,
                }
            )
        )
        os.replace(temporary, output / "trainer_state.json")
        timings["checkpoint_seconds"] += time.monotonic() - began

    try:
        for key, value in (cached_vectors or {}).items():
            limit()
            artifacts.admit(value)
            cache[key] = torch.tensor(value, device=device)
        baseline_phase = True
        baseline = measurements(request["groups"]["test"])
        validation_baseline = measurements(request["groups"]["validation"])
        baseline_phase = False
    except InterruptedError:
        save("stopped", current_parameters)
        return {"status": "stopped", "checkpoint": str(output), "steps": global_step}
    except BaseException:
        save("stopped", current_parameters)
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
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        global_step, data_position = saved["step"], saved["data_position"]
        order, order_position = saved.get("sampler_order", []), saved.get("sampler_position", 0)
        best_parameters, best_step = saved.get("selected_parameters"), saved.get("selected_step", 0)
        best_score, stale_checks = saved.get("best_score"), saved.get("stale_checks", 0)
        history = saved.get("history", [])
        best_validation = saved.get("selected_validation")
        counters.update(saved.get("counters", {}))

    try:
        rows = request["groups"]["train"]
        while global_step < settings["optimizer_steps"] and stale_checks < settings.get(
            "early_stop_patience", 3
        ):
            if (output / "cancel").exists():
                save("stopped")
                return {"status": "stopped", "checkpoint": str(output), "steps": global_step}
            optimizer.zero_grad()
            boundary = (
                data_position,
                list(order),
                order_position,
                random.getstate(),
                torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            )
            if model is not None:
                model.train(mode == "adapter")
            step_loss = 0.0
            family_counts: dict[str, int] = {}
            began = time.monotonic()
            for _ in range(settings["accumulation"]):
                losses: list[Any] = []
                for _ in range(settings["microbatch"]):
                    row = next_row(rows)
                    family = row.get("task_family", "general")
                    family_counts[family] = family_counts.get(family, 0) + 1
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
                        logits = (products * weights(d)).sum(dim=-1) / settings["temperature"]
                        contrastive.append(
                            functional.cross_entropy(
                                logits.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=device)
                            )
                        )
                    loss = torch.stack(contrastive).mean()
                    if relation is not None:
                        loss += relation_weight * functional.cross_entropy(
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
                step_loss += float(objective.detach())
                limit()
            norms = {}
            for prefix in ("ranker.", "relation.", "encoder."):
                parameters = [p for name, p in trainable if name.startswith(prefix)]
                if parameters:
                    norms[prefix.removesuffix(".")] = float(
                        torch.nn.utils.clip_grad_norm_(
                            parameters, max_norm=policy["max_grad_norm"], error_if_nonfinite=True
                        )
                    )
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for _, p in trainable):
                raise ValueError("Nonfinite trainable parameters")
            scheduler.step()
            global_step += 1
            boundary = None
            timings["optimizer_seconds"] += time.monotonic() - began
            progress: dict[str, Any] = {
                "step": global_step,
                "loss": step_loss,
                "gradient_norms": norms,
                "family_counts": family_counts,
            }
            should_check = (
                global_step % settings.get("validation_every_steps", 16) == 0
                or global_step == settings["optimizer_steps"]
            )
            if should_check:
                if model is not None:
                    model.eval()
                checked = measurements(request["groups"]["validation"])
                score = (
                    all(
                        validation_baseline["slices"][key] - value <= 0.02
                        for key, value in checked["slices"].items()
                    ),
                    checked["ranking_accuracy"],
                    checked["positive_negative_margin"],
                )
                progress["validation"] = {k: v for k, v in checked.items() if k != "predictions"}
                if best_score is None or score > tuple(best_score):
                    best_score, best_step, stale_checks = score, global_step, 0
                    best_validation = checked
                    best_parameters = {name: p.detach().cpu().clone() for name, p in trainable}
                else:
                    stale_checks += 1
            history.append(progress)
            if should_check or global_step % settings.get("checkpoint_every_steps", 16) == 0:
                save("running")
            if stale_checks >= settings.get("early_stop_patience", 3):
                break
        current_parameters = {name: p.detach().cpu().clone() for name, p in trainable}
        if best_parameters is not None:
            with torch.no_grad():
                for name, p in trainable:
                    p.copy_(best_parameters[name])
        if model is not None:
            model.eval()
        candidate = measurements(request["groups"]["test"])
        validation = (
            best_validation
            if best_validation is not None
            else measurements(request["groups"]["validation"])
        )
        updated = [name for name, p in trainable if not torch.equal(initial[name], p)]
        result: dict[str, Any] = {
            "status": "complete",
            "checkpoint": str(output),
            "steps": global_step,
            "baseline": baseline,
            "candidate": candidate,
            "validation": validation,
            "validation_baseline": validation_baseline,
            "updated_parameters": updated,
            "targets": targets,
            "elapsed_seconds": time.monotonic() - started,
            "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "selected_step": best_step,
            "history": history,
            "timings": timings,
            "counters": counters,
            "model_loads": int(model is not None),
        }
        artifacts.atomic_json(output / "measurements.json", result)
        save("complete", current_parameters)
        return result
    except InterruptedError:
        restore_boundary()
        save("stopped", current_parameters)
        return {"status": "stopped", "checkpoint": str(output), "steps": global_step}
    except BaseException:
        restore_boundary()
        save("stopped", current_parameters)
        raise
