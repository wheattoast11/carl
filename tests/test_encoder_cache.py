"""Carrier reuse, processor budgets and independent ranking-head objectives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from carl_encoders import artifacts, worker


def sample(text: str, event: str = "event") -> dict[str, Any]:
    return {
        "event_id": event,
        "recipe": "Document",
        "max_tokens": 2048,
        "parts": [{"modality": "text", "text": text}],
    }


def request(root: Path) -> dict[str, Any]:
    rows = [
        {
            "id": str(i),
            "task_family": str(i % 4),
            "relation": "corrects",
            "query": sample(f"query-{i}"),
            "positive": sample(f"positive-{i}"),
            "negatives": [sample(f"negative-{i}")],
        }
        for i in range(8)
    ]
    return {
        "groups": {k: rows for k in ("train", "validation", "test")},
        "settings": {
            "processed_tokens": 2048,
            "memory_gib": 6,
            "runtime_s": 30,
            "seed": 42,
            "mode": "frozen_heads",
            "optimizer_steps": 4,
            "microbatch": 4,
            "accumulation": 1,
            "temperature": 0.05,
            "head_layout": "per_rung",
            "relation_weight": 0,
            "validation_every_steps": 1,
            "execution": {"processor_sha256": "a" * 64},
        },
        "output": str(root / "out"),
        "binding": "fixture",
        "learning_rate": 0.01,
        "manifest": str(root / "manifest.json"),
        "cache_binding": {"source": "fixture"},
    }


def cached(req: dict[str, Any]) -> dict[str, list[float]]:
    result = {}
    for key, value in artifacts.inputs(req).items():
        text = value["parts"][0]["text"]
        vector = [1.0] * 768
        vector[0] = 2.0 if "positive" in text else -1.0
        vector[767] = float(len(text))
        result[key] = vector
    return result


def test_occurrences_share_carrier_but_goals_and_recipes_do_not():
    a = sample("goal-a", "first")
    assert artifacts.sample_key(a, 2048) == artifacts.sample_key({**a, "event_id": "second"}, 2048)
    assert artifacts.sample_key(a, 2048) != artifacts.sample_key(sample("goal-b"), 2048)
    assert artifacts.sample_key(a, 2048) != artifacts.sample_key(
        {**a, "recipe": "SearchQuery"}, 2048
    )
    assert artifacts.sample_key(a, 2048) != artifacts.sample_key(a, 1024)


def test_durable_cache_restart_never_loads_encoder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    req = request(tmp_path)
    entries = {
        key: artifacts.write_carrier(tmp_path, key, vector, req["cache_binding"])
        for key, vector in cached(req).items()
    }
    artifacts.atomic_json(
        Path(req["manifest"]),
        {
            "schema": artifacts.SCHEMA,
            "binding": req["cache_binding"],
            "population": sorted(entries),
            "entries": entries,
        },
    )
    monkeypatch.setattr(worker, "load_model", lambda *args: pytest.fail("encoder loaded"))
    first = worker.cache_carriers(req)
    second = worker.cache_carriers(req)
    assert first["sha256"] == second["sha256"]
    assert second["cache_hits"] == len(entries)
    assert second["encoder_forwards"] == second["model_loads"] == 0
    vector_path = Path(next(iter(entries.values()))["path"])
    vector_path.write_text("changed")
    with pytest.raises(ValueError, match="bytes changed"):
        worker.cache_carriers(req)


def test_cache_rejects_wrong_binding_and_population(tmp_path: Path):
    path = tmp_path / "manifest.json"
    artifacts.atomic_json(
        path, {"schema": artifacts.SCHEMA, "binding": {"source": "a"}, "entries": {}}
    )
    with pytest.raises(ValueError, match="binding"):
        artifacts.load_carriers(path, {"source": "b"}, set())
    with pytest.raises(ValueError, match="population"):
        artifacts.load_carriers(path, {"source": "a"}, {"missing"})


def test_batch_padding_uses_each_actual_length():
    class Processor:
        prompts: ClassVar[dict[str, str]] = {"Document": "document"}
        device = "cpu"

        def preprocess(self, inputs, **kwargs):
            assert len(inputs) == 2
            assert kwargs["processing_kwargs"]["text"] == {"truncation": False, "padding": True}
            return {
                "input_ids": np.zeros((2, 8)),
                "attention_mask": np.array([[1, 1, 1, 0, 0, 0, 0, 0], [1] * 8]),
            }

    rows = [{**sample("short"), "max_tokens": 3}, {**sample("long"), "max_tokens": 8}]
    worker.batch_features(Processor(), rows)
    with pytest.raises(ValueError, match="budget"):
        worker.batch_features(Processor(), [rows[0], {**rows[1], "max_tokens": 7}])
    with pytest.raises(ValueError, match="one known recipe"):
        worker.batch_features(Processor(), [rows[0], {**rows[1], "recipe": "SearchQuery"}])


def test_cached_fit_balances_families_and_serves_selected_heads(tmp_path: Path):
    import torch
    from carl_encoders.fit_worker import fit

    req = request(tmp_path)
    result = fit(req, None, None, None, cached_vectors=cached(req))
    assert result["counters"]["encoder_forwards"] == result["model_loads"] == 0
    assert result["history"][0]["family_counts"] == {str(i): 1 for i in range(4)}
    assert all(name.startswith("ranker.") for name in result["updated_parameters"])
    saved = torch.load(Path(req["output"]) / "encoder_state.pt", weights_only=True)
    assert saved["sampler_order"] and saved["selected_step"] >= 1
    state = json.loads((Path(req["output"]) / "trainer_state.json").read_text())
    assert state["status"] == "complete"
    assert (Path(req["output"]) / "measurements.json").exists()

    class Model:
        device = "cpu"

    _, heads = worker.restore_candidate(Model(), req["output"])
    assert isinstance(heads, dict)
    assert heads["128"] == saved["selected_parameters"]["ranker.128.weight"][0].tolist()


def test_auxiliary_clipping_cannot_change_frozen_ranker_updates(tmp_path: Path):
    import torch
    from carl_encoders.fit_worker import fit

    req = request(tmp_path)
    fit(req, None, None, None, cached_vectors=cached(req))
    with_relation = {
        **req,
        "output": str(tmp_path / "relation"),
        "settings": {**req["settings"], "relation_weight": 100.0},
    }
    fit(with_relation, None, None, None, cached_vectors=cached(req))
    left = torch.load(Path(req["output"]) / "encoder_state.pt", weights_only=True)["parameters"]
    right = torch.load(Path(with_relation["output"]) / "encoder_state.pt", weights_only=True)[
        "parameters"
    ]
    assert all(torch.equal(value, right[name]) for name, value in left.items())


def test_cached_measurement_observes_cancellation(tmp_path: Path):
    from carl_encoders.fit_worker import fit

    req = request(tmp_path)
    output = Path(req["output"])
    output.mkdir()
    (output / "cancel").touch()
    result = fit(req, None, None, None, cached_vectors=cached(req))
    assert result["status"] == "stopped"
    assert result["steps"] == 0


def test_cached_measurement_observes_deadline(tmp_path: Path):
    from carl_encoders.fit_worker import fit

    req = {**request(tmp_path), "_started": 0.0}
    with pytest.raises(TimeoutError, match="resource ceiling"):
        fit(req, None, None, None, cached_vectors=cached(req))
    state = json.loads((Path(req["output"]) / "trainer_state.json").read_text())
    assert state["status"] == "stopped" and state["global_step"] == 0


def test_final_evaluation_interruption_keeps_optimizer_parameter_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import torch
    from carl_encoders import fit_worker

    req = request(tmp_path)
    req["settings"]["early_stop_patience"] = 10
    req["groups"]["validation"] = [
        {**row, "positive": row["negatives"][0], "negatives": [row["positive"]]}
        for row in req["groups"]["validation"]
    ]
    original_load = fit_worker.importlib.util.module_from_spec
    before: dict[str, Any] = {}

    def load(spec):
        module = original_load(spec)
        original_exec = spec.loader.exec_module

        def execute(target):
            original_exec(target)

            def interrupt(path, value):
                before.update(
                    torch.load(Path(req["output"]) / "encoder_state.pt", weights_only=True)
                )
                assert before["selected_step"] < before["step"]
                assert any(
                    not torch.equal(v, before["selected_parameters"][k])
                    for k, v in before["parameters"].items()
                )
                raise InterruptedError("final measurement interrupted")

            target.atomic_json = interrupt

        spec.loader.exec_module = execute
        return module

    monkeypatch.setattr(fit_worker.importlib.util, "module_from_spec", load)
    result = fit_worker.fit(req, None, None, None, cached_vectors=cached(req))
    assert result["status"] == "stopped"
    stopped = torch.load(Path(req["output"]) / "encoder_state.pt", weights_only=True)
    assert stopped["step"] == before["step"]
    assert all(torch.equal(v, stopped["parameters"][k]) for k, v in before["parameters"].items())
