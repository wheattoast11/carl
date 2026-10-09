"""Carrier reuse, processor budgets and independent ranking-head objectives."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
from carl_encoders import artifacts, worker


@pytest.mark.parametrize("exceeded", ["host", "reserved"])
def test_gpu_cache_enforces_both_memory_spaces(tmp_path, monkeypatch, exceeded):
    import resource

    req = request(tmp_path)
    req["model"] = "fixture"
    req["cache_binding"] = {"execution": {"device": "cuda:0", "dtype": "float32"}}
    loaded = False

    def load(*args):
        nonlocal loaded
        loaded = True
        return SimpleNamespace(device="cuda:0", eval=lambda: None)

    monkeypatch.setattr(worker, "load_model", load)
    monkeypatch.setattr(
        resource,
        "getrusage",
        lambda who: SimpleNamespace(
            ru_maxrss=8 * 1024**2 if loaded and exceeded == "host" else 1024,
        ),
    )
    monkeypatch.setattr(
        worker.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            cuda=SimpleNamespace(
                max_memory_reserved=lambda device: 8 * 1024**3 if exceeded == "reserved" else 1024,
                max_memory_allocated=lambda device: 1024,
            ),
            inference_mode=lambda: pytest.fail("Inference ran above its memory ceiling"),
        ),
    )
    with pytest.raises(TimeoutError, match="memory exceeded"):
        worker.cache_carriers(req)
    assert loaded


def test_metadata_binds_native_thread_limits_before_loading_models(monkeypatch: pytest.MonkeyPatch):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "32")
    worker.configure_threads()
    assert {
        os.environ[name] for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    } == {"4"}
    assert (
        worker.metadata()["dependencies"]["carl.encoder.thread_policy"]
        == "OMP=4,OPENBLAS=4,MKL=4,torch<=4"
    )


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


def test_cpu_checkpoint_does_not_initialize_available_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import torch
    from carl_encoders.fit_worker import fit

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "get_rng_state_all", lambda: pytest.fail("CPU fit accessed GPU RNG")
    )
    req = request(tmp_path)
    Path(req["output"]).mkdir()
    (Path(req["output"]) / "cancel").touch()
    result = fit(req, None, None, None, cached_vectors=cached(req))
    assert result["status"] == "stopped"
    saved = torch.load(Path(req["output"]) / "encoder_state.pt", weights_only=True)
    assert saved["cuda_rng"] == []


def test_cpu_worker_hides_gpu_devices_only_in_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import sys
    from types import SimpleNamespace

    from carl_studio.semantic.local import invoke

    environments = []

    class Processes:
        def spawn(self, argv, **kwargs):
            environments.append(kwargs["env"])
            return {"ref_id": "worker"}

        def wait(self, *args, **kwargs):
            return {"exit_code": 0, "stdout_ref": {"ref_id": "stdout"}}

        def terminate(self, *args):
            pass

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    session = SimpleNamespace(
        subprocess_toolkit=Processes(),
        data_toolkit=SimpleNamespace(read_text=lambda *args, **kwargs: {"text": "{}"}),
    )
    invoke(session, Path(sys.executable), "fit", {"settings": {"execution": {"device": "cpu"}}})
    assert all(
        environments[0][key] == ""
        for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    )
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    invoke(session, Path(sys.executable), "encode", {"device": "cuda"})
    assert environments[1] is None


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
