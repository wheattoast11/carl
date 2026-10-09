"""Adapter baselines reuse bound raw carriers without freezing adaptation."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from carl_encoders.artifacts import SCHEMA, atomic_json, inputs, write_carrier
from carl_encoders.fit_worker import fit


def test_installed_peft_source_changes_are_bound(tmp_path, monkeypatch):
    from carl_encoders.fit_worker import peft_implementation_digest

    source = tmp_path / "lora.py"
    source.write_text("rank = 8\n")
    distribution = SimpleNamespace(files=["peft/lora.py"], locate_file=lambda entry: source)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)
    before = peft_implementation_digest()
    source.write_text("rank = 16\n")
    assert peft_implementation_digest() != before
    distribution.files = []
    with pytest.raises(ValueError, match="source is unavailable"):
        peft_implementation_digest()


def test_adapter_cache_baseline_and_differentiable_batches(tmp_path, monkeypatch):
    torch = importlib.import_module("torch")

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.q_proj = torch.nn.Linear(1, 1, bias=False)
            self.language_model.v_proj = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.language_model.q_proj.weight.fill_(0)
                self.language_model.v_proj.weight.fill_(0)

    class Tower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.auto_model = Backbone()

    class Model(torch.nn.ModuleList):
        device = torch.device("cpu")
        prompts: ClassVar[dict[str, str]] = {"Document": "document"}

        def __init__(self):
            super().__init__([Tower()])
            self.batches = []

        def preprocess(self, messages, **kwargs):
            self.batches.append(len(messages))
            numbers = [int(m[0]["content"][0]["text"]) for m in messages]
            return {
                "input_ids": torch.tensor(numbers).reshape(-1, 1),
                "attention_mask": torch.ones((len(messages), 1)),
            }

    def forward(model, features):
        numbers = features["input_ids"].float()
        carrier = torch.ones((len(numbers), 768))
        carrier[:, :128] = numbers
        delta = model[0].auto_model.language_model.q_proj(numbers)
        return carrier + delta * torch.linspace(-1, 1, 768)

    def process(model, sample):
        return model.preprocess([[{"content": [{"text": sample["parts"][0]["text"]}]}]])

    def adapt(backbone, config):
        backbone.requires_grad_(True)
        return backbone

    fake_peft = SimpleNamespace(LoraConfig=lambda **kw: kw, get_peft_model=adapt)
    original_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, package=None: fake_peft if name == "peft" else original_import(name, package),
    )

    def sample(identity, number):
        return {
            "event_id": identity,
            "recipe": "Document",
            "max_tokens": 16,
            "parts": [{"modality": "text", "text": str(number)}],
        }

    rows = [
        {
            "id": name,
            "query": sample(name + "q", 1),
            "positive": sample(name + "p", 2),
            "negatives": [sample(name + "n", -1)],
            "relation": "supports",
        }
        for name in ("a", "b")
    ]
    settings = {
        "mode": "adapter",
        "rank": 8,
        "alpha": 16,
        "dropout": 0,
        "seed": 42,
        "runtime_s": 30,
        "memory_gib": 12,
        "optimizer_steps": 2,
        "microbatch": 1,
        "accumulation": 1,
        "processed_tokens": 16,
        "temperature": 0.05,
        "validation_every_steps": 1,
        "relation_weight": 0,
        "execution": {
            "processor_sha256": "a" * 64,
            "trainable_modules": ["language_model.q_proj", "language_model.v_proj"],
        },
    }
    request = {
        "settings": settings,
        "groups": {name: rows for name in ("train", "validation", "test")},
        "output": str(tmp_path / "fit"),
        "learning_rate": 0.01,
        "binding": "fixture",
        "cache_binding": {"model": "fixture", "execution": {"trainable_modules": []}},
    }
    model = Model()
    entries = {}
    for key, value in inputs(request).items():
        with torch.no_grad():
            vector = forward(model, process(model, value))[0].tolist()
        entries[key] = write_carrier(tmp_path / "carriers", key, vector, request["cache_binding"])
    manifest = tmp_path / "manifest.json"
    atomic_json(
        manifest, {"schema": SCHEMA, "binding": request["cache_binding"], "entries": entries}
    )
    settings["baseline_cache"] = str(manifest)
    model.batches.clear()
    result = fit(request, model, process, forward)
    assert result["counters"]["gradient_forwards"] == 6
    assert result["counters"]["cache_hits"] >= 12
    assert any(size > 1 for size in model.batches)
    assert any(name.startswith("encoder.") for name in result["updated_parameters"])
    state = torch.load(Path(result["checkpoint"]) / "encoder_state.pt", weights_only=True)
    assert state["selected_validation"] == result["validation"]
    assert state["counters"] == result["counters"]

    corrupted = json.loads(manifest.read_text())
    key = next(iter(entries))
    corrupted["entries"][key] = write_carrier(
        tmp_path / "carriers", key, [2.0] * 768, request["cache_binding"]
    )
    atomic_json(manifest, corrupted)
    with pytest.raises(ValueError, match="numerical correspondence"):
        fit({**request, "output": str(tmp_path / "wrong")}, Model(), process, forward)
