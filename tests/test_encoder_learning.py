"""Offline split, processing and optimizer counterexamples."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, ClassVar

import pytest

from carl_studio.semantic.learning import EncoderExample, validate_splits
from carl_studio.semantic.worker import features_for


def example(identity: str) -> EncoderExample:
    def sample(suffix: str):
        return {
            "event_id": identity + suffix,
            "parts": [{"modality": "text", "text": identity + suffix}],
        }

    return EncoderExample(
        id=identity,
        episode_id=identity,
        original_sources=(identity,),
        observed_at="2026-10-08T00:00:00+00:00",
        feedback_ref="feedback:" + identity,
        query=sample("query"),
        positive=sample("positive"),
        negatives=(sample("negative"),),
        relation="corrects",
        action_label=identity,
    )


def test_split_lineage_and_temporal_cutoff():
    groups = {"train": [example("a")], "validation": [example("b")], "test": [example("c")]}
    validate_splits(groups, "2026-10-09T00:00:00+00:00")
    for key, value in (
        ("episode_id", "a"),
        ("original_sources", ("a",)),
        ("observed_at", "2026-10-10T00:00:00+00:00"),
    ):
        groups["test"] = [example("c").model_copy(update={key: value})]
        with pytest.raises(ValueError):
            validate_splits(groups, "2026-10-09T00:00:00+00:00")


def test_processor_order_and_combined_budget():
    torch: Any = importlib.import_module("torch")
    calls = []

    class Processor:
        prompts: ClassVar[dict[str, str]] = {"Document": "document"}
        device = "cpu"

        def preprocess(self, inputs, **kwargs):
            calls.append((inputs, kwargs))
            return {"input_ids": torch.zeros((1, 12), dtype=torch.long)}

    request = {
        "recipe": "Document",
        "max_tokens": 12,
        "parts": [
            {"modality": "text", "text": "before"},
            {"modality": "image", "path": "/fixture.png"},
            {"modality": "text", "text": "after"},
            {"modality": "audio", "path": "/fixture.wav"},
        ],
    }
    features_for(Processor(), request)
    assert [part["type"] for part in calls[0][0][0][0]["content"]] == [
        "text",
        "image",
        "text",
        "audio",
    ]
    assert calls[0][1]["processing_kwargs"]["text"]["truncation"] is False
    with pytest.raises(ValueError, match="budget"):
        features_for(Processor(), {**request, "max_tokens": 11})


@pytest.mark.parametrize("feature_dtype", ["float32", "bfloat16"])
def test_real_head_optimizer_checkpoint_and_constant_counterexample(
    tmp_path: Path, feature_dtype: str
):
    torch: Any = importlib.import_module("torch")
    from carl_studio.semantic.fit_worker import fit

    class Model:
        device = torch.device("cpu")

        def requires_grad_(self, enabled):
            return self

        def named_parameters(self):
            return []

        def eval(self):
            return self

        def train(self, enabled):
            return self

    settings = {
        "seed": 42,
        "runtime_s": 30,
        "memory_gib": 12,
        "mode": "frozen_heads",
        "optimizer_steps": 2,
        "accumulation": 1,
        "microbatch": 1,
        "processed_tokens": 2048,
        "temperature": 0.05,
        "execution": {"processor_sha256": "a" * 64},
    }
    rows = [example("a").model_dump(mode="json"), example("b").model_dump(mode="json")]
    request = {
        "settings": settings,
        "output": str(tmp_path),
        "learning_rate": 0.01,
        "groups": {key: rows for key in ("train", "validation", "test")},
        "binding": "fixture",
    }

    def process(model, sample):
        return {"label": sample["parts"][0]["text"]}

    def forward(model, features):
        values = torch.ones((1, 768), dtype=getattr(torch, feature_dtype))
        values[0, 0] = 2 if "positive" in features["label"] else -1
        return values

    result = fit(request, Model(), process, forward)
    assert result["steps"] == 2
    assert set(result["updated_parameters"]) >= {
        "ranker.weight",
        "relation.weight",
        "relation.bias",
    }
    assert (tmp_path / "encoder_state.pt").exists()
    constant = fit(
        {**request, "output": str(tmp_path / "constant")},
        Model(),
        process,
        lambda model, features: torch.ones((1, 768)),
    )
    assert constant["baseline"]["ranking_accuracy"] == 0
    assert constant["baseline"]["positive_negative_margin"] == 0
    with pytest.raises(ValueError, match="Nonfinite"):
        fit(
            {**request, "output": str(tmp_path / "nan")},
            Model(),
            process,
            lambda model, features: torch.full((1, 768), float("nan")),
        )


def test_encoder_settings_require_explicit_groups():
    from carl_studio.types.config import TrainingConfig

    with pytest.raises(ValueError):
        TrainingConfig(
            run_name="missing",
            base_model="source",
            output_repo="local/candidate",
            method="encoder",
            encoder={"interpreter": "python"},
        )


def test_serving_identity_includes_processor_selection(tmp_path):
    from carl_studio.semantic.local import source_identity_checkpoint

    (tmp_path / "encoder_state.pt").write_bytes(b"parameters")
    (tmp_path / "trainer_state.json").write_text('{"mode":"heads"}')
    first = source_identity_checkpoint(tmp_path)
    (tmp_path / "trainer_state.json").write_text('{"mode":"adapter"}')
    assert source_identity_checkpoint(tmp_path) != first


def test_media_only_has_no_text_prompt():
    torch: Any = importlib.import_module("torch")
    prompts = []

    class Model:
        device = "cpu"

        def __init__(self):
            self.prompts = {"Document": "text-prefix"}

        def preprocess(self, inputs, **kwargs):
            prompts.append(kwargs["prompt"])
            return {"input_ids": torch.zeros((1, 8), dtype=torch.long)}

    features_for(
        Model(),
        {
            "recipe": "Document",
            "max_tokens": 8,
            "parts": [{"modality": "image", "path": "/fixture.png"}],
        },
    )
    assert prompts == [None]
