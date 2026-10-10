"""Device qualification refuses silent fallback and binds carrier precision."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from carl_core.encoder_settings import EncoderSettings
from carl_encoders import worker


def fake_torch() -> Any:
    def device(name: str) -> Any:
        kind, _, index = name.partition(":")
        return SimpleNamespace(type=kind, index=int(index) if index else None)

    return SimpleNamespace(
        device=device,
        float32="float32",
        bfloat16="bfloat16",
        set_num_threads=lambda count: None,
        get_num_threads=lambda: 16,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_properties=lambda index: SimpleNamespace(name="A40", major=8, minor=6),
        ),
        version=SimpleNamespace(cuda="13.0", hip=None),
    )


def test_gpu_metadata_requires_available_device(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = fake_torch()
    monkeypatch.setattr(worker.importlib, "import_module", lambda name: torch)
    result = worker.metadata("cuda", "float32")
    assert result["device"] == "cuda:0"
    assert result["dtype"] == "float32"
    assert result["dependencies"]["carl.encoder.cuda_device"] == "A40"
    assert result["dependencies"]["carl.encoder.cuda_capability"] == "8.6"
    torch.cuda.is_available = lambda: False
    with pytest.raises(ValueError, match="unavailable"):
        worker.metadata("cuda", "float32")


@pytest.mark.parametrize(
    ("actual_device", "actual_dtype", "error"),
    [
        ("cpu", "float32", "device"),
        ("cuda:1", "float32", "device"),
        ("cuda:0", "bfloat16", "precision"),
        ("cuda:0", "float32", None),
    ],
)
def test_loaded_encoder_matches_declared_device_and_precision(
    monkeypatch: pytest.MonkeyPatch,
    actual_device: str,
    actual_dtype: str,
    error: str | None,
) -> None:
    torch = fake_torch()
    calls: list[dict[str, Any]] = []

    class Model:
        device = actual_device

        def __getitem__(self, index: int) -> Any:
            return SimpleNamespace(
                auto_model=SimpleNamespace(
                    parameters=lambda: iter([SimpleNamespace(dtype=actual_dtype)])
                )
            )

    def construct(path: str, **kwargs: Any) -> Model:
        calls.append(kwargs)
        return Model()

    monkeypatch.setattr(
        worker.importlib,
        "import_module",
        lambda name: torch if name == "torch" else SimpleNamespace(SentenceTransformer=construct),
    )
    if error:
        with pytest.raises(ValueError, match=error):
            worker.load_model("fixture", "cuda:0", {"text"}, "float32")
    else:
        worker.load_model("fixture", "cuda:0", {"text"}, "float32")
    assert calls[0]["device"] == "cuda:0"
    assert calls[0]["model_kwargs"]["torch_dtype"] == "float32"


def test_cpu_metadata_does_not_import_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker.importlib, "import_module", lambda name: pytest.fail(name))
    result = worker.metadata()
    assert result["device"] == "cpu"
    assert result["dtype"] == "float32"


def test_settings_require_explicit_supported_precision() -> None:
    fields = {"interpreter": "python", "validation_dataset": "validation", "cutoff": "today"}
    assert EncoderSettings(**fields).device == "cpu"
    assert EncoderSettings(**fields, device="cuda:0").dtype == "float32"
    with pytest.raises(ValueError):
        EncoderSettings(**fields, device="cpu", dtype="bfloat16")
    with pytest.raises(ValueError):
        EncoderSettings(**fields, device="auto")
