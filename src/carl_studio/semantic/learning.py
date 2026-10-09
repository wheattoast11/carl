"""Compatibility imports for carl-encoders.learning."""

from __future__ import annotations

try:
    from carl_encoders.learning import (
        EncoderExample,
        EncoderSettings,
        RepresentationMeasurement,
        representation_acceptance,
        validate_splits,
    )
except ImportError as exc:
    raise ImportError("Install carl-studio[encoders] for encoder learning") from exc


__all__ = [
    "EncoderExample",
    "EncoderSettings",
    "RepresentationMeasurement",
    "representation_acceptance",
    "validate_splits",
]
