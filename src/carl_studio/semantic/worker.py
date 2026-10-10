"""Compatibility imports for carl-encoders.worker."""

from __future__ import annotations

from carl_encoders.worker import (
    features_for,
    load_model,
    main,
    metadata,
    raw_forward,
    restore_candidate,
)

if __name__ == "__main__":
    main()

__all__ = ["features_for", "load_model", "main", "metadata", "raw_forward", "restore_candidate"]
