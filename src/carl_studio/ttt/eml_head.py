"""Public handle for the EML TTT head.

EML is Odrzywolek's exp-minus-log magma; the operator itself is public
(arXiv 2603.21852). The fitted trees and Adam training procedure behind
them live in ``terminals-runtime`` (BUSL-1.1, private). This module
exposes an opaque handle so MIT callers can carry an EML head around
without gaining access to the fitter.

Usage::

    from carl_studio.ttt.eml_head import EMLHead

    head = EMLHead()
    head.fit(xs, ys)                       # delegates to private runtime
    y_pred = head.eval(xs)                 # evaluates (numpy, public math)
    head.save(Path("model.eml"))           # signs + serializes
    head.load(Path("model.eml"))           # verifies signature on load
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from carl_core.eml import EMLTree


def _require_runtime(thing: str) -> None:
    raise ImportError(
        f"{thing} requires terminals-runtime (private BUSL package). "
        "Contact admin or run via carl admin status."
    )


class EMLHead:
    """Opaque handle to a (possibly-fitted, possibly-signed) EML tree.

    The fitter and signer live in ``terminals_runtime.eml``. Importing
    this module never triggers the import — we only resolve it when the
    caller asks for a capability that requires private code (``fit``,
    ``save``, ``load`` with signature verify). Direct ``eval`` on an
    already-loaded tree stays in the MIT path.
    """

    def __init__(self, tree: EMLTree | None = None) -> None:
        self._tree: EMLTree | None = tree
        self._sig: bytes | None = None
        self._fit_impl: Any | None = None
        self._codec_impl: Any | None = None
        self._sign_impl: Any | None = None
        self._eval_impl: Any | None = None
        try:
            from terminals_runtime.eml import (
                codec_impl,
                eval_impl,
                fit_impl,
                sign_impl,
            )
        except ImportError:
            # Deferred — admin path will try to resolve on demand.
            return
        self._fit_impl = fit_impl
        self._codec_impl = codec_impl
        self._sign_impl = sign_impl
        self._eval_impl = eval_impl

    # -- lazy loader via admin gate ----------------------------------------

    def _resolve_runtime(self) -> None:
        if self._fit_impl is not None:
            return
        from carl_studio.admin import is_admin, load_private

        if not is_admin():
            _require_runtime("EMLHead")
        # load_private pulls individual module files. For a multi-module
        # package we expect the admin pack to shim these four.
        self._fit_impl = load_private("eml_fit")
        self._codec_impl = load_private("eml_codec")
        self._sign_impl = load_private("eml_sign")
        self._eval_impl = load_private("eml_eval")

    # -- API ----------------------------------------------------------------

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fit tree to ``(inputs, targets)``. Returns fit metrics."""
        self._resolve_runtime()
        assert self._fit_impl is not None
        tree, metrics = self._fit_impl.fit_eml(inputs, targets, **kwargs)
        self._tree = tree
        # Newly-fit tree invalidates any prior signature.
        self._sig = None
        return cast(dict[str, Any], metrics)

    def eval(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the tree on ``inputs``. Must be fitted or loaded first."""
        if self._tree is None:
            raise RuntimeError("EMLHead has no tree — call fit() or load() first")
        if self._eval_impl is not None:
            return cast(
                np.ndarray,
                self._eval_impl.eval_tree(self._tree, inputs),
            )
        # Public fallback using carl_core.eml.
        arr = np.asarray(inputs, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return self._tree.forward_batch(arr)

    def save(self, path: str | Path, user_secret: bytes | None = None) -> None:
        """Sign + serialize to ``path`` (requires private runtime)."""
        if self._tree is None:
            raise RuntimeError("EMLHead has no tree to save")
        self._resolve_runtime()
        assert self._sign_impl is not None and self._codec_impl is not None
        sig = self._sign_impl.sign_tree(self._tree, user_secret=user_secret)
        data = self._codec_impl.encode(self._tree, include_signature=True, sig=sig)
        Path(path).write_bytes(data)
        self._sig = sig

    def load(
        self, path: str | Path, user_secret: bytes | None = None
    ) -> None:
        """Load + verify signature from ``path`` (requires private runtime)."""
        self._resolve_runtime()
        assert self._sign_impl is not None and self._codec_impl is not None
        data = Path(path).read_bytes()
        tree, sig = self._codec_impl.decode(data)
        if sig is None:
            raise RuntimeError(
                f"EML file at {path} carries no signature — refusing to load"
            )
        if not self._sign_impl.verify_signature(tree, sig, user_secret=user_secret):
            raise RuntimeError(
                f"EML signature verification failed for {path} on this machine"
            )
        self._tree = tree
        self._sig = sig

    @property
    def tree(self) -> EMLTree | None:
        return self._tree


__all__ = ["EMLHead"]
