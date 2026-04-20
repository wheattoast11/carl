"""Typed config-key store over :class:`~carl_studio.db.LocalDB`.

``ConfigRegistry[T]`` wraps a string-keyed blob in ``LocalDB.config`` with
a Pydantic v2 model so callers get strong typing, validation, and a
single place to put JSON (de)serialization. No SQLite schema change —
only the value shape changes (JSON blob instead of bare scalar).

Typical use::

    class MyState(BaseModel):
        counter: int = 0

    reg = db.config_registry(MyState, namespace="carl.example")
    state = reg.with_defaults(MyState)
    state.counter += 1
    reg.set(state)

Schema mismatches surface as :class:`~carl_core.errors.CARLError` with
``code='carl.config.schema_mismatch'`` so callers can handle them
uniformly (and the standard redaction policy applies to their context).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from carl_core.errors import CARLError

if TYPE_CHECKING:  # pragma: no cover — type-only import to avoid cycles
    from carl_studio.db import LocalDB


T = TypeVar("T", bound=BaseModel)


__all__ = ["ConfigRegistry"]


class ConfigRegistry(Generic[T]):
    """Typed view over a single :class:`LocalDB` config key.

    Parameters
    ----------
    model_cls:
        Pydantic v2 ``BaseModel`` subclass. Every value passed to
        :meth:`set` must be an instance; every value returned by
        :meth:`get` is deserialized through
        :meth:`BaseModel.model_validate_json`.
    db:
        Backing :class:`~carl_studio.db.LocalDB`. The registry never
        opens its own connection; all I/O routes through the shared
        ``get_config`` / ``set_config`` path.
    namespace:
        Dotted namespace prefix (e.g. ``"carl.x402"``). Used to derive
        the storage key when ``key`` is not given.
    key:
        Optional explicit storage key. When ``None`` the key is derived
        as ``f"{namespace}.{model_cls.__name__.lower()}"``. Example:
        namespace ``"carl.x402"`` + model ``SpendState`` →
        ``"carl.x402.spendstate"``.
    """

    def __init__(
        self,
        model_cls: type[T],
        *,
        db: LocalDB,
        namespace: str,
        key: str | None = None,
    ) -> None:
        if not namespace:
            raise ValueError("ConfigRegistry: namespace must be a non-empty string")
        if key is not None and not key:
            raise ValueError("ConfigRegistry: key must be a non-empty string when provided")
        self._model_cls = model_cls
        self._db = db
        self._namespace = namespace
        self._key = key or f"{namespace}.{model_cls.__name__.lower()}"

    # -- properties ------------------------------------------------------

    @property
    def key(self) -> str:
        """The resolved storage key (namespace + model name, or explicit override)."""
        return self._key

    @property
    def namespace(self) -> str:
        """The namespace prefix supplied at construction time."""
        return self._namespace

    @property
    def model_cls(self) -> type[T]:
        """The bound Pydantic model class."""
        return self._model_cls

    # -- core api --------------------------------------------------------

    def get(self) -> T | None:
        """Return the stored value, or ``None`` when the key is absent.

        Raises
        ------
        CARLError
            With ``code='carl.config.schema_mismatch'`` when a stored
            blob exists but does not validate against ``model_cls``.
            ``context`` carries the key and model name.
        """
        raw = self._db.get_config(self._key)
        if raw is None:
            return None
        try:
            return self._model_cls.model_validate_json(raw)
        except ValidationError as exc:
            raise CARLError(
                f"stored config does not match {self._model_cls.__name__} schema",
                code="carl.config.schema_mismatch",
                context={"key": self._key, "model": self._model_cls.__name__},
                cause=exc,
            ) from exc

    def set(self, value: T) -> None:
        """Persist ``value`` under the resolved key.

        Raises
        ------
        TypeError
            When ``value`` is not an instance of the bound ``model_cls``.
        """
        if not isinstance(value, self._model_cls):
            raise TypeError(
                f"ConfigRegistry[{self._model_cls.__name__}].set expected "
                f"{self._model_cls.__name__}, got {type(value).__name__}",
            )
        self._db.set_config(self._key, value.model_dump_json())

    def clear(self) -> None:
        """Delete the stored value. No-op when absent.

        Uses the shared LocalDB connection; preserves the ``config``
        row's ``updated_at`` semantics by issuing a plain ``DELETE``.
        """
        with self._db.connect() as conn:
            conn.execute("DELETE FROM config WHERE key = ?", (self._key,))

    def with_defaults(self, factory: Callable[[], T]) -> T:
        """Return the stored value, or the factory-produced default.

        The default is *not* persisted automatically — callers that want
        to persist must call :meth:`set` explicitly. This keeps
        ``with_defaults`` side-effect free on the hot read path.

        Raises
        ------
        TypeError
            When ``factory`` does not return an instance of ``model_cls``.
        """
        stored = self.get()
        if stored is not None:
            return stored
        produced = factory()
        if not isinstance(produced, self._model_cls):
            raise TypeError(
                f"ConfigRegistry[{self._model_cls.__name__}].with_defaults "
                f"factory returned {type(produced).__name__}, "
                f"expected {self._model_cls.__name__}",
            )
        return produced
