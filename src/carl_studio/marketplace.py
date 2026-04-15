"""Marketplace data models and client for carl.camp.

Pydantic models mirroring the Supabase marketplace schema (001_marketplace.sql).
Client calls Supabase Edge Functions via urllib (no supabase-py dependency).

Public reads (list/get) work without auth.
Writes (publish/star) require JWT from ``carl camp login``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "MarketplaceError",
    "MarketplaceModel",
    "MarketplaceAdapter",
    "MarketplaceRecipe",
    "MarketplaceKit",
    "MarketplaceClient",
]

# Valid item types accepted by Edge Functions.
_VALID_TYPES = frozenset({"models", "adapters", "recipes", "kits"})


class MarketplaceError(Exception):
    """Raised on marketplace API or network failures."""


class _Base(BaseModel):
    id: str = ""
    owner_id: str = ""
    public: bool = False
    stars: int = 0
    created_at: str = ""
    updated_at: str = ""


class MarketplaceModel(_Base):
    hub_id: str
    name: str
    base_model: str
    source_type: str = "repo"
    capability_dims: list[str] = Field(default_factory=list)
    description: str = ""
    downloads: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class MarketplaceAdapter(_Base):
    hub_id: str
    name: str
    compatible_bases: list[str] = Field(default_factory=list)
    capability_dims: list[str] = Field(default_factory=list)
    description: str = ""
    rank: int = 64
    downloads: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class MarketplaceRecipe(_Base):
    name: str
    slug: str
    spec: dict[str, Any] = Field(default_factory=dict)
    source_types: list[str] = Field(default_factory=list)
    courses_count: int = 0
    description: str = ""


class MarketplaceKit(_Base):
    name: str
    slug: str
    ingredients: list[dict[str, Any]] = Field(default_factory=list)
    molds: list[str] = Field(default_factory=list)
    gates: list[dict[str, Any]] = Field(default_factory=list)
    description: str = ""


class MarketplaceClient:
    """Supabase-backed marketplace client for carl.camp.

    Calls Edge Functions via urllib (no supabase-py dep).
    Public reads (list/get) work without auth.
    Writes (publish/star) require JWT from carl camp login.
    """

    def __init__(
        self,
        supabase_url: str = "",
        jwt: str = "",
        profile: Any | None = None,
    ) -> None:
        self._url = supabase_url
        self._jwt = jwt
        self._profile = profile

    # -- Factory constructors -------------------------------------------------

    @classmethod
    def from_db(cls) -> MarketplaceClient:
        """Create client from local auth cache (~/.carl/carl.db)."""
        from carl_studio.db import LocalDB

        db = LocalDB()
        url = db.get_config("supabase_url") or ""
        jwt = db.get_auth("jwt") or ""
        return cls(supabase_url=url, jwt=jwt)

    @classmethod
    def from_settings(cls) -> MarketplaceClient:
        """Create client from CARLSettings (no JWT -- read-only)."""
        from carl_studio.settings import CARLSettings

        s = CARLSettings.load()
        return cls(supabase_url=s.supabase_url, jwt="")

    @classmethod
    def from_camp_session(
        cls,
        session: Any,
        profile: Any | None = None,
    ) -> MarketplaceClient:
        """Create client from a resolved CampSession + optional CampProfile."""
        return cls(
            supabase_url=getattr(session, "supabase_url", "") or "",
            jwt=getattr(session, "jwt", "") or "",
            profile=profile,
        )

    @property
    def camp_profile(self) -> Any | None:
        """The resolved CampProfile, if available."""
        return self._profile

    # -- HTTP transport -------------------------------------------------------

    def _request(
        self,
        function: str,
        method: str = "GET",
        params: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make request to a Supabase Edge Function.

        Raises MarketplaceError on HTTP or network failures.
        """
        if not self._url:
            raise MarketplaceError(
                "Marketplace URL not configured. Run: carl config set supabase_url <URL>"
            )

        url = f"{self._url}/functions/v1/{function}"
        if params:
            query = urllib.parse.urlencode({k: v for k, v in params.items() if v})
            if query:
                url = f"{url}?{query}"

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._jwt:
            headers["Authorization"] = f"Bearer {self._jwt}"

        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            raise MarketplaceError(f"Marketplace API error ({e.code}): {error_body}")
        except urllib.error.URLError as e:
            raise MarketplaceError(f"Network error: {e.reason}")

    # -- Read operations (public, no auth required) ---------------------------

    def list_models(
        self,
        public_only: bool = True,
        query: str = "",
        limit: int = 20,
    ) -> list[MarketplaceModel]:
        """List models from the marketplace."""
        data = self._request(
            "marketplace-list",
            params={
                "type": "models",
                "public": str(public_only).lower(),
                "q": query,
                "limit": str(limit),
            },
        )
        return [MarketplaceModel(**item) for item in data.get("items", [])]

    def list_adapters(
        self,
        public_only: bool = True,
        query: str = "",
        limit: int = 20,
    ) -> list[MarketplaceAdapter]:
        """List adapters from the marketplace."""
        data = self._request(
            "marketplace-list",
            params={
                "type": "adapters",
                "public": str(public_only).lower(),
                "q": query,
                "limit": str(limit),
            },
        )
        return [MarketplaceAdapter(**item) for item in data.get("items", [])]

    def list_recipes(self, limit: int = 20) -> list[MarketplaceRecipe]:
        """List recipes from the marketplace."""
        data = self._request(
            "marketplace-list",
            params={
                "type": "recipes",
                "limit": str(limit),
            },
        )
        return [MarketplaceRecipe(**item) for item in data.get("items", [])]

    def list_kits(self, limit: int = 20) -> list[MarketplaceKit]:
        """List kits from the marketplace."""
        data = self._request(
            "marketplace-list",
            params={
                "type": "kits",
                "limit": str(limit),
            },
        )
        return [MarketplaceKit(**item) for item in data.get("items", [])]

    def get_model(self, model_id: str) -> MarketplaceModel | None:
        """Get a single model by ID or hub_id. Returns None if not found."""
        if not model_id:
            return None
        try:
            data = self._request(
                "marketplace-get",
                params={
                    "type": "models",
                    "id": model_id,
                },
            )
            return MarketplaceModel(**data) if data else None
        except MarketplaceError:
            return None

    def get_adapter(self, adapter_id: str) -> MarketplaceAdapter | None:
        """Get a single adapter by ID or hub_id. Returns None if not found."""
        if not adapter_id:
            return None
        try:
            data = self._request(
                "marketplace-get",
                params={
                    "type": "adapters",
                    "id": adapter_id,
                },
            )
            return MarketplaceAdapter(**data) if data else None
        except MarketplaceError:
            return None

    # -- Write operations (require JWT + PAID tier) ---------------------------

    def _require_auth(self, action: str) -> None:
        """Raise MarketplaceError if no JWT is set."""
        if not self._jwt:
            raise MarketplaceError(f"{action} requires authentication. Run: carl camp login")

    def publish_model(self, model: MarketplaceModel) -> MarketplaceModel:
        """Publish or update a model. Requires JWT + PAID tier."""
        self._require_auth("Publishing")
        data = self._request(
            "marketplace-publish",
            method="POST",
            body={
                "type": "models",
                "item": model.model_dump(exclude_defaults=True),
            },
        )
        return MarketplaceModel(**data.get("item", {}))

    def publish_adapter(self, adapter: MarketplaceAdapter) -> MarketplaceAdapter:
        """Publish or update an adapter. Requires JWT + PAID tier."""
        self._require_auth("Publishing")
        data = self._request(
            "marketplace-publish",
            method="POST",
            body={
                "type": "adapters",
                "item": adapter.model_dump(exclude_defaults=True),
            },
        )
        return MarketplaceAdapter(**data.get("item", {}))

    def publish_recipe(self, recipe: MarketplaceRecipe) -> MarketplaceRecipe:
        """Publish or update a recipe. Requires JWT + PAID tier."""
        self._require_auth("Publishing")
        data = self._request(
            "marketplace-publish",
            method="POST",
            body={
                "type": "recipes",
                "item": recipe.model_dump(exclude_defaults=True),
            },
        )
        return MarketplaceRecipe(**data.get("item", {}))

    def publish_kit(self, kit: MarketplaceKit) -> MarketplaceKit:
        """Publish or update a kit. Requires JWT + PAID tier."""
        self._require_auth("Publishing")
        data = self._request(
            "marketplace-publish",
            method="POST",
            body={
                "type": "kits",
                "item": kit.model_dump(exclude_defaults=True),
            },
        )
        return MarketplaceKit(**data.get("item", {}))

    def star(self, item_type: str, item_id: str) -> bool:
        """Star an item. Requires JWT.

        Args:
            item_type: One of "models", "adapters", "recipes", "kits".
            item_id: The item's UUID or hub_id/slug.

        Returns:
            True if the star was toggled on.
        """
        self._require_auth("Starring")
        if item_type not in _VALID_TYPES:
            raise MarketplaceError(
                f"Invalid item type {item_type!r}. "
                f"Must be one of: {', '.join(sorted(_VALID_TYPES))}"
            )
        data = self._request(
            "marketplace-star",
            method="POST",
            body={
                "type": item_type,
                "id": item_id,
            },
        )
        return bool(data.get("starred", False))
