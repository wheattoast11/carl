"""Marketplace data models for carl.camp.

Pydantic models mirroring the Supabase marketplace schema.
Client stub for future wiring — not connected to live DB yet.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "MarketplaceModel", "MarketplaceAdapter",
    "MarketplaceRecipe", "MarketplaceKit",
    "MarketplaceClient",
]


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
    """Supabase marketplace client. Not wired to live DB yet."""

    def __init__(self, url: str = "", anon_key: str = ""):
        self._url = url
        self._anon_key = anon_key

    def list_models(self, public_only: bool = True) -> list[MarketplaceModel]:
        raise NotImplementedError("Marketplace not yet wired to Supabase")

    def publish_model(self, model: MarketplaceModel) -> MarketplaceModel:
        raise NotImplementedError("Marketplace not yet wired to Supabase")

    def list_recipes(self) -> list[MarketplaceRecipe]:
        raise NotImplementedError("Marketplace not yet wired to Supabase")

    def list_kits(self) -> list[MarketplaceKit]:
        raise NotImplementedError("Marketplace not yet wired to Supabase")
