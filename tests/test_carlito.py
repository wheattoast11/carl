"""Tests for carl_studio.carlito -- CarlitoSpec, CarlitoRegistry, CarlitoSpawner."""

from __future__ import annotations

from pathlib import Path

import pytest

from carl_studio.carlito import (
    CarlitoRegistry,
    CarlitoSpec,
    CarlitoSpawner,
    CarlitoStatus,
)
from carl_studio.curriculum import CurriculumPhase, CurriculumTrack


class TestCarlitoStatus:
    def test_values(self) -> None:
        assert CarlitoStatus.INCUBATING.value == "incubating"
        assert CarlitoStatus.DEPLOYED.value == "deployed"

    def test_count(self) -> None:
        assert len(CarlitoStatus) == 5


class TestCarlitoSpec:
    def test_default_status(self) -> None:
        spec = CarlitoSpec(name="test", parent_model="org/model")
        assert spec.status == CarlitoStatus.INCUBATING

    def test_skills_default_empty(self) -> None:
        spec = CarlitoSpec(name="test", parent_model="org/model")
        assert spec.skills == []

    def test_created_at_populated(self) -> None:
        spec = CarlitoSpec(name="test", parent_model="org/model")
        assert spec.created_at != ""

    def test_model_dump_roundtrip(self) -> None:
        spec = CarlitoSpec(
            name="bot",
            parent_model="org/model",
            domain="coding",
            skills=["observer", "grader"],
        )
        dumped = spec.model_dump()
        restored = CarlitoSpec.model_validate(dumped)
        assert restored.name == "bot"
        assert restored.skills == ["observer", "grader"]


class TestCarlitoRegistry:
    @pytest.fixture()
    def registry(self, tmp_path: Path) -> CarlitoRegistry:
        db = tmp_path / "test_carlito.db"
        r = CarlitoRegistry(db_path=db)
        yield r
        r.close()

    def test_save_and_load_roundtrip(self, registry: CarlitoRegistry) -> None:
        spec = CarlitoSpec(name="bot", parent_model="org/model", domain="math")
        registry.save(spec)
        loaded = registry.load("bot")
        assert loaded is not None
        assert loaded.name == "bot"
        assert loaded.domain == "math"

    def test_load_nonexistent_returns_none(self, registry: CarlitoRegistry) -> None:
        assert registry.load("no-such-carlito") is None

    def test_list_all(self, registry: CarlitoRegistry) -> None:
        registry.save(CarlitoSpec(name="a", parent_model="m1"))
        registry.save(CarlitoSpec(name="b", parent_model="m2"))
        assert len(registry.list_all()) == 2

    def test_list_filtered_by_status(self, registry: CarlitoRegistry) -> None:
        registry.save(CarlitoSpec(name="a", parent_model="m1", status=CarlitoStatus.GRADUATED))
        registry.save(CarlitoSpec(name="b", parent_model="m2", status=CarlitoStatus.DORMANT))
        graduated = registry.list_all(status=CarlitoStatus.GRADUATED)
        assert len(graduated) == 1
        assert graduated[0].name == "a"

    def test_retire_sets_dormant(self, registry: CarlitoRegistry) -> None:
        registry.save(CarlitoSpec(name="bot", parent_model="m1", status=CarlitoStatus.DEPLOYED))
        assert registry.retire("bot") is True
        loaded = registry.load("bot")
        assert loaded is not None
        assert loaded.status == CarlitoStatus.DORMANT

    def test_retire_nonexistent_returns_false(self, registry: CarlitoRegistry) -> None:
        assert registry.retire("nope") is False

    def test_delete_removes(self, registry: CarlitoRegistry) -> None:
        registry.save(CarlitoSpec(name="bot", parent_model="m1"))
        assert registry.delete("bot") is True
        assert registry.load("bot") is None

    def test_save_upserts(self, registry: CarlitoRegistry) -> None:
        registry.save(CarlitoSpec(name="bot", parent_model="m1", domain="old"))
        registry.save(CarlitoSpec(name="bot", parent_model="m1", domain="new"))
        loaded = registry.load("bot")
        assert loaded is not None
        assert loaded.domain == "new"

    def test_skills_survive_roundtrip(self, registry: CarlitoRegistry) -> None:
        registry.save(
            CarlitoSpec(name="bot", parent_model="m1", skills=["observer", "grader"])
        )
        loaded = registry.load("bot")
        assert loaded is not None
        assert loaded.skills == ["observer", "grader"]


class TestCarlitoSpawner:
    def test_spawn_from_graduated_track(self) -> None:
        track = CurriculumTrack(model_id="org/model", phase=CurriculumPhase.GRADUATED)
        spec = CarlitoSpec(
            name="coder-bot",
            parent_model="org/model",
            domain="coding",
            skills=["observer", "grader"],
            status=CarlitoStatus.GRADUATED,
            curriculum_model_id="org/model",
        )
        spawner = CarlitoSpawner()
        card = spawner.spawn(spec, track)
        assert card.name == "coder-bot"
        assert "observer" in card.skills
        assert card.metadata["domain"] == "coding"

    def test_spawn_rejects_non_graduated(self) -> None:
        track = CurriculumTrack(model_id="org/model", phase=CurriculumPhase.DRILLING)
        spec = CarlitoSpec(name="test", parent_model="org/model")
        spawner = CarlitoSpawner()
        with pytest.raises(ValueError, match="must be 'graduated'"):
            spawner.spawn(spec, track)

    def test_spawn_accepts_deployed_track(self) -> None:
        track = CurriculumTrack(model_id="org/model", phase=CurriculumPhase.DEPLOYED)
        spec = CarlitoSpec(
            name="test", parent_model="org/model", status=CarlitoStatus.GRADUATED
        )
        spawner = CarlitoSpawner()
        card = spawner.spawn(spec, track)
        assert card.name == "test"

    def test_from_graduated_track_factory(self) -> None:
        track = CurriculumTrack(model_id="org/model", phase=CurriculumPhase.GRADUATED)
        spec = CarlitoSpawner.from_graduated_track(
            name="my-carlito", track=track, domain="math", skills=["grader"]
        )
        assert spec.name == "my-carlito"
        assert spec.parent_model == "org/model"
        assert spec.status == CarlitoStatus.GRADUATED
        assert spec.curriculum_model_id == "org/model"

    def test_spawn_updates_registry(self, tmp_path: Path) -> None:
        registry = CarlitoRegistry(db_path=tmp_path / "test.db")
        track = CurriculumTrack(model_id="org/model", phase=CurriculumPhase.GRADUATED)
        spec = CarlitoSpec(
            name="bot",
            parent_model="org/model",
            status=CarlitoStatus.GRADUATED,
            curriculum_model_id="org/model",
        )
        registry.save(spec)

        spawner = CarlitoSpawner(registry=registry)
        spawner.spawn(spec, track)

        reloaded = registry.load("bot")
        assert reloaded is not None
        assert reloaded.status == CarlitoStatus.DEPLOYED
        registry.close()
