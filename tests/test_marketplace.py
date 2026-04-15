"""Tests for carl_studio.marketplace and carl_studio.marketplace_cli."""
from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from carl_studio.marketplace import (
    MarketplaceAdapter,
    MarketplaceClient,
    MarketplaceError,
    MarketplaceKit,
    MarketplaceModel,
    MarketplaceRecipe,
)
from carl_studio.marketplace_cli import (
    create_marketplace_app,
    list_adapters,
    list_kits,
    list_models,
    list_recipes,
    publish_cmd,
    show_item,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

runner = CliRunner()


def _make_app() -> typer.Typer:
    """Build a test app with marketplace commands registered."""
    a = typer.Typer()
    mp = create_marketplace_app()
    a.add_typer(mp, name="marketplace")
    a.command(name="publish")(publish_cmd)
    return a


app = _make_app()


def _mock_response(data: dict[str, Any]) -> MagicMock:
    """Create a mock urllib response object."""
    body = json.dumps(data).encode()
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _http_error(code: int, body: str = "") -> urllib.error.HTTPError:
    """Create a mock HTTPError."""
    fp = BytesIO(body.encode()) if body else None
    return urllib.error.HTTPError(
        url="https://test.supabase.co/functions/v1/marketplace-list",
        code=code,
        msg="error",
        hdrs={},  # type: ignore[arg-type]
        fp=fp,
    )


_SAMPLE_MODEL = {
    "id": "abc-123",
    "hub_id": "wheattoast11/OmniCoder-9B",
    "name": "OmniCoder 9B",
    "base_model": "Tesslate/OmniCoder-9B",
    "source_type": "repo",
    "downloads": 42,
    "stars": 7,
    "public": True,
}

_SAMPLE_ADAPTER = {
    "id": "def-456",
    "hub_id": "wheattoast11/OmniCoder-9B-Zero-Phase2",
    "name": "Phase 2 Adapter",
    "compatible_bases": ["Tesslate/OmniCoder-9B"],
    "rank": 32,
    "downloads": 10,
    "stars": 3,
    "public": True,
}

_SAMPLE_RECIPE = {
    "id": "rec-789",
    "name": "Basic GRPO",
    "slug": "basic-grpo",
    "spec": {"method": "grpo"},
    "courses_count": 3,
    "stars": 5,
    "public": True,
}

_SAMPLE_KIT = {
    "id": "kit-012",
    "name": "Coding Agent Kit",
    "slug": "coding-agent",
    "ingredients": [{"type": "reward", "name": "tool_engagement"}],
    "molds": ["json"],
    "gates": [{"threshold": 0.9}],
    "stars": 2,
    "public": True,
}


# ---------------------------------------------------------------------------
# MarketplaceClient unit tests
# ---------------------------------------------------------------------------


class TestClientConstruction:
    def test_from_db_no_auth(self) -> None:
        """from_db with empty DB produces client with empty jwt."""
        with patch("carl_studio.db.LocalDB") as mock_db_cls:
            db = MagicMock()
            db.get_config.return_value = "https://test.supabase.co"
            db.get_auth.return_value = None
            mock_db_cls.return_value = db

            client = MarketplaceClient.from_db()

            assert client._jwt == ""
            assert client._url == "https://test.supabase.co"

    def test_direct_construction(self) -> None:
        client = MarketplaceClient(supabase_url="https://x.co", jwt="tok123")
        assert client._url == "https://x.co"
        assert client._jwt == "tok123"


class TestListModels:
    @patch("urllib.request.urlopen")
    def test_returns_models(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"items": [_SAMPLE_MODEL]})
        client = MarketplaceClient(supabase_url="https://test.supabase.co")
        models = client.list_models()
        assert len(models) == 1
        assert isinstance(models[0], MarketplaceModel)
        assert models[0].hub_id == "wheattoast11/OmniCoder-9B"
        assert models[0].downloads == 42

    @patch("urllib.request.urlopen")
    def test_empty_response(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"items": []})
        client = MarketplaceClient(supabase_url="https://test.supabase.co")
        assert client.list_models() == []

    @patch("urllib.request.urlopen")
    def test_network_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        client = MarketplaceClient(supabase_url="https://test.supabase.co")
        with pytest.raises(MarketplaceError, match="Network error"):
            client.list_models()

    @patch("urllib.request.urlopen")
    def test_http_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _http_error(500, "Internal Server Error")
        client = MarketplaceClient(supabase_url="https://test.supabase.co")
        with pytest.raises(MarketplaceError, match="500"):
            client.list_models()

    def test_no_url_configured(self) -> None:
        client = MarketplaceClient(supabase_url="")
        with pytest.raises(MarketplaceError, match="not configured"):
            client.list_models()


class TestListAdapters:
    @patch("urllib.request.urlopen")
    def test_returns_adapters(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"items": [_SAMPLE_ADAPTER]})
        client = MarketplaceClient(supabase_url="https://test.supabase.co")
        adapters = client.list_adapters()
        assert len(adapters) == 1
        assert isinstance(adapters[0], MarketplaceAdapter)
        assert adapters[0].rank == 32


class TestListRecipes:
    @patch("urllib.request.urlopen")
    def test_returns_recipes(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"items": [_SAMPLE_RECIPE]})
        client = MarketplaceClient(supabase_url="https://test.supabase.co")
        recipes = client.list_recipes()
        assert len(recipes) == 1
        assert isinstance(recipes[0], MarketplaceRecipe)
        assert recipes[0].slug == "basic-grpo"


class TestListKits:
    @patch("urllib.request.urlopen")
    def test_returns_kits(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"items": [_SAMPLE_KIT]})
        client = MarketplaceClient(supabase_url="https://test.supabase.co")
        kits = client.list_kits()
        assert len(kits) == 1
        assert isinstance(kits[0], MarketplaceKit)
        assert kits[0].slug == "coding-agent"


class TestGetModel:
    @patch("urllib.request.urlopen")
    def test_found(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response(_SAMPLE_MODEL)
        client = MarketplaceClient(supabase_url="https://test.supabase.co")
        model = client.get_model("abc-123")
        assert model is not None
        assert model.hub_id == "wheattoast11/OmniCoder-9B"

    @patch("urllib.request.urlopen")
    def test_not_found(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _http_error(404, "Not found")
        client = MarketplaceClient(supabase_url="https://test.supabase.co")
        assert client.get_model("nonexistent") is None

    def test_empty_id(self) -> None:
        client = MarketplaceClient(supabase_url="https://test.supabase.co")
        assert client.get_model("") is None


class TestGetAdapter:
    @patch("urllib.request.urlopen")
    def test_found(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response(_SAMPLE_ADAPTER)
        client = MarketplaceClient(supabase_url="https://test.supabase.co")
        adapter = client.get_adapter("def-456")
        assert adapter is not None
        assert adapter.rank == 32

    @patch("urllib.request.urlopen")
    def test_not_found(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _http_error(404, "Not found")
        client = MarketplaceClient(supabase_url="https://test.supabase.co")
        assert client.get_adapter("nonexistent") is None


class TestPublishModel:
    def test_no_jwt(self) -> None:
        client = MarketplaceClient(supabase_url="https://test.supabase.co", jwt="")
        model = MarketplaceModel(
            hub_id="user/model", name="Model", base_model="base"
        )
        with pytest.raises(MarketplaceError, match="authentication"):
            client.publish_model(model)

    @patch("urllib.request.urlopen")
    def test_success(self, mock_urlopen: MagicMock) -> None:
        published = dict(_SAMPLE_MODEL, id="new-id")
        mock_urlopen.return_value = _mock_response({"item": published})
        client = MarketplaceClient(supabase_url="https://test.supabase.co", jwt="tok")
        model = MarketplaceModel(
            hub_id="wheattoast11/OmniCoder-9B",
            name="OmniCoder 9B",
            base_model="Tesslate/OmniCoder-9B",
        )
        result = client.publish_model(model)
        assert result.id == "new-id"
        assert result.hub_id == "wheattoast11/OmniCoder-9B"

        # Verify the request was POST with auth header
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.method == "POST"
        assert "Bearer tok" in req.get_header("Authorization")


class TestPublishAdapter:
    def test_no_jwt(self) -> None:
        client = MarketplaceClient(supabase_url="https://test.supabase.co", jwt="")
        adapter = MarketplaceAdapter(hub_id="user/adapter", name="Adapter")
        with pytest.raises(MarketplaceError, match="authentication"):
            client.publish_adapter(adapter)

    @patch("urllib.request.urlopen")
    def test_success(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"item": _SAMPLE_ADAPTER})
        client = MarketplaceClient(supabase_url="https://test.supabase.co", jwt="tok")
        adapter = MarketplaceAdapter(
            hub_id="wheattoast11/OmniCoder-9B-Zero-Phase2",
            name="Phase 2 Adapter",
        )
        result = client.publish_adapter(adapter)
        assert result.hub_id == "wheattoast11/OmniCoder-9B-Zero-Phase2"


class TestStar:
    def test_no_jwt(self) -> None:
        client = MarketplaceClient(supabase_url="https://test.supabase.co", jwt="")
        with pytest.raises(MarketplaceError, match="authentication"):
            client.star("models", "abc-123")

    @patch("urllib.request.urlopen")
    def test_success(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"starred": True})
        client = MarketplaceClient(supabase_url="https://test.supabase.co", jwt="tok")
        assert client.star("models", "abc-123") is True

    @patch("urllib.request.urlopen")
    def test_unstar(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"starred": False})
        client = MarketplaceClient(supabase_url="https://test.supabase.co", jwt="tok")
        assert client.star("models", "abc-123") is False

    def test_invalid_type(self) -> None:
        client = MarketplaceClient(supabase_url="https://test.supabase.co", jwt="tok")
        with pytest.raises(MarketplaceError, match="Invalid item type"):
            client.star("invalid", "abc-123")


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLIListModels:
    @patch("carl_studio.marketplace_cli._get_client")
    def test_renders_table(self, mock_get: MagicMock) -> None:
        client = MagicMock()
        client.list_models.return_value = [MarketplaceModel(**_SAMPLE_MODEL)]
        mock_get.return_value = client

        result = runner.invoke(app, ["marketplace", "models"])
        assert result.exit_code == 0
        # Rich table renders name and hub_id (may truncate in narrow terminals)
        assert "OmniCoder" in result.output

    @patch("carl_studio.marketplace_cli._get_client")
    def test_empty(self, mock_get: MagicMock) -> None:
        client = MagicMock()
        client.list_models.return_value = []
        mock_get.return_value = client

        result = runner.invoke(app, ["marketplace", "models"])
        assert result.exit_code == 0
        assert "No models found" in result.output

    @patch("carl_studio.marketplace_cli._get_client")
    def test_network_error(self, mock_get: MagicMock) -> None:
        client = MagicMock()
        client.list_models.side_effect = MarketplaceError("Network error: Connection refused")
        mock_get.return_value = client

        result = runner.invoke(app, ["marketplace", "models"])
        assert result.exit_code == 1
        assert "carl.camp" in result.output


class TestCLIListAdapters:
    @patch("carl_studio.marketplace_cli._get_client")
    def test_renders_table(self, mock_get: MagicMock) -> None:
        client = MagicMock()
        client.list_adapters.return_value = [MarketplaceAdapter(**_SAMPLE_ADAPTER)]
        mock_get.return_value = client

        result = runner.invoke(app, ["marketplace", "adapters"])
        assert result.exit_code == 0
        assert "Phase 2 Adapter" in result.output


class TestCLIListRecipes:
    @patch("carl_studio.marketplace_cli._get_client")
    def test_renders_table(self, mock_get: MagicMock) -> None:
        client = MagicMock()
        client.list_recipes.return_value = [MarketplaceRecipe(**_SAMPLE_RECIPE)]
        mock_get.return_value = client

        result = runner.invoke(app, ["marketplace", "recipes"])
        assert result.exit_code == 0
        assert "Basic GRPO" in result.output


class TestCLIListKits:
    @patch("carl_studio.marketplace_cli._get_client")
    def test_renders_table(self, mock_get: MagicMock) -> None:
        client = MagicMock()
        client.list_kits.return_value = [MarketplaceKit(**_SAMPLE_KIT)]
        mock_get.return_value = client

        result = runner.invoke(app, ["marketplace", "kits"])
        assert result.exit_code == 0
        assert "Coding Agent Kit" in result.output


class TestCLIShow:
    @patch("carl_studio.marketplace_cli._get_client")
    def test_show_model(self, mock_get: MagicMock) -> None:
        client = MagicMock()
        client.get_model.return_value = MarketplaceModel(**_SAMPLE_MODEL)
        mock_get.return_value = client

        result = runner.invoke(app, ["marketplace", "show", "wheattoast11/OmniCoder-9B"])
        assert result.exit_code == 0
        assert "OmniCoder 9B" in result.output
        assert "Tesslate/OmniCoder-9B" in result.output

    @patch("carl_studio.marketplace_cli._get_client")
    def test_show_adapter_fallback(self, mock_get: MagicMock) -> None:
        client = MagicMock()
        client.get_model.return_value = None
        client.get_adapter.return_value = MarketplaceAdapter(**_SAMPLE_ADAPTER)
        mock_get.return_value = client

        result = runner.invoke(app, ["marketplace", "show", "wheattoast11/OmniCoder-9B-Zero-Phase2"])
        assert result.exit_code == 0
        assert "Adapter" in result.output

    @patch("carl_studio.marketplace_cli._get_client")
    def test_show_not_found(self, mock_get: MagicMock) -> None:
        client = MagicMock()
        client.get_model.return_value = None
        client.get_adapter.return_value = None
        mock_get.return_value = client

        result = runner.invoke(app, ["marketplace", "show", "nonexistent/model"])
        assert result.exit_code == 1
        assert "Not found" in result.output


class TestCLIPublish:
    @patch("carl_studio.marketplace_cli.check_tier")
    @patch("carl_studio.marketplace_cli._get_client")
    def test_publish_model(self, mock_get: MagicMock, mock_tier: MagicMock) -> None:
        from carl_studio.tier import Tier

        mock_tier.return_value = (True, Tier.PAID, Tier.PAID)
        client = MagicMock()
        published = MarketplaceModel(**{**_SAMPLE_MODEL, "id": "new-id"})
        client.publish_model.return_value = published
        mock_get.return_value = client

        result = runner.invoke(app, [
            "publish", "wheattoast11/OmniCoder-9B",
            "--type", "model",
            "--name", "OmniCoder 9B",
            "--base", "Tesslate/OmniCoder-9B",
        ])
        assert result.exit_code == 0
        assert "Published model" in result.output

    @patch("carl_studio.marketplace_cli.check_tier")
    def test_publish_blocked_free_tier(self, mock_tier: MagicMock) -> None:
        from carl_studio.tier import Tier

        mock_tier.return_value = (False, Tier.FREE, Tier.PAID)

        result = runner.invoke(app, ["publish", "user/model"])
        assert result.exit_code == 1
        assert "Paid" in result.output or "carl.camp/pricing" in result.output

    @patch("carl_studio.marketplace_cli.check_tier")
    @patch("carl_studio.marketplace_cli._get_client")
    def test_publish_adapter(self, mock_get: MagicMock, mock_tier: MagicMock) -> None:
        from carl_studio.tier import Tier

        mock_tier.return_value = (True, Tier.PAID, Tier.PAID)
        client = MagicMock()
        published = MarketplaceAdapter(**_SAMPLE_ADAPTER)
        client.publish_adapter.return_value = published
        mock_get.return_value = client

        result = runner.invoke(app, [
            "publish", "wheattoast11/OmniCoder-9B-Zero-Phase2",
            "--type", "adapter",
            "--rank", "32",
        ])
        assert result.exit_code == 0
        assert "Published adapter" in result.output

    @patch("carl_studio.marketplace_cli.check_tier")
    @patch("carl_studio.marketplace_cli._get_client")
    def test_publish_invalid_type(self, mock_get: MagicMock, mock_tier: MagicMock) -> None:
        from carl_studio.tier import Tier

        mock_tier.return_value = (True, Tier.PAID, Tier.PAID)
        mock_get.return_value = MagicMock()

        result = runner.invoke(app, [
            "publish", "user/model", "--type", "invalid"
        ])
        assert result.exit_code == 1
        assert "Invalid type" in result.output

    @patch("carl_studio.marketplace_cli.check_tier")
    @patch("carl_studio.marketplace_cli._get_client")
    def test_publish_api_error(self, mock_get: MagicMock, mock_tier: MagicMock) -> None:
        from carl_studio.tier import Tier

        mock_tier.return_value = (True, Tier.PAID, Tier.PAID)
        client = MagicMock()
        client.publish_model.side_effect = MarketplaceError(
            "Marketplace API error (401): Unauthorized"
        )
        mock_get.return_value = client

        result = runner.invoke(app, ["publish", "user/model"])
        assert result.exit_code == 1
        assert "login" in result.output.lower() or "authenticated" in result.output.lower()
