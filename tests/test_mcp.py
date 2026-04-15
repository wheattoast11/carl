"""Tests for MCP server tools — importability, auth, tier enforcement, skills."""
from __future__ import annotations

import base64
import json
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jwt(payload: dict) -> str:
    """Build a syntactically valid (unsigned) JWT for testing."""
    header = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(
        json.dumps(payload).encode()
    ).rstrip(b"=").decode()
    return f"{header}.{body}.fakesig"


def _reset_session() -> None:
    """Reset module-level _session to unauthenticated state."""
    import carl_studio.mcp.server as srv
    srv._session["jwt"] = ""
    srv._session["tier"] = "free"
    srv._session["user_id"] = ""


# ---------------------------------------------------------------------------
# Original smoke test
# ---------------------------------------------------------------------------


def test_mcp_server_importable():
    from carl_studio.mcp import mcp
    assert mcp is not None
    assert mcp.name == "carl-studio"


# ---------------------------------------------------------------------------
# authenticate
# ---------------------------------------------------------------------------


class TestAuthenticate(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _reset_session()

    async def test_authenticate_valid_jwt(self) -> None:
        import carl_studio.mcp.server as srv

        jwt = _make_jwt({
            "sub": "user-abc",
            "app_metadata": {"tier": "paid"},
        })
        result_raw = await srv.authenticate(jwt)
        result = json.loads(result_raw)

        assert result["authenticated"] is True
        assert result["tier"] == "paid"
        assert result["user_id"] == "user-abc"
        assert isinstance(result["features_available"], list)
        assert len(result["features_available"]) > 0
        # Session should be updated
        assert srv._session["tier"] == "paid"
        assert srv._session["user_id"] == "user-abc"
        assert srv._session["jwt"] == jwt

    async def test_authenticate_free_tier_jwt(self) -> None:
        import carl_studio.mcp.server as srv

        jwt = _make_jwt({"sub": "user-free", "user_metadata": {"tier": "free"}})
        result = json.loads(await srv.authenticate(jwt))

        assert result["authenticated"] is True
        assert result["tier"] == "free"
        assert srv._session["tier"] == "free"

    async def test_authenticate_pro_alias_normalizes_to_paid(self) -> None:
        import carl_studio.mcp.server as srv

        jwt = _make_jwt({"sub": "user-pro", "app_metadata": {"tier": "pro"}})
        result = json.loads(await srv.authenticate(jwt))

        assert result["tier"] == "paid"

    async def test_authenticate_enterprise_alias_normalizes_to_paid(self) -> None:
        import carl_studio.mcp.server as srv

        jwt = _make_jwt({"sub": "user-ent", "app_metadata": {"tier": "enterprise"}})
        result = json.loads(await srv.authenticate(jwt))

        assert result["tier"] == "paid"

    async def test_authenticate_invalid_jwt_two_parts(self) -> None:
        import carl_studio.mcp.server as srv

        result = json.loads(await srv.authenticate("bad.jwt"))

        assert result["authenticated"] is False
        assert "error" in result

    async def test_authenticate_invalid_jwt_not_base64(self) -> None:
        import carl_studio.mcp.server as srv

        result = json.loads(await srv.authenticate("hdr.!!!invalid!!!!!.sig"))

        assert result["authenticated"] is False
        assert "error" in result

    async def test_authenticate_empty_string(self) -> None:
        import carl_studio.mcp.server as srv

        result = json.loads(await srv.authenticate(""))

        assert result["authenticated"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# get_tier_status
# ---------------------------------------------------------------------------


class TestGetTierStatus(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _reset_session()

    async def test_get_tier_status_unauthenticated(self) -> None:
        import carl_studio.mcp.server as srv

        result = json.loads(await srv.get_tier_status())

        assert result["tier"] == "free"
        assert result["authenticated"] is False
        assert isinstance(result["features"], dict)
        # Free features should be True
        assert result["features"]["observe"] is True
        assert result["features"]["train"] is True
        # Paid features should be False
        assert result["features"]["sync.cloud"] is False
        assert result["features"]["orchestration"] is False

    async def test_get_tier_status_authenticated_paid(self) -> None:
        import carl_studio.mcp.server as srv

        # Manually set session to paid
        srv._session["jwt"] = "fake.jwt.token"
        srv._session["tier"] = "paid"
        srv._session["user_id"] = "user-123"

        result = json.loads(await srv.get_tier_status())

        assert result["tier"] == "paid"
        assert result["authenticated"] is True
        assert result["user_id"] == "user-123"
        assert result["features"]["sync.cloud"] is True
        assert result["features"]["orchestration"] is True


# ---------------------------------------------------------------------------
# get_project_state
# ---------------------------------------------------------------------------


class TestGetProjectState(unittest.IsolatedAsyncioTestCase):
    async def test_get_project_state_missing_config(self) -> None:
        import carl_studio.mcp.server as srv

        result = json.loads(await srv.get_project_state("/nonexistent/path/carl.yaml"))

        # Should not raise — returns error inside project key
        assert "project" in result
        assert "error" in result["project"]

    async def test_get_project_state_with_mocked_db(self) -> None:
        import carl_studio.mcp.server as srv

        mock_run = {
            "id": "run-001",
            "model_id": "test-model",
            "mode": "grpo",
            "status": "completed",
        }
        mock_db = MagicMock()
        mock_db.list_runs.return_value = [mock_run]
        mock_db.get_pending_sync.return_value = [{"id": 1}, {"id": 2}]

        with patch("carl_studio.mcp.server.json", wraps=json), \
             patch("carl_studio.db.LocalDB", return_value=mock_db):
            result_raw = await srv.get_project_state("/nonexistent/carl.yaml")
            result = json.loads(result_raw)

        # project will have an error (no file), but structure must be present
        assert "project" in result
        assert "last_run" in result
        assert "last_eval" in result
        assert "pending_sync" in result

    async def test_get_project_state_returns_valid_json(self) -> None:
        import carl_studio.mcp.server as srv

        # Any path — should return valid JSON even on failure
        result_raw = await srv.get_project_state("totally_missing.yaml")
        parsed = json.loads(result_raw)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# run_skill
# ---------------------------------------------------------------------------


class TestRunSkill(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _reset_session()

    async def test_run_skill_skills_unavailable(self) -> None:
        import carl_studio.mcp.server as srv

        original = srv._skills_available
        srv._skills_available = False
        try:
            result = json.loads(await srv.run_skill("observer"))
            assert "error" in result
            assert "pip install" in result["error"]
        finally:
            srv._skills_available = original

    async def test_run_skill_observer_free_tier(self) -> None:
        """ObserverSkill (free) should succeed even without auth."""
        import carl_studio.mcp.server as srv

        if not srv._skills_available:
            self.skipTest("Skills not installed")

        mock_result = MagicMock()
        mock_result.model_dump_json.return_value = json.dumps({
            "skill_name": "observer",
            "success": True,
            "badge_earned": False,
            "metrics": {},
            "message": "ok",
            "artifact": {},
        })

        mock_runner = MagicMock()
        mock_runner.get.return_value = MagicMock(
            name="observer",
            requires_tier="free",
        )
        mock_runner.run.return_value = mock_result
        mock_runner.list_skills.return_value = []

        with patch("carl_studio.mcp.server._build_skill_runner", return_value=mock_runner):
            result = json.loads(await srv.run_skill("observer", "{}"))

        assert result["success"] is True
        assert result["skill_name"] == "observer"

    async def test_run_skill_paid_requires_auth(self) -> None:
        """A paid skill without session JWT returns a tier error."""
        import carl_studio.mcp.server as srv

        if not srv._skills_available:
            self.skipTest("Skills not installed")

        mock_runner = MagicMock()
        mock_runner.get.return_value = MagicMock(
            name="grader",
            requires_tier="paid",
        )
        mock_runner.list_skills.return_value = []

        with patch("carl_studio.mcp.server._build_skill_runner", return_value=mock_runner):
            result = json.loads(await srv.run_skill("grader", "{}"))

        assert "error" in result
        assert "upgrade" in result
        assert result["current_tier"] == "free"

    async def test_run_skill_paid_succeeds_with_paid_session(self) -> None:
        """A paid skill with a paid session JWT should proceed."""
        import carl_studio.mcp.server as srv

        if not srv._skills_available:
            self.skipTest("Skills not installed")

        srv._session["tier"] = "paid"
        srv._session["jwt"] = "fake.paid.jwt"

        mock_result = MagicMock()
        mock_result.model_dump_json.return_value = json.dumps({
            "skill_name": "grader",
            "success": True,
            "badge_earned": True,
            "metrics": {"score": 0.92},
            "message": "Grade PASS",
            "artifact": {},
        })

        mock_runner = MagicMock()
        mock_runner.get.return_value = MagicMock(
            name="grader",
            requires_tier="paid",
        )
        mock_runner.run.return_value = mock_result
        mock_runner.list_skills.return_value = []

        with patch("carl_studio.mcp.server._build_skill_runner", return_value=mock_runner):
            result = json.loads(await srv.run_skill("grader", "{}"))

        assert result["success"] is True
        assert result["badge_earned"] is True

    async def test_run_skill_invalid_inputs_json(self) -> None:
        import carl_studio.mcp.server as srv

        if not srv._skills_available:
            self.skipTest("Skills not installed")

        result = json.loads(await srv.run_skill("observer", "{not valid json}"))
        assert "error" in result

    async def test_run_skill_unknown_skill_name(self) -> None:
        import carl_studio.mcp.server as srv

        if not srv._skills_available:
            self.skipTest("Skills not installed")

        mock_runner = MagicMock()
        mock_runner.get.return_value = None
        mock_runner.list_skills.return_value = []

        with patch("carl_studio.mcp.server._build_skill_runner", return_value=mock_runner):
            result = json.loads(await srv.run_skill("nonexistent_skill_xyz"))

        assert "error" in result
        assert "available" in result


# ---------------------------------------------------------------------------
# list_skills
# ---------------------------------------------------------------------------


class TestListSkills(unittest.IsolatedAsyncioTestCase):
    async def test_list_skills_unavailable(self) -> None:
        import carl_studio.mcp.server as srv

        original = srv._skills_available
        srv._skills_available = False
        try:
            result = json.loads(await srv.list_skills())
            assert "error" in result
        finally:
            srv._skills_available = original

    async def test_list_skills_returns_skill_list(self) -> None:
        import carl_studio.mcp.server as srv

        if not srv._skills_available:
            self.skipTest("Skills not installed")

        mock_skill = MagicMock()
        mock_skill.name = "observer"
        mock_skill.badge = "Observer Badge"
        mock_skill.description = "Observe the training loop"
        mock_skill.requires_tier = "free"

        mock_runner = MagicMock()
        mock_runner.list_skills.return_value = [mock_skill]

        with patch("carl_studio.mcp.server._build_skill_runner", return_value=mock_runner):
            result = json.loads(await srv.list_skills())

        assert "skills" in result
        assert len(result["skills"]) == 1
        skill = result["skills"][0]
        assert skill["name"] == "observer"
        assert skill["badge"] == "Observer Badge"
        assert skill["requires_tier"] == "free"


# ---------------------------------------------------------------------------
# get_agent_card
# ---------------------------------------------------------------------------


class TestGetAgentCard(unittest.IsolatedAsyncioTestCase):
    async def test_get_agent_card_returns_valid_json(self) -> None:
        import carl_studio.mcp.server as srv

        result_raw = await srv.get_agent_card()
        result = json.loads(result_raw)

        # CARLAgentCard required fields
        assert "name" in result
        assert result["name"] == "carl-studio"
        assert "version" in result
        assert "tier" in result
        assert "capabilities" in result
        assert isinstance(result["capabilities"], list)
        assert "skills" in result
        assert isinstance(result["skills"], list)

    async def test_get_agent_card_capabilities_coverage(self) -> None:
        import carl_studio.mcp.server as srv

        result = json.loads(await srv.get_agent_card())
        caps = result["capabilities"]
        # Core capabilities must always be present
        for expected in ("train", "eval", "observe"):
            assert expected in caps, f"Missing capability: {expected}"


# ---------------------------------------------------------------------------
# dispatch_a2a_task
# ---------------------------------------------------------------------------


class TestDispatchA2ATask(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _reset_session()

    async def test_dispatch_a2a_task_no_auth_returns_tier_error(self) -> None:
        import carl_studio.mcp.server as srv

        result = json.loads(await srv.dispatch_a2a_task("observer"))

        assert "error" in result
        assert "upgrade" in result
        assert result["current_tier"] == "free"

    async def test_dispatch_a2a_task_paid_session_succeeds(self) -> None:
        import carl_studio.mcp.server as srv

        srv._session["tier"] = "paid"
        srv._session["jwt"] = "fake.paid.jwt"

        mock_bus = MagicMock()
        mock_bus.post.return_value = "task-uuid-123"
        mock_bus.close = MagicMock()

        with patch("carl_studio.a2a.bus.LocalBus", return_value=mock_bus):
            result = json.loads(await srv.dispatch_a2a_task(
                "observer",
                inputs_json='{"run_id": "r1"}',
                sender="test-agent",
            ))

        assert result["status"] == "pending"
        assert "task_id" in result
        assert result["skill"] == "observer"
        assert result["sender"] == "test-agent"

    async def test_dispatch_a2a_task_invalid_inputs_json(self) -> None:
        import carl_studio.mcp.server as srv

        srv._session["tier"] = "paid"
        srv._session["jwt"] = "fake.paid.jwt"

        result = json.loads(await srv.dispatch_a2a_task("observer", inputs_json="{bad}"))

        assert "error" in result


# ---------------------------------------------------------------------------
# sync_data
# ---------------------------------------------------------------------------


class TestSyncData(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _reset_session()

    async def test_sync_data_no_auth_returns_tier_error(self) -> None:
        import carl_studio.mcp.server as srv

        result = json.loads(await srv.sync_data())

        assert "error" in result
        assert "upgrade" in result
        assert result["current_tier"] == "free"

    async def test_sync_data_invalid_direction(self) -> None:
        import carl_studio.mcp.server as srv

        srv._session["tier"] = "paid"
        srv._session["jwt"] = "fake.paid.jwt"

        result = json.loads(await srv.sync_data(direction="sideways"))

        assert "error" in result
        assert "sideways" in result["error"]

    async def test_sync_data_push_paid_calls_sync_push(self) -> None:
        import carl_studio.mcp.server as srv

        srv._session["tier"] = "paid"
        srv._session["jwt"] = "fake.paid.jwt"

        with patch("carl_studio.sync.push", return_value={"runs": 3}) as mock_push:
            result = json.loads(await srv.sync_data(direction="push", entity_types="runs"))

        assert result["status"] == "ok"
        assert result["direction"] == "push"
        mock_push.assert_called_once()

    async def test_sync_data_pull_paid_calls_sync_pull(self) -> None:
        import carl_studio.mcp.server as srv

        srv._session["tier"] = "paid"
        srv._session["jwt"] = "fake.paid.jwt"

        with patch("carl_studio.sync.pull", return_value={"pulled": 5}) as mock_pull:
            result = json.loads(await srv.sync_data(direction="pull", entity_types="runs"))

        assert result["status"] == "ok"
        assert result["direction"] == "pull"
        mock_pull.assert_called_once()

    async def test_sync_data_network_error_returns_failed_status(self) -> None:
        import carl_studio.mcp.server as srv

        srv._session["tier"] = "paid"
        srv._session["jwt"] = "fake.paid.jwt"

        with patch("carl_studio.sync.push", side_effect=Exception("Network timeout")):
            result = json.loads(await srv.sync_data(direction="push"))

        assert result["status"] == "failed"
        assert "Network timeout" in result["error"]


# ---------------------------------------------------------------------------
# _tier_error format contract
# ---------------------------------------------------------------------------


class TestTierErrorFormat(unittest.TestCase):
    def test_tier_error_has_required_keys(self) -> None:
        import carl_studio.mcp.server as srv
        _reset_session()

        result = json.loads(srv._tier_error("sync.cloud"))

        required_keys = {"error", "current_tier", "upgrade", "hint"}
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_tier_error_upgrade_url_is_carl_camp(self) -> None:
        import carl_studio.mcp.server as srv

        result = json.loads(srv._tier_error("experiment"))
        assert "carl.camp" in result["upgrade"]

    def test_tier_error_includes_feature_name(self) -> None:
        import carl_studio.mcp.server as srv

        result = json.loads(srv._tier_error("orchestration"))
        assert "orchestration" in result["error"]

    def test_tier_error_current_tier_reflects_session(self) -> None:
        import carl_studio.mcp.server as srv

        srv._session["tier"] = "free"
        result = json.loads(srv._tier_error("experiment"))
        assert result["current_tier"] == "free"


# ---------------------------------------------------------------------------
# _require_tier helper
# ---------------------------------------------------------------------------


class TestRequireTier(unittest.TestCase):
    def setUp(self) -> None:
        _reset_session()

    def test_require_tier_free_feature_always_allowed(self) -> None:
        import carl_studio.mcp.server as srv
        # Free session, free feature
        assert srv._require_tier("observe") is True
        assert srv._require_tier("train") is True

    def test_require_tier_paid_feature_blocked_on_free(self) -> None:
        import carl_studio.mcp.server as srv
        assert srv._require_tier("sync.cloud") is False
        assert srv._require_tier("orchestration") is False

    def test_require_tier_paid_feature_allowed_on_paid_session(self) -> None:
        import carl_studio.mcp.server as srv
        srv._session["tier"] = "paid"
        assert srv._require_tier("sync.cloud") is True
        assert srv._require_tier("orchestration") is True
        assert srv._require_tier("experiment") is True
