"""MCP HTTP-transport authentication tests.

Covers the fix for the previously confirmed gap where the MCP server exposed
tools over streamable-http *without any authentication*. The transport now:

  * reuses the static ``API_TOKENS`` table through the SDK's supported
    ``TokenVerifier`` + ``AuthSettings`` bearer mechanism;
  * binds to a dedicated ``MCP_HOST`` (loopback by default);
  * refuses to start when auth or an explicit tenant/user scope is missing;
  * keeps stdio a local, unauthenticated pipe (unchanged).

The integration test drives the **real SDK transport** (BearerAuthBackend +
RequireAuthMiddleware + StreamableHTTP session manager) through Starlette's
TestClient, asserting ``401`` for missing/invalid tokens and success for a valid
one.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.testclient import TestClient

import mcp_server
from config import settings
from mcp_auth import (
    StaticTokenVerifier,
    build_auth_settings,
    build_token_verifier,
    http_transport_enabled,
    require_http_auth_config,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

INITIALIZE_BODY = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "0"},
    },
}
MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


@pytest.fixture
def mcp_settings():
    previous = (
        settings.MEMORY_TENANT_ID,
        settings.MEMORY_USER_ID,
        settings.API_TOKENS,
        settings.MCP_TRANSPORT,
    )
    yield
    (
        settings.MEMORY_TENANT_ID,
        settings.MEMORY_USER_ID,
        settings.API_TOKENS,
        settings.MCP_TRANSPORT,
    ) = previous


# --------------------------------------------------------------------------- #
# StaticTokenVerifier
# --------------------------------------------------------------------------- #
def test_verifier_accepts_configured_token():
    verifier = StaticTokenVerifier("tok-A:tenantA/user1")
    token = asyncio.run(verifier.verify_token("tok-A"))
    assert token is not None
    assert token.claims["tenant_id"] == "tenantA"
    assert token.claims["user_id"] == "user1"


def test_verifier_rejects_unknown_token():
    verifier = StaticTokenVerifier("tok-A:tenantA/user1")
    assert asyncio.run(verifier.verify_token("nope")) is None
    assert asyncio.run(verifier.verify_token("")) is None


def test_verifier_requires_non_empty_tokens():
    with pytest.raises(ValueError):
        StaticTokenVerifier("")


# --------------------------------------------------------------------------- #
# Auth settings / startup guards
# --------------------------------------------------------------------------- #
def test_build_auth_settings_advertises_resource_url():
    auth = build_auth_settings(host="127.0.0.1", port=8123)
    assert str(auth.resource_server_url).rstrip("/") == "http://127.0.0.1:8123"
    # Static tokens are not audience-bound; validation must be explicitly off.
    assert auth.validate_token_resource is False


def test_http_transport_enabled_tracks_setting(mcp_settings):
    settings.MCP_TRANSPORT = "stdio"
    assert http_transport_enabled() is False
    settings.MCP_TRANSPORT = "streamable-http"
    assert http_transport_enabled() is True


def test_require_http_auth_config_rejects_empty_tokens(mcp_settings):
    settings.API_TOKENS = ""
    settings.MEMORY_TENANT_ID = "tenantA"
    settings.MEMORY_USER_ID = "user1"
    with pytest.raises(RuntimeError):
        require_http_auth_config()


def test_require_http_auth_config_rejects_missing_scope(mcp_settings):
    settings.API_TOKENS = "tok-A:tenantA/user1"
    settings.MEMORY_TENANT_ID = ""
    with pytest.raises(RuntimeError):
        require_http_auth_config()
    settings.MEMORY_TENANT_ID = "tenantA"
    settings.MEMORY_USER_ID = ""
    with pytest.raises(RuntimeError):
        require_http_auth_config()


def test_require_http_auth_config_accepts_valid(mcp_settings):
    settings.API_TOKENS = "tok-A:tenantA/user1"
    settings.MEMORY_TENANT_ID = "tenantA"
    settings.MEMORY_USER_ID = "user1"
    require_http_auth_config()  # must not raise


# --------------------------------------------------------------------------- #
# stdio stays a local, unauthenticated transport
# --------------------------------------------------------------------------- #
def test_stdio_transport_has_no_auth():
    if mcp_server._HTTP_AUTH:
        pytest.skip("MCP_TRANSPORT is configured for HTTP in this environment")
    # The module-level server is built for the default (stdio) transport.
    assert mcp_server.mcp.settings.auth is None


# --------------------------------------------------------------------------- #
# Tool arguments cannot override the server-injected identity
# --------------------------------------------------------------------------- #
def test_tool_arguments_cannot_override_injected_scope(mcp_settings):
    settings.MEMORY_TENANT_ID = "tenantA"
    settings.MEMORY_USER_ID = "user1"
    assert mcp_server._scope_tenant("attacker_tenant") == "tenantA"
    assert mcp_server._scope_user("attacker_user") == "user1"


# --------------------------------------------------------------------------- #
# End-to-end: real SDK transport enforces bearer auth
# --------------------------------------------------------------------------- #
def _authenticated_app():
    server = MCPServer(
        name="ai-memory-test",
        version="1.0.0",
        auth=build_auth_settings(host="127.0.0.1", port=8099),
        token_verifier=build_token_verifier("tok-A:tenantA/user1"),
    )
    return server.streamable_http_app(
        host="127.0.0.1",
        transport_security=TransportSecuritySettings(
            allowed_hosts=["127.0.0.1"], allowed_origins=["*"]
        ),
    )


def test_http_transport_requires_bearer_token():
    # Use a loopback base_url so the SDK's DNS-rebinding protection accepts the
    # Host header (the default "testserver" host is intentionally rejected).
    with TestClient(_authenticated_app(), base_url="http://127.0.0.1") as client:
        # Missing token -> 401
        r = client.post("/mcp", json=INITIALIZE_BODY, headers=MCP_HEADERS)
        assert r.status_code == 401

        # Invalid token -> 401
        r = client.post(
            "/mcp",
            json=INITIALIZE_BODY,
            headers={**MCP_HEADERS, "Authorization": "Bearer not-valid"},
        )
        assert r.status_code == 401

        # Valid token -> authenticated (no longer 401)
        r = client.post(
            "/mcp",
            json=INITIALIZE_BODY,
            headers={**MCP_HEADERS, "Authorization": "Bearer tok-A"},
        )
        assert r.status_code == 200, r.text
        assert "serverInfo" in r.text or "result" in r.text


def test_protected_resource_metadata_route_is_exposed():
    with TestClient(_authenticated_app()) as client:
        r = client.get("/.well-known/oauth-protected-resource")
        assert r.status_code == 200
        payload = json.loads(r.text) if r.text else {}
        assert payload.get("resource")
