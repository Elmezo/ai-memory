"""Bearer-token authentication for the MCP HTTP (streamable-http) transport.

The stdio transport is a local, process-bound pipe and needs no network auth.
The streamable-http transport exposes the same tools over TCP, so it must never
run unauthenticated. This module wires the MCP SDK's supported bearer-token
mechanism (``TokenVerifier`` + ``AuthSettings``) onto the existing static
``API_TOKENS`` table used by the REST API, so operators have a single credential
source and the endpoint returns ``401`` for missing/invalid tokens.

The authorized identity of every tool call remains the server-controlled scope
(``MEMORY_TENANT_ID`` / ``MEMORY_USER_ID``); agent-supplied tool arguments can
never widen it. Each HTTP MCP deployment is therefore a single authorized scope.
"""

from __future__ import annotations

import os

from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings

from auth import parse_token_scopes, validate_token_config
from config import settings

HTTP_TRANSPORTS = {"streamable-http", "http"}


def http_transport_enabled() -> bool:
    """True when the configured MCP transport is exposed over the network."""
    return settings.MCP_TRANSPORT in HTTP_TRANSPORTS


class StaticTokenVerifier(TokenVerifier):
    """Verify static ``API_TOKENS`` bearer tokens for the MCP HTTP transport.

    Returns an :class:`AccessToken` carrying the ``tenant_id``/``user_id`` scope
    as claims, or ``None`` for unknown tokens (the SDK then answers ``401``).
    """

    def __init__(self, tokens: str | None = None) -> None:
        raw = settings.API_TOKENS if tokens is None else tokens
        validate_token_config(raw, require_non_empty=True, source="API_TOKENS")
        self._table = parse_token_scopes(raw)

    async def verify_token(self, token: str) -> AccessToken | None:
        identity = self._table.get(token or "")
        if identity is None:
            return None
        return AccessToken(
            token=token,
            client_id="ai-memory-mcp",
            scopes=[],
            claims={
                "tenant_id": identity.tenant_id,
                "user_id": identity.user_id,
                "wildcard_user": identity.wildcard_user,
            },
        )


def build_auth_settings(
    host: str | None = None, port: int | None = None
) -> AuthSettings:
    """Build ``AuthSettings`` for the HTTP transport.

    We only run the static-token verifier (no authorization server), so the
    issuer/resource URLs exist solely to satisfy the SDK's protected-resource
    metadata contract. Token audience validation is intentionally disabled
    because these static tokens are not audience-bound.
    """
    base = settings.MCP_PUBLIC_URL.strip()
    if not base:
        scheme = "https" if _behind_tls_proxy() else "http"
        base = f"{scheme}://{host or settings.MCP_HOST}:{port or settings.MCP_PORT}"
    base = base.rstrip("/")
    return AuthSettings(
        issuer_url=base,
        resource_server_url=base,
        validate_token_resource=False,
    )


def _behind_tls_proxy() -> bool:
    # Honor common reverse-proxy hints so the advertised resource URL is https.
    for name in ("MCP_FORWARDED_PROTO", "X_FORWARDED_PROTO"):
        if os.getenv(name, "").strip().lower() == "https":
            return True
    return False


def build_token_verifier(tokens: str | None = None) -> StaticTokenVerifier:
    return StaticTokenVerifier(tokens)


def require_http_auth_config() -> None:
    """Validate the HTTP-transport configuration; raise ``RuntimeError`` if unsafe.

    Guards the three prerequisites for a safe network-exposed MCP server:
      1. authentication is configured (non-empty, well-formed ``API_TOKENS``);
      2. an explicit, non-empty tenant and user scope is configured.
    """
    try:
        validate_token_config(settings.API_TOKENS, require_non_empty=True, source="API_TOKENS")
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if not settings.MEMORY_TENANT_ID.strip():
        raise RuntimeError(
            "MCP_TRANSPORT=%s requires MEMORY_TENANT_ID to be set (explicit, "
            "non-empty tenant scope)" % settings.MCP_TRANSPORT
        )
    if not settings.MEMORY_USER_ID.strip():
        raise RuntimeError(
            "MCP_TRANSPORT=%s requires MEMORY_USER_ID to be set (explicit, "
            "non-empty user scope)" % settings.MCP_TRANSPORT
        )


__all__ = [
    "HTTP_TRANSPORTS",
    "StaticTokenVerifier",
    "build_auth_settings",
    "build_token_verifier",
    "http_transport_enabled",
    "require_http_auth_config",
]
